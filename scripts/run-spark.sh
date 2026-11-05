#!/bin/bash
# Launches a Spark local-mode job with the RAPIDS plugin and the TornadoVM runtime in the same JVM.
#
# The JVM flags below are not optional extras: TornadoVM is a modular runtime built with preview
# features on JDK 21, so the executor JVM that is expected to compile and launch kernels has to
# resolve its modules, open the packages its exports lists name, and allow preview class files.
# Running Spark without them does not degrade gracefully -- the plugin loads, the plan says GPU,
# and the first kernel compilation fails inside a task. Local mode means driver and executor are
# one JVM, so --driver-java-options covers both.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Refuse to start beside another Spark job, using a lock rather than a process-name match.
#
# This guard exists because its absence invalidated a whole benchmark run. The check that seemed
# obvious -- look for a running "spark-submit" -- cannot work: spark-submit is a shell script that
# execs a JVM, so once a run is properly under way no process has that name any more. A run in
# progress therefore looked like no run at all, a second one was started on top of it, the two raced
# on the same output files, and each was timing a GPU the other was also using.
#
# flock answers the real question (is another run holding the lock) instead of a proxy for it.
LOCK=$ROOT/.spark-run.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "refusing to start: another Spark run from this project holds $LOCK" >&2
  echo "a measurement that shares the machine with another measurement is not a measurement" >&2
  exit 3
fi

# NO_TORNADO_OPTS=true drops the TornadoVM JVM options. Needed for the native-UDF baseline:
# --enable-preview makes the JVM report an empty getSimpleName() for Spark's UDF-registration
# lambda, and RAPIDS decides whether to unwrap that lambda by testing whether its simple name
# contains "lambda". With the flag set the test fails, RAPIDS never looks inside, and a
# hand-written RapidsUDF is silently routed to the CPU bridge. TornadoVM needs the flag; RAPIDS
# and the native UDF do not, so a configuration that uses neither should not pay for it.
CLASS="$1"; shift

TORNADO_OPTS="--enable-preview \
-Djava.library.path=$TORNADO_SDK/lib \
--module-path $TORNADO_SDK/share/java/tornado \
-Dtornado.device.memory=256MB -Dtornado.spark.profile=${TORNADO_SPARK_PROFILE:-false} \
-Dtornado.load.api.implementation=uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph \
-Dtornado.load.runtime.implementation=uk.ac.manchester.tornado.runtime.TornadoCoreRuntime \
-Dtornado.load.tornado.implementation=uk.ac.manchester.tornado.runtime.common.Tornado \
@$TORNADO_SDK/etc/exportLists/common-exports \
--enable-native-access=tornado.runtime \
@$TORNADO_SDK/etc/exportLists/cuda-exports \
--add-modules jdk.unsupported,java.management,java.logging,jdk.jfr,java.naming,jdk.management,tornado.runtime,tornado.drivers.common,tornado.drivers.cuda \
${BENCH_EXTRA_OPTS:-}"

# Watch the GPU for anyone else while this run is in flight.
#
# The lock above only stops a second run launched through this script. It cannot see another tool,
# another agent, or a hand-typed spark-submit. So rather than assume exclusivity, sample it: every
# two seconds, list the processes holding a CUDA context and flag any whose command line is not
# this project's. A run that shares the GPU is still useful for checking correctness; its timings
# are not, and the verdict at the end says which kind of run it was.
mkdir -p "$ROOT/logs"
WITNESS="${BENCH_WITNESS:-$ROOT/logs/gpu-witness-$(date +%Y%m%d-%H%M%S).log}"
TRANSIENT="${WITNESS%.log}-transient.log"
: > "$WITNESS"
: > "$TRANSIENT"
(
  while true; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
      if [ -z "$cmd" ]; then
        # The PID is gone by the time /proc is read. nvidia-smi reports a compute app for a short
        # window around context creation and teardown, so this is routinely our own JVM caught
        # mid-transition rather than someone else's work. Sampling the same runs at 0.3s showed
        # exactly one process on the GPU throughout -- our java, child of run-spark.sh -- so an
        # unidentifiable PID is recorded as transient and does NOT by itself void the timings.
        # A process that is readable and is not this project still does.
        echo "$(date +%H:%M:%S) transient pid=$pid exited-before-identification" >> "$TRANSIENT"
      elif ! printf '%s' "$cmd" | grep -qF "$ROOT"; then
        echo "$(date +%H:%M:%S) foreign pid=$pid ${cmd:0:120}" >> "$WITNESS"
      fi
    done
    sleep 2
  done
) &
WITNESS_PID=$!
trap 'kill $WITNESS_PID 2>/dev/null || true' EXIT

# set -e off around the run itself: a failing job's exit code is a result to report, not a reason
# to abandon the script before it can say whether the GPU was shared. The reported code is the
# job's own, printed explicitly, because a launcher that returns 1 for a run whose checks all
# passed is indistinguishable from a run that failed.
# NO_TORNADO_OPTS=true: run without the TornadoVM JVM options (see the note above CLASS=).
if [ -n "${NO_TORNADO_OPTS:-}" ]; then
  DRIVER_OPTS="${BENCH_EXTRA_OPTS:-}"
  echo "note: running WITHOUT TornadoVM JVM options (no --enable-preview)"
else
  DRIVER_OPTS="$TORNADO_OPTS"
fi

set +e
# PROFILE_WRAP lets a profiler (nsys, ncu) wrap the whole submit without a second copy of this
# launcher. Unset by default, so an ordinary run is byte-for-byte the command it always was; a
# profiled run is never reported as a timing, because profiling perturbs what it measures.
# One string decides the master, and the application is told what it was so it can check.
# The application no longer sets a master of its own; if Spark builds a session that disagrees
# with this value, Benchmark fails the run rather than recording it under the wrong label.
EFFECTIVE_MASTER="${BENCH_MASTER:-local[2]}"
DRIVER_OPTS="$DRIVER_OPTS -Dapp.expectMaster=$EFFECTIVE_MASTER"
# Identity of the code and artifacts that produce each measurement, so a saved number can be tied
# to the bytes that made it. HEAD alone does not describe a dirty working tree, which this one is.
DRIVER_OPTS="$DRIVER_OPTS -Dapp.repoRoot=$ROOT -Dapp.integrationJar=$PLUGIN_JAR -Dapp.harnessJar=$BENCH_JAR"

${PROFILE_WRAP:-} "$SPARK_HOME/bin/spark-submit" \
  --master "$EFFECTIVE_MASTER" \
  --class "$CLASS" \
  --jars "$RAPIDS_JAR,$PLUGIN_JAR${NATIVE_UDF_JAR:+,$NATIVE_UDF_JAR}" \
  --conf spark.driver.memory=6g \
  --driver-java-options "$DRIVER_OPTS" \
  "${SUBMIT_JAR:-$BENCH_JAR}" "$@"
RC=$?
set -e
echo "spark-submit exit code: $RC"

kill $WITNESS_PID 2>/dev/null || true
FOREIGN=$(wc -l < "$WITNESS" | tr -d ' ')
TRANS=$(wc -l < "$TRANSIENT" 2>/dev/null | tr -d ' ' || echo 0)
[ "$TRANS" -gt 0 ] && echo "  (note: $TRANS transient PID(s) seen on the GPU and gone before identification; see $TRANSIENT)"
[ "$TRANS" -eq 0 ] && rm -f "$TRANSIENT"
if [ "$FOREIGN" -gt 0 ]; then
  echo ""
  echo "GPU-EXCLUSIVE: NO  ($FOREIGN samples saw another process on the GPU)"
  echo "  Timings from this run are not trustworthy. Correctness results still are."
  echo "  Witness log: $WITNESS"
  sort -u -k3 "$WITNESS" | head -5 | sed 's/^/  /'
else
  echo ""
  echo "GPU-EXCLUSIVE: YES  (no other process held a CUDA context during this run)"
  rm -f "$WITNESS"
fi
exit $RC
