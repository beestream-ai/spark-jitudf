# Validation and benchmark guide

How to run every test suite that validates the TC/cuDF backend, what each one checks, what a
passing run looks like, and how the representative benchmark is measured.

**Correctness validation and performance evaluation are separate activities.** The suites below
decide whether the integration computes what Spark computes, and on which path. They print
timings incidentally; those are not benchmark results. The benchmark at the end measures
performance on a path the suites have already shown to be correct, and checks that path again for
every timed configuration.

## Prerequisites

The artifacts from [INSTALL.md](INSTALL.md) steps 2–7:

| Artifact | Default location (`scripts/env.sh`) | Variable |
|---|---|---|
| Spark 4.0.4 | `third_party/spark-dist/spark-4.0.4-bin-hadoop3` | `SPARK_HOME` |
| RAPIDS distribution with the planner patch | `third_party/rapids-4-spark_2.13-26.08.1-tornado.jar` | `RAPIDS_JAR`'s default; used as-is only by the legacy suite |
| … plus the generalized JNI | `third_party/rapids-4-spark_2.13-26.08.1-tornado-jni.jar` | set `RAPIDS_JAR` to this for every TC/cuDF suite |
| Provider (shaded) | `spark-jitudf-plugin/target/spark-jitudf-plugin-0.1.0-SNAPSHOT-with-deps.jar` | `PLUGIN_JAR` |
| Suites and benchmark | `spark-jitudf-bench/target/spark-jitudf-bench-0.1.0-SNAPSHOT.jar` | `BENCH_JAR` |

Every GPU suite is launched by [`scripts/run-spark.sh`](../scripts/run-spark.sh) `<main class>
[args]`, which:

* holds an exclusive `flock` on `.spark-run.lock`, so two runs from this checkout cannot overlap;
* samples `nvidia-smi` every two seconds and reports `GPU-EXCLUSIVE: YES/NO` at the end. A GPU
  process whose command line does not contain this checkout's path counts as foreign. With the
  documented layout (Spark under `third_party/`), `local-cluster` executor JVMs are recognised as
  this run's. If `SPARK_HOME` is elsewhere, they are reported as foreign: check the witness log
  (`logs/gpu-witness-*.log`) for their PIDs. Non-exclusive runs are valid for correctness, not for
  timing;
* adds `spark.rapids.cudfVersionOverride=true` when the RAPIDS jar carries a cuDF version other
  than the released `26.08.0`;
* exits with `spark-submit`'s exit status. Every suite exits `1` when any check fails, so a
  failing run cannot look like a passing one.

For every TC/cuDF suite and the benchmark, first point the launcher at the distribution that
carries the generalized JNI. `scripts/env.sh` defaults `RAPIDS_JAR` to the jar *without* it:

```bash
export RAPIDS_JAR=$PWD/third_party/rapids-4-spark_2.13-26.08.1-tornado-jni.jar
```

The TC/cuDF suites need no TornadoVM. Set `NO_TORNADO_OPTS=1` for them. Without it, the launcher
adds the TornadoVM runtime's JVM module options, which require a `TORNADO_SDK`. Each suite sets
`-Dtornado.spark.backend` itself.

Each suite prints `PASS`/`FAIL` lines followed by `=== CHECKS: n, FAILURES: m ===` and
`spark-submit exit code: …`. A check that compares with the CPU runs the same query with
`spark.rapids.sql.enabled=false` in the same session and compares every row. A check about the
GPU path reads the executed physical plan.

## Unit tests and the provider corpus (no GPU)

```bash
mvn package
```

| Expected | Notes |
|---|---|
| `spark-jitudf-plugin`: `Tests run: 148, Failures: 0, Errors: 0, Skipped: 0` | Includes `SparkRegistrationTest` (CPU Spark registration), `ExecutorInvariantFieldsTest` (genuine second JVMs) and `CoverageMatrixTest`. |
| `spark-jitudf-bench`: `Tests run: 5, Failures: 0` | |
| `spark-jitudf-plugin/target/coverage-matrix.tsv` | 153 rows: 102 SAFE→GPU, 42 UNSAFE→CPU, 9 GAP→CPU. `CoverageMatrixTest` fails if a SAFE row moves to the CPU, an UNSAFE row compiles, or a GAP row starts compiling without being reclassified. See [SUPPORTED_UDFS.md](SUPPORTED_UDFS.md#the-coverage-corpus). |

`-Dcoverage.report=only` writes the matrix without asserting, which is useful when comparing two
revisions.

## Spark differential suite

```bash
NO_TORNADO_OPTS=1 ./scripts/run-spark.sh ai.beestream.jitudf.bench.CodexDifferentialTest 200000
```

`local[2]`. Registers each case through `spark.udf().register` and checks four things. The
provider's decision must match the expected eligibility. The physical plan must contain
`gpu_<name>` and no `GpuCpuBridge`/`GpuRowBasedScalaUDF` where the UDF should run on the GPU, or
keep the UDF on the CPU where it should not. Every row and null must match CPU Spark. Expected
failures must fail for the right reason: Spark CPU raises the Java exception, and the GPU run
fails with the cuDF evaluation error, explicitly not `UnsatisfiedLinkError`, NVRTC,
`NoSuchMethodError` or `CUDA_ERROR`. Covers resolution and dispatch, forwarding adapters, method
references and their null semantics, records and executor-invariant fields, conditional divisors,
LIST folds and casts.

Expected: `=== CHECKS: 259, FAILURES: 0 ===` (RTX 4070 and A100).

## cuDF integration suite

```bash
NO_TORNADO_OPTS=1 ./scripts/run-spark.sh ai.beestream.jitudf.bench.CudfTransformIntegrationTest 200000
```

`local[2]`. 1–5 input columns, INT32/INT64/FLOAT32/FLOAT64/BOOL8 outputs, nullable inputs and
outputs, the arithmetic error channel (`DIVISION_BY_ZERO`), LIST folds (`gpu_active_days`), and the
existing supported UDF shapes, each against CPU Spark with plan checks.

Expected: `=== CHECKS: 79, FAILURES: 0 ===`.

## Separate-executor suite

```bash
NO_TORNADO_OPTS=1 BENCH_MASTER='local-cluster[2,1,4096]' \
  ./scripts/run-spark.sh ai.beestream.jitudf.bench.SeparateExecutorTest 200000
```

`local-cluster` starts a standalone master and two executor JVMs separate from the driver. Executors
are given `-Dreview.factor=77`; the driver is not. The suite checks:

* at least two executor JVMs, none of them the driver;
* every executor loads the generalized cuDF binding and the provider's executor-side class, and
  none can load the TornadoVM API or runtime or sees a TornadoVM SDK;
* a compiled UDF and a LIST fold run as `gpu_…` on the executors, a declined UDF does not, and
  200,000 rows match CPU Spark;
* state decided on the executor: a `readResolve` reading the executor's property is not compiled
  and every row has the executor's value; a record whose constructor reads the property gets the
  executor's value, through the CPU fallback without a second reconstruction, in both fallback
  configurations; stable records with different constants compile and match;
* a task that fails once is retried and the result still matches;
* an executor that dies is replaced, the replacement runs tasks with no TornadoVM runtime, and a
  further query matches.

Expected: `FAILURES: 0`. **The total number of checks is not fixed.** After the executor
replacement, the suite takes a snapshot of the executors (three checks per snapshot) and repeats it
until the replacement has run tasks. The independent verification recorded 39/39 on the RTX 4070
(four snapshots) and 42/42 on the A100 (five snapshots, in each of three runs). The set of distinct
checks was identical (30) and none failed. The suite does not observe which partition compiled a
kernel or found it in cuDF's cache, and makes no claim about it.

If `SPARK_HOME` is outside the checkout, `GPU-EXCLUSIVE: NO` is expected here because the executor
JVMs are counted as foreign (see above).

## Legacy TornadoVM runtime backend

```bash
TORNADO_SDK=/path/to/tornadovm-8.2.0-cuda RAPIDS_JAR=third_party/rapids-4-spark_2.13-26.08.1-tornado.jar \
  ./scripts/run-spark.sh ai.beestream.jitudf.bench.IntegrationTest
```

The earlier backend that executes TornadoVM kernels inside executors. It needs a TornadoVM 8.2.0
SDK with the CUDA backend (`TORNADO_SDK`, used by the launcher's module options, so
`NO_TORNADO_OPTS` must be unset). It runs against the patched distribution **without** the JNI
swap. It is kept for comparison and is not part of the TC/cuDF path.

Expected: `=== CHECKS: 30, FAILURES: 0 ===`.

## Diagnostics

| Class | Use |
|---|---|
| `RapidsSmokeTest` | RAPIDS itself initializes and puts a plain query on the GPU. Run this first when a suite fails before reaching any UDF. |
| `PlanDump` | Prints the executed physical plan of a UDF query. |
| `WrapperProbe`, `ArrayUdfProbe` | What Spark hands the provider for a registered UDF. |
| `JitCacheProbe` | Cold versus warm cuDF kernel-cache behavior (`LIBCUDF_KERNEL_CACHE_PATH`). |

## Representative benchmark

```bash
NO_TORNADO_OPTS=1 ./scripts/run-spark.sh ai.beestream.jitudf.bench.CudfTransformBenchmark [collatzRows] [otherRows] [warm]
# e.g. a small run: ... CudfTransformBenchmark 2000000 2000000 3
# add the RAPIDS row-based operator as a fourth configuration:
NO_TORNADO_OPTS=1 BENCH_EXTRA_OPTS=-Dbench.rowBased=true ./scripts/run-spark.sh ai.beestream.jitudf.bench.CudfTransformBenchmark
```

Defaults: 50,000,000 Collatz rows, 20,000,000 rows for the other workloads, 3 warm iterations.

### Workloads

All integer-exact, so every configuration must return identical results:

| Workload | UDF | Query |
|---|---|---|
| Collatz steps | `CollatzUdf` (data-dependent loop) | `SELECT sum(b_collatz(id + 1)) FROM range(n)`; the 50M aggregate is asserted (8,600,532,084) |
| Murmur3 bucketing | `BucketUdf`, checked against Guava's `murmur3_32_fixed(0)` before any GPU run | salt-search style bucketing sum |
| Activity bitmap | `ActiveDaysUdf` over `array<bigint>` (LIST fold) | set-bit count per row |
| Multi-input | `BlendUdf` | three input columns |
| Nullable input | `NullDefaultingUdf` | a nullable column |

Collatz and bucketing are reconstructions of experiments described in NVIDIA's evaluation, written
from the published descriptions, not from NVIDIA's code. Their absolute times are not comparable to
NVIDIA's (different GPU, Spark 3.5.7 vs 4.0.4, different RAPIDS build).

### Configurations, kept distinct

| Label | Configuration | Path the plan must show |
|---|---|---|
| CPU Spark | `spark.rapids.sql.enabled=false` | no GPU operators |
| RAPIDS + GpuCpuBridge | RAPIDS on, provider off | `GpuCpuBridge`, no `gpu_<udf>` |
| RAPIDS row-based (optional) | RAPIDS on, provider off, `spark.rapids.sql.rowBasedUDF.enabled=true` | `GpuRowBasedScalaUDF` only |
| TC/cuDF | RAPIDS on, provider on, `cudf-transform` backend | `gpu_<udf>`, no bridge, no row-based |

A negative control first points the provider at a non-existent class and checks that the TC
configuration is then detected as not on the GPU, so the path check is shown to be able to fail.

Built-in GPU SQL alternatives are not part of this benchmark. Where an equivalent built-in
expression exists (for example Spark's `hash()` instead of a custom Murmur3 UDF), RAPIDS already
runs it on the GPU, and that is the preferable route.

### Protocol

* `local[2]` master (from `IntegrationTest.session`), 2 shuffle partitions, RAPIDS ASYNC pool,
  `spark.rapids.sql.batchSizeBytes=256MiB`.
* Per configuration: one untimed warm-up execution (JVM JIT, plan, cuDF kernel compile into its
  cache), then `warm` timed executions of `spark.sql(query).collect()`, timed with
  `System.nanoTime()` on the driver.
* Reported: the median of the timed executions, every sample, and speedups of the TC/cuDF median
  against the CPU Spark and RAPIDS-bridge medians **of the same run**.
* The executed plan of the last timed query of each configuration is checked against the table
  above. A mismatch aborts the benchmark with `UNEXPECTED PATH` and exit status 1; that sample is
  not reported as a measurement.
* Driver-side compilation time (resolve, analyse, generate, TC) is reported separately, once per
  UDF (`CompilePhases.summary()`); executor-side NVRTC compilation happens inside cuDF during the
  warm-up and is not in the timed samples.
* `LIBCUDF_KERNEL_CACHE_PATH` and the number of cached kernels before the run are printed.

### Interpreting results

Speedups against **RAPIDS + GpuCpuBridge** answer "what does this change for a RAPIDS user whose
UDF falls back". Speedups against **CPU Spark** answer a different question, and on light workloads
with many CPU cores plain CPU Spark can be faster than any GPU configuration. A GPU result from a
run with `GPU-EXCLUSIVE: NO` is not a timing. Numbers from different machines, Spark versions or
RAPIDS builds are not comparable.
