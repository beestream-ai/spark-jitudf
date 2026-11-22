#!/bin/bash
# Puts a locally built cuDF JNI artifact inside the RAPIDS distribution jar, replacing the one
# NVIDIA shipped in it.
#
# WHY NOT JUST PUT THE NEW JAR FIRST ON THE CLASS PATH
#
#   Because two jars would then carry `ai/rapids/cudf/ColumnView.class` and
#   `amd64/Linux/libcudf.so`, and which one wins would be decided by class path order -- for the
#   classes, for the native library, and for the two independently. A mismatch between them is not
#   a loud failure: the Java class is found, the native symbol it calls is not, and the query dies
#   inside a task with UnsatisfiedLinkError. RAPIDS also refuses to start when it finds
#   `spark-rapids-jni-version-info.properties` in more than one jar, which is the same hazard seen
#   from its side.
#
#   One jar, one copy of cudf, one libcudf.so. There is then nothing to order.
#
# WHAT IT DOES
#
#   Every path the new JNI jar provides is taken from the new JNI jar; everything else is taken
#   from the distribution jar unchanged. That is the same substitution the RAPIDS dist build makes
#   when it unpacks spark-rapids-jni into its parallel world, done to an already-built jar.
#
# USAGE
#
#   scripts/swap-cudf-jni.sh [jni-jar] [in-jar] [out-jar]
#
#   Defaults: the 26.08.0-tcudf artifact from the local Maven repository, the project's patched
#   RAPIDS jar, and that jar with `-jni` before the extension.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

CUDF_JNI_VERSION=${CUDF_JNI_VERSION:-26.08.0-tcudf}

# The classifier is not always cuda12. The build appends the host architecture on aarch64, so the
# same version is spark-rapids-jni-<v>-cuda12.jar on x86-64 and spark-rapids-jni-<v>-cuda12-arm64.jar
# on ARM, and a CUDA 13 build says cuda13. Resolving it beats assuming it: hard-coding cuda12 here
# would have made this script silently wrong on the second machine it ever ran on.
resolve_jni_jar() {
  local dir=$HOME/.m2/repository/com/nvidia/spark-rapids-jni/$CUDF_JNI_VERSION
  [ -d "$dir" ] || return 1
  local found=()
  local f
  for f in "$dir"/spark-rapids-jni-"$CUDF_JNI_VERSION"-*.jar; do
    # -tests.jar and -sources.jar are not the artifact.
    case "$f" in *-tests.jar|*-sources.jar|*'*'*) continue ;; esac
    found+=("$f")
  done
  case ${#found[@]} in
    0) return 1 ;;
    1) printf '%s\n' "${found[0]}" ;;
    *) echo "several candidate JNI jars in $dir; set CUDF_JNI_JAR to the one you mean:" >&2
       printf '  %s\n' "${found[@]}" >&2
       return 1 ;;
  esac
}

JNI_JAR=${1:-${CUDF_JNI_JAR:-$(resolve_jni_jar || true)}}
IN_JAR=${2:-$RAPIDS_JAR}
OUT_JAR=${3:-${IN_JAR%.jar}-jni.jar}

[ -n "$JNI_JAR" ] && [ -f "$JNI_JAR" ] || {
  echo "no JNI jar found for version $CUDF_JNI_VERSION" >&2
  echo "build one with cudf-spark's upstream-patches/build-cudf-spark-jni.sh, or set CUDF_JNI_JAR" >&2
  exit 2
}
[ -f "$IN_JAR" ] || { echo "no RAPIDS jar at $IN_JAR" >&2; exit 2; }

echo "jni : $JNI_JAR"
echo "in  : $IN_JAR"
echo "out : $OUT_JAR"

# VERIFY_ONLY=1 re-runs the checks at the bottom against an existing output jar. Rewriting a
# gigabyte takes minutes, and the checks are what one actually wants to repeat.
if [ "${VERIFY_ONLY:-0}" = "1" ]; then
  [ -f "$OUT_JAR" ] || { echo "no jar to verify at $OUT_JAR" >&2; exit 2; }
  echo "(verify only: the jar is not rebuilt)"
else

python3 - "$JNI_JAR" "$IN_JAR" "$OUT_JAR" <<'PY'
import sys, zipfile, os

jni_path, in_path, out_path = sys.argv[1:4]
tmp = out_path + ".part"

with zipfile.ZipFile(jni_path) as jni:
    provided = {i.filename for i in jni.infolist()}
    print(f"the JNI jar provides {len(provided)} entries")

    with zipfile.ZipFile(in_path) as src, zipfile.ZipFile(tmp, "w", allowZip64=True) as out:
        replaced = kept = 0
        for info in src.infolist():
            if info.filename in provided:
                replaced += 1
                continue
            # Large members are stored rather than recompressed: the input is already compressed,
            # and deflating a gigabyte again to save nothing costs minutes.
            method = zipfile.ZIP_STORED if info.file_size > 8 << 20 else zipfile.ZIP_DEFLATED
            with src.open(info) as f:
                out.writestr(info.filename, f.read(), compress_type=method)
            kept += 1
        added = 0
        for info in jni.infolist():
            if info.is_dir():
                continue
            method = zipfile.ZIP_STORED if info.file_size > 8 << 20 else zipfile.ZIP_DEFLATED
            with jni.open(info) as f:
                out.writestr(info.filename, f.read(), compress_type=method)
            added += 1
        print(f"kept {kept}, dropped {replaced}, added {added}")

os.replace(tmp, out_path)
PY

fi

echo
echo "=== what the result carries ==="
unzip -p "$OUT_JAR" spark-rapids-jni-version-info.properties | sed 's/^/  jni  /'
unzip -p "$OUT_JAR" cudf-java-version-info.properties | sed 's/^/  cudf /'

# Listed once into a file rather than piped. `unzip -l | grep -q` looks right and is not: grep -q
# exits at the first match, unzip takes SIGPIPE, and under `set -o pipefail` the pipeline then
# reports failure precisely when the thing being looked for was found.
LISTING=$(mktemp)
trap 'rm -f "$LISTING"' EXIT
unzip -l "$OUT_JAR" > "$LISTING"

# The generalized binding has to be in there, or the swap silently did nothing useful.
if grep -q "ai/rapids/cudf/OutputNullability.class" "$LISTING"; then
  echo "  OutputNullability.class present: the generalized transform binding is in this jar"
else
  echo "  ERROR: OutputNullability.class is missing; this jar does not carry the new binding" >&2
  exit 3
fi
# And exactly once each, which is the failure the whole exercise is about.
for prop in spark-rapids-jni-version-info.properties cudf-java-version-info.properties; do
  n=$(grep -c " $prop$" "$LISTING" || true)
  [ "$n" -eq 1 ] || { echo "  ERROR: $prop appears $n times" >&2; exit 3; }
done
echo "  one copy each of the cudf classes, the native library and the version stamps"
echo
echo "use it with:  export RAPIDS_JAR=$OUT_JAR"
