#!/bin/bash
# Exercises swap-cudf-jni.sh on small fabricated jars: a distribution and a JNI artifact shaped like
# the real ones (version stamps, a native directory with ELF headers, the new JNI symbol, the cuDF
# classes, each side's own META-INF). Each case checks the exit code and whether an output appeared.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
trap '[ -n "${KEEP:-}" ] || rm -rf -- "$WORK"' EXIT
PASS=0; FAIL=0
check() { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS  $1"; else FAIL=$((FAIL + 1)); echo "  FAIL  $1"; fi; }

# make_jar <out> <arch> <revision> <with-symbol> <extra-lib> <manifest-text>
make_jar() {
  python3 - "$@" <<'PY'
import sys, zipfile
out, arch, revision, symbol, extra, manifest = sys.argv[1:7]
machine = {"amd64": 62, "aarch64": 183}[arch]
def elf(body=b""):
    head = bytearray(b"\x7fELF" + b"\x02\x01\x01" + bytes(11))
    head += machine.to_bytes(2, "little")
    return bytes(head) + body
with zipfile.ZipFile(out, "w") as z:
    z.writestr("META-INF/MANIFEST.MF", manifest)
    z.writestr("META-INF/LICENSE", manifest + " licence")
    z.writestr("ai/rapids/cudf/ColumnView.class", "columnview " + revision + manifest)
    z.writestr("ai/rapids/cudf/OutputNullability.class", "nullability")
    z.writestr(f"{arch}/Linux/libcudfjni.so", elf(b"stub"))
    z.writestr(f"{arch}/Linux/libcudf.so", elf(b"cudf " + manifest.encode() + (b" Java_ai_rapids_cudf_ColumnView_transformMulti" if symbol == "yes" else b"")))
    if extra == "yes":
        z.writestr(f"{arch}/Linux/libextra.so", elf())
    z.writestr("spark-rapids-jni-version-info.properties", f"version=v\nrevision={revision}\n")
    z.writestr("cudf-java-version-info.properties", f"version=v\nrevision={revision}\n")
    if manifest == "dist":
        z.writestr("com/nvidia/spark/SQLPlugin.class", "plugin")
        z.writestr("META-INF/NOTICE", "dist notice")
PY
}

run() { CUDF_JNI_JAR="$1" TARGET_ARCH=amd64 RAPIDS_JAR="$2" bash "$HERE/swap-cudf-jni.sh" "$1" "$2" "$3" > "$WORK/log" 2>&1; }

make_jar "$WORK/dist.jar" amd64 r1 no no dist
make_jar "$WORK/jni.jar" amd64 r1 yes no jni

echo "--- a matching JNI artifact"
run "$WORK/jni.jar" "$WORK/dist.jar" "$WORK/out.jar"
check "succeeds" "[ $? = 0 ]"
check "the output exists, with provenance" "[ -f $WORK/out.jar ] && [ -f $WORK/out.jar.provenance ]"
check "no .part is left" "[ ! -e $WORK/out.jar.part ]"
check "the distribution's manifest is kept" "[ \"\$(unzip -p $WORK/out.jar META-INF/MANIFEST.MF)\" = dist ]"
check "the JNI's native library is in" "unzip -p $WORK/out.jar amd64/Linux/libcudf.so | grep -aq transformMulti"
check "the plugin's own classes are kept" "unzip -l $WORK/out.jar | grep -q SQLPlugin.class"

echo "--- a JNI artifact for the other architecture"
make_jar "$WORK/jni-arm.jar" aarch64 r1 yes no jni
run "$WORK/jni-arm.jar" "$WORK/dist.jar" "$WORK/out-arm.jar"
check "is refused" "[ $? != 0 ]"
check "no output is written" "[ ! -e $WORK/out-arm.jar ] && [ ! -e $WORK/out-arm.jar.part ]"
check "the reason names the architecture" "grep -q 'not exactly amd64' $WORK/log"

echo "--- an ELF header that disagrees with its directory"
python3 - "$WORK/jni.jar" "$WORK/jni-lying.jar" <<'PY'
import sys, zipfile
src, dst = sys.argv[1:3]
with zipfile.ZipFile(src) as s, zipfile.ZipFile(dst, "w") as d:
    for i in s.infolist():
        data = s.read(i)
        if i.filename.endswith("libcudf.so"):
            data = data[:18] + (183).to_bytes(2, "little") + data[20:]
        d.writestr(i, data)
PY
run "$WORK/jni-lying.jar" "$WORK/dist.jar" "$WORK/out-lying.jar"
check "is refused" "[ $? != 0 ] && [ ! -e $WORK/out-lying.jar ]"

echo "--- a different set of native libraries (a stale one would stay behind)"
make_jar "$WORK/jni-extra.jar" amd64 r1 yes yes jni
run "$WORK/jni-extra.jar" "$WORK/dist.jar" "$WORK/out-extra.jar"
check "is refused" "[ $? != 0 ] && [ ! -e $WORK/out-extra.jar ]"

echo "--- a JNI artifact built from a different revision"
make_jar "$WORK/jni-r2.jar" amd64 r2 yes no jni
run "$WORK/jni-r2.jar" "$WORK/dist.jar" "$WORK/out-r2.jar"
check "is refused" "[ $? != 0 ] && [ ! -e $WORK/out-r2.jar ]"

echo "--- a native library without the new binding's symbol"
make_jar "$WORK/jni-nosym.jar" amd64 r1 no no jni
run "$WORK/jni-nosym.jar" "$WORK/dist.jar" "$WORK/out-nosym.jar"
check "is refused" "[ $? != 0 ] && [ ! -e $WORK/out-nosym.jar ]"

echo "--- verify-only on a jar whose manifest was replaced (the earlier behaviour)"
python3 - "$WORK/out.jar" "$WORK/old-style.jar" <<'PY'
import sys, zipfile
src, dst = sys.argv[1:3]
with zipfile.ZipFile(src) as s, zipfile.ZipFile(dst, "w") as d:
    for i in s.infolist():
        d.writestr(i, b"jni" if i.filename == "META-INF/MANIFEST.MF" else s.read(i))
PY
VERIFY_ONLY=1 run "$WORK/jni.jar" "$WORK/dist.jar" "$WORK/old-style.jar"
check "is reported" "[ $? != 0 ] && grep -q 'MANIFEST.MF is not the distribution' $WORK/log"

echo
echo "=== CHECKS: $((PASS + FAIL)), FAILURES: $FAIL ==="
[ "$FAIL" = 0 ]
