#!/bin/bash
# Exercises swap-cudf-jni.sh on small jars shaped like the real ones, built from real artifacts: the
# native libraries are shared objects compiled here with cc, and the cuDF classes are compiled here
# with javac, so that what is checked is a real dynamic symbol table and a real native-method
# declaration -- not a fake ELF header followed by a symbol's name as bytes, which is what the
# previous version of this test supplied and the previous check accepted.
#
# Each case checks the exit code, whether an output appeared, and (for refusals) the reason.
# Needs cc and javac for the host; the host architecture is the target.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
trap '[ -n "${KEEP:-}" ] || rm -rf -- "$WORK"' EXIT
PASS=0; FAIL=0
check() { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS  $1"; else FAIL=$((FAIL + 1)); echo "  FAIL  $1"; fi; }

case "$(uname -m)" in x86_64) ARCH=amd64 OTHER=aarch64 ;; aarch64) ARCH=aarch64 OTHER=amd64 ;; *) echo "unsupported host"; exit 2 ;; esac
command -v cc > /dev/null && command -v javac > /dev/null || { echo "this test needs cc and javac"; exit 2; }

# --- the Java half: ColumnView with the generalized binding, and the old one -----------------------
mkdir -p "$WORK/java-new/ai/rapids/cudf" "$WORK/java-old/ai/rapids/cudf"
cat > "$WORK/java-new/ai/rapids/cudf/ColumnView.java" <<'JAVA'
package ai.rapids.cudf;
public class ColumnView {
  static native long transform(long view, String udf, boolean isPtx);
  static native long transformMulti(long[] views, String udf, boolean isPtx, int outType, int nullability);
  static native long bitwise_op(long a, long b);
  static native long concat(long a);
  static native long concat(long a, long b);
}
JAVA
printf 'package ai.rapids.cudf;\npublic enum OutputNullability { PRESERVE, ALL_VALID }\n' > "$WORK/java-new/ai/rapids/cudf/OutputNullability.java"
cat > "$WORK/java-old/ai/rapids/cudf/ColumnView.java" <<'JAVA'
package ai.rapids.cudf;
public class ColumnView {
  static native long transform(long view, String udf, boolean isPtx);
  static native long bitwise_op(long a, long b);
  static native long concat(long a);
  static native long concat(long a, long b);
}
JAVA
javac -d "$WORK/classes-new" "$WORK"/java-new/ai/rapids/cudf/*.java
javac -d "$WORK/classes-old" "$WORK"/java-old/ai/rapids/cudf/ColumnView.java

# --- the native half: one shared object per variant of the transformMulti symbol -----------------
# The other natives are always exported correctly: transform and bitwise_op by their short names
# (bitwise_op exercises the _1 escape), and the overloaded concat by its long names.
common='
void Java_ai_rapids_cudf_ColumnView_transform(void) {}
void Java_ai_rapids_cudf_ColumnView_bitwise_1op(void) {}
void Java_ai_rapids_cudf_ColumnView_concat__J(void) {}
void Java_ai_rapids_cudf_ColumnView_concat__JJ(void) {}
const char toolkit[] = "Cuda compilation tools, release CUDA_RELEASE, V0";
'
lib() { # lib <name> <cuda-release> <extra C source>
  printf '%s\n%s\n' "${common/CUDA_RELEASE/$2}" "$3" > "$WORK/$1.c"
  cc -shared -fPIC -O0 -o "$WORK/$1.so" "$WORK/$1.c" || { echo "cc failed for $1"; exit 2; }
}
lib exported 12.9 'void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {}'
lib old 12.9 ''
lib cuda13 13.0 'void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {}'
lib text-only 12.9 'const char note[] = "Java_ai_rapids_cudf_ColumnView_transformMulti";'
lib undefined 12.9 'extern void Java_ai_rapids_cudf_ColumnView_transformMulti(void); void caller(void) { Java_ai_rapids_cudf_ColumnView_transformMulti(); }'
lib hidden 12.9 '__attribute__((visibility("hidden"))) void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {}'
lib local 12.9 'static void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {} void (*keep)(void) = Java_ai_rapids_cudf_ColumnView_transformMulti;'
lib missing 12.9 ''
lib short-overload 12.9 'void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {}' # rebuilt below without the long concat names
printf '%s\n' 'void Java_ai_rapids_cudf_ColumnView_transform(void) {}
void Java_ai_rapids_cudf_ColumnView_bitwise_1op(void) {}
void Java_ai_rapids_cudf_ColumnView_concat(void) {}
void Java_ai_rapids_cudf_ColumnView_transformMulti(void) {}
const char toolkit[] = "Cuda compilation tools, release 12.9, V0";' > "$WORK/short-overload.c"
cc -shared -fPIC -O0 -o "$WORK/short-overload.so" "$WORK/short-overload.c"
cc -shared -fPIC -O0 -o "$WORK/jnistub.so" -x c - <<< 'void jni_stub(void) {}'

# make_jar <out> <arch-dir> <libcudf.so> <classes-dir> <revision> <manifest> [extra-lib]
make_jar() {
  python3 - "$@" <<'PY'
import os, sys, zipfile
out, arch, libcudf, classes, revision, manifest = sys.argv[1:7]
extra = sys.argv[7] if len(sys.argv) > 7 else ""
stub = os.path.join(os.path.dirname(libcudf), "jnistub.so")
with zipfile.ZipFile(out, "w") as z:
    z.writestr("META-INF/MANIFEST.MF", manifest)
    z.writestr("META-INF/LICENSE", manifest + " licence")
    for root, _, files in os.walk(classes):
        for f in files:
            path = os.path.join(root, f)
            z.write(path, os.path.relpath(path, classes))
    z.write(libcudf, f"{arch}/Linux/libcudf.so")
    z.write(stub, f"{arch}/Linux/libcudfjni.so")
    if extra:
        z.write(stub, f"{arch}/Linux/{extra}")
    z.writestr("spark-rapids-jni-version-info.properties", f"version=v\nrevision={revision}\n")
    z.writestr("cudf-java-version-info.properties", f"version=v\nrevision={revision}\n")
    if manifest == "dist":
        z.writestr("com/nvidia/spark/SQLPlugin.class", "plugin")
        z.writestr("META-INF/NOTICE", "dist notice")
PY
}

run() { CUDF_JNI_JAR="$1" TARGET_ARCH=$ARCH RAPIDS_JAR="$2" bash "$HERE/swap-cudf-jni.sh" "$1" "$2" "$3" > "$WORK/log" 2>&1; }

refused() { # refused <label> <jni-jar> <expected reason>
  local out=$WORK/out-$RANDOM.jar
  run "$1" "$WORK/dist.jar" "$out"
  local rc=$?
  check "$3: refused" "[ $rc != 0 ]"
  check "$3: no output, no .part" "[ ! -e $out ] && [ ! -e $out.part ]"
  check "$3: the reason says '$2'" "grep -qF -- \"$2\" $WORK/log"
}

make_jar "$WORK/dist.jar" "$ARCH" "$WORK/old.so" "$WORK/classes-old" r1 dist

echo "--- a matching JNI artifact with every native method exported"
make_jar "$WORK/jni.jar" "$ARCH" "$WORK/exported.so" "$WORK/classes-new" r1 jni
run "$WORK/jni.jar" "$WORK/dist.jar" "$WORK/out.jar"
check "succeeds" "[ $? = 0 ]"
check "reports every native method linked" "grep -q 'all 5 native methods resolve' $WORK/log"
check "the output exists, with provenance" "[ -f $WORK/out.jar ] && [ -f $WORK/out.jar.provenance ]"
check "no .part is left" "[ ! -e $WORK/out.jar.part ]"
check "the distribution's manifest is kept" "[ \"\$(unzip -p $WORK/out.jar META-INF/MANIFEST.MF)\" = dist ]"
check "the JNI's native library is in" "cmp -s <(unzip -p $WORK/out.jar $ARCH/Linux/libcudf.so) $WORK/exported.so"
check "the plugin's own classes are kept" "unzip -l $WORK/out.jar | grep -q SQLPlugin.class"

echo "--- Codex's fixture: a fake ELF header followed by the symbol as bytes"
python3 - "$WORK/fake.so" <<'PY'
import sys
machine = {"x86_64": 62, "aarch64": 183}[__import__("os").uname().machine]
head = bytearray(b"\x7fELF" + b"\x02\x01\x01" + bytes(11)) + machine.to_bytes(2, "little")
open(sys.argv[1], "wb").write(bytes(head) + b" Java_ai_rapids_cudf_ColumnView_transformMulti Cuda compilation tools, release 12.9")
PY
make_jar "$WORK/jni-fake.jar" "$ARCH" "$WORK/fake.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-fake.jar" "no readable ELF dynamic symbol table" "fake ELF with the symbol as text"

echo "--- the symbol present only as text in a real library"
make_jar "$WORK/jni-text.jar" "$ARCH" "$WORK/text-only.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-text.jar" "present only as text" "text only"

echo "--- the symbol present only as an undefined reference"
make_jar "$WORK/jni-undef.jar" "$ARCH" "$WORK/undefined.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-undef.jar" "an undefined reference" "undefined only"

echo "--- the symbol defined but hidden"
make_jar "$WORK/jni-hidden.jar" "$ARCH" "$WORK/hidden.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-hidden.jar" "not exported" "hidden"

echo "--- the symbol defined but local"
make_jar "$WORK/jni-local.jar" "$ARCH" "$WORK/local.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-local.jar" "not exported" "local"

echo "--- the symbol missing altogether"
make_jar "$WORK/jni-missing.jar" "$ARCH" "$WORK/missing.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-missing.jar" "transformMulti" "missing"

echo "--- an overloaded native exported only by its short name (the JVM needs the long name)"
make_jar "$WORK/jni-short.jar" "$ARCH" "$WORK/short-overload.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-short.jar" "is overloaded" "overload by short name"

echo "--- Java and native halves disagree: the classes are the old binding"
make_jar "$WORK/jni-oldjava.jar" "$ARCH" "$WORK/exported.so" "$WORK/classes-old" r1 jni
refused "$WORK/jni-oldjava.jar" "declares no native transformMulti" "old Java half"

echo "--- a JNI artifact for the other architecture"
make_jar "$WORK/jni-arm.jar" "$OTHER" "$WORK/exported.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-arm.jar" "not exactly $ARCH" "other architecture directory"

echo "--- an ELF header that disagrees with its directory"
python3 - "$WORK/exported.so" "$WORK/lying.so" "$OTHER" <<'PY'
import sys
src, dst, other = sys.argv[1:4]
data = open(src, "rb").read()
machine = {"amd64": 62, "aarch64": 183}[other]
open(dst, "wb").write(data[:18] + machine.to_bytes(2, "little") + data[20:])
PY
make_jar "$WORK/jni-lying.jar" "$ARCH" "$WORK/lying.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-lying.jar" "is not an ELF $ARCH library" "wrong ELF machine"

echo "--- a CUDA 13 artifact for a CUDA 12 distribution (the classifier)"
make_jar "$WORK/jni-cuda13.jar" "$ARCH" "$WORK/cuda13.so" "$WORK/classes-new" r1 jni
refused "$WORK/jni-cuda13.jar" "a cuda13 artifact in a cuda12 distribution" "wrong CUDA classifier"

echo "--- a JNI artifact built from a different revision"
make_jar "$WORK/jni-r2.jar" "$ARCH" "$WORK/exported.so" "$WORK/classes-new" r2 jni
refused "$WORK/jni-r2.jar" "is not the distribution's" "mismatched version metadata"

echo "--- a different set of native libraries (a stale one would stay behind)"
make_jar "$WORK/jni-extra.jar" "$ARCH" "$WORK/exported.so" "$WORK/classes-new" r1 jni libextra.so
refused "$WORK/jni-extra.jar" "native libraries differ" "extra library"

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
