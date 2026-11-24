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
#   when it unpacks spark-rapids-jni into its parallel world, done to an already-built jar. The JNI
#   jar's own META-INF (manifest, licence) is not part of it: the distribution keeps its own.
#
#   Before anything is written it checks that the JNI jar is for exactly the target architecture
#   (directory and ELF header), replaces exactly the distribution's native paths for it, was built
#   from the same pinned revisions, and carries both halves of the new binding; afterwards, that the
#   result has no duplicate entries and still the distribution's manifest, licence and notice. A
#   jar that fails any of that is never moved into place. A .provenance file beside the result
#   records the SHA-256 of both inputs and of the output.
#
#   For the deliverable, the RAPIDS distribution build is the better route; this is the
#   development shortcut, made safe rather than replaced.
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

# The architecture the result is for: this machine's, unless TARGET_ARCH says otherwise (building a
# jar for caprica on the laptop, say). Spelled the way the jars spell their native directories.
case "${TARGET_ARCH:-$(uname -m)}" in
  x86_64|amd64) TARGET_ARCH=amd64 ;;
  aarch64|arm64) TARGET_ARCH=aarch64 ;;
  *) echo "unknown TARGET_ARCH ${TARGET_ARCH:-$(uname -m)}" >&2; exit 2 ;;
esac
echo "arch: $TARGET_ARCH"

# VERIFY_ONLY=1 re-runs the checks against an existing output jar. Rewriting a gigabyte takes
# minutes, and the checks are what one actually wants to repeat.
#
# Otherwise the result is built as OUT_JAR.part, verified there, and only then renamed into place,
# so a jar that fails a check is never left where a launcher would pick it up. The inputs -- the
# JNI artifact in the local Maven repository and the distribution jar -- are only ever read.
if [ "${VERIFY_ONLY:-0}" = "1" ]; then
  [ -f "$OUT_JAR" ] || { echo "no jar to verify at $OUT_JAR" >&2; exit 2; }
  echo "(verify only: the jar is not rebuilt)"
  CANDIDATE=$OUT_JAR
else
  CANDIDATE=$OUT_JAR.part
fi

discard_candidate() {
  # A result that failed its checks is removed by name, never moved into place.
  if [ "${VERIFY_ONLY:-0}" != "1" ]; then
    rm -f -- "$CANDIDATE" "$CANDIDATE.provenance"
  fi
  exit 3
}
python3 - "$JNI_JAR" "$IN_JAR" "$CANDIDATE" "$TARGET_ARCH" "${VERIFY_ONLY:-0}" <<'PY' || discard_candidate
import hashlib, os, re, sys, zipfile

jni_path, in_path, out_path, target_arch, verify_only = sys.argv[1:6]
ELF_MACHINE = {"amd64": 62, "aarch64": 183}
NEW_SYMBOL = b"Java_ai_rapids_cudf_ColumnView_transformMulti"
errors = []

def fail(message):
    errors.append(message)
    print("  ERROR: " + message, file=sys.stderr)

def native_arch(name):
    m = re.match(r"^([^/]+)/Linux/[^/]+\.so$", name)
    return m.group(1) if m else None

def props(z, name):
    try:
        return dict(line.split("=", 1) for line in z.read(name).decode().splitlines() if "=" in line)
    except KeyError:
        return {}

def fresh(info, method):
    """A new entry for the output with the source's name, time and attributes -- not the source's
    ZipInfo itself, which carries offsets into the jar it came from."""
    entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    entry.external_attr = info.external_attr
    entry.compress_type = method
    return entry

def provided_by_jni(name):
    """Which JNI entries take part in the substitution. The JNI jar's own META-INF -- its manifest
    and licence -- is not the distribution's and must not replace it; only its Maven coordinates
    are carried, under their own path, as a record of what was put in."""
    if name.endswith("/"):
        return False
    if name.startswith("META-INF/"):
        return name.startswith("META-INF/maven/com.nvidia/spark-rapids-jni/")
    return True

with zipfile.ZipFile(jni_path) as jni, zipfile.ZipFile(in_path) as dist:
    jni_names = [i.filename for i in jni.infolist() if provided_by_jni(i.filename)]
    dist_names = [i.filename for i in dist.infolist()]

    # --- checks on the inputs, before anything is written
    jni_archs = {native_arch(n) for n in jni_names} - {None}
    dist_archs = {native_arch(n) for n in dist_names} - {None}
    print(f"  native architectures: jni {sorted(jni_archs)}, distribution {sorted(dist_archs)}")
    if jni_archs != {target_arch}:
        fail(f"the JNI jar carries native code for {sorted(jni_archs)}, not exactly {target_arch}; "
             "substituting it would leave the distribution's own libraries beside a second architecture's")
    if target_arch not in dist_archs:
        fail(f"the distribution has no {target_arch} native libraries to replace")
    for name in jni_names:
        if native_arch(name):
            head = jni.read(name)[:20]
            machine = int.from_bytes(head[18:20], "little") if head[:4] == b"\x7fELF" else -1
            if machine != ELF_MACHINE.get(target_arch):
                fail(f"{name} is not an ELF {target_arch} library (e_machine {machine})")
    # Exactly the paths being replaced: the distribution's native set for this architecture must be
    # the JNI's, or a stale library from the old build stays behind next to the new one.
    jni_native = {n for n in jni_names if native_arch(n) == target_arch}
    dist_native = {n for n in dist_names if native_arch(n) == target_arch}
    if jni_native != dist_native:
        fail(f"native libraries differ: only in the distribution {sorted(dist_native - jni_native)}, "
             f"only in the JNI jar {sorted(jni_native - dist_native)}")
    # Same pinned sources: the JNI build must be of the revisions the distribution was built against.
    for stamp in ("spark-rapids-jni-version-info.properties", "cudf-java-version-info.properties"):
        a, b = props(jni, stamp), props(dist, stamp)
        if not a or a.get("revision") != b.get("revision"):
            fail(f"{stamp}: JNI revision {a.get('revision')} is not the distribution's {b.get('revision')}")
        else:
            print(f"  {stamp}: revision {a['revision'][:12]} on both sides; version {b.get('version')} -> {a.get('version')}")
    # Java and native halves of the new binding together. The cudf JNI layer is linked into
    # libcudf.so itself (nm -D shows the Java_ai_rapids_cudf_* exports there), so the symbol is
    # looked for across this architecture's libraries rather than in one assumed file.
    if not any(NEW_SYMBOL in jni.read(n) for n in sorted(jni_native)):
        fail(f"no {target_arch} native library exports {NEW_SYMBOL.decode()}, which the new ColumnView.transform calls")
    if "ai/rapids/cudf/OutputNullability.class" not in jni_names:
        fail("the JNI jar has no OutputNullability.class: it is not the generalized binding")
    # Nowhere else in the distribution may a second copy of cuDF's classes hide.
    stray = [n for n in dist_names if "/ai/rapids/cudf/" in n and not n.startswith("ai/rapids/cudf/")]
    if stray:
        fail(f"the distribution carries cuDF classes outside ai/rapids/cudf/, which this would not replace: {stray[:3]}")

    if errors:
        sys.exit(3)

    # --- the substitution
    if verify_only != "1":
        provided = set(jni_names)
        with zipfile.ZipFile(out_path, "w", allowZip64=True) as out:
            kept = replaced = 0
            for info in dist.infolist():
                if info.filename in provided:
                    replaced += 1
                    continue
                # Large members are stored rather than recompressed: the input is already
                # compressed, and deflating a gigabyte again to save nothing costs minutes.
                method = zipfile.ZIP_STORED if info.file_size > 8 << 20 else zipfile.ZIP_DEFLATED
                out.writestr(fresh(info, method), dist.read(info))
                kept += 1
            added = 0
            for name in jni_names:
                info = jni.getinfo(name)
                method = zipfile.ZIP_STORED if info.file_size > 8 << 20 else zipfile.ZIP_DEFLATED
                out.writestr(fresh(info, method), jni.read(info))
                added += 1
            print(f"  kept {kept}, replaced {replaced}, added {added}")

    # --- checks on the result
    with zipfile.ZipFile(out_path) as out:
        names = [i.filename for i in out.infolist()]
        dupes = sorted({n for n in names if names.count(n) > 1}) if len(names) != len(set(names)) else []
        if dupes:
            fail(f"the result has duplicate entries: {dupes[:5]}")
        for name in jni_names:
            if out.read(name) != jni.read(name):
                fail(f"{name} in the result is not the JNI jar's")
        for meta in ("META-INF/MANIFEST.MF", "META-INF/NOTICE", "META-INF/LICENSE"):
            if meta in dist_names and out.read(meta) != dist.read(meta):
                fail(f"{meta} is not the distribution's")
        out_archs = {native_arch(n) for n in names} - {None}
        if out_archs != dist_archs:
            fail(f"the result's native architectures {sorted(out_archs)} are not the distribution's {sorted(dist_archs)}")
        for stamp in ("spark-rapids-jni-version-info.properties", "cudf-java-version-info.properties"):
            if names.count(stamp) != 1:
                fail(f"{stamp} appears {names.count(stamp)} times")
    if errors:
        sys.exit(3)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

# Provenance, beside the jar: which inputs, byte for byte, produced it.
with open(out_path + ".provenance", "w") as p:
    p.write(f"jni {sha256(jni_path)} {os.path.abspath(jni_path)}\n")
    p.write(f"in {sha256(in_path)} {os.path.abspath(in_path)}\n")
    p.write(f"out {sha256(out_path)}\n")
    p.write(f"arch {target_arch}\n")
print("  verified: one architecture, exact native paths, matching revisions, the new JNI symbol,")
print("  no duplicate entries, and the distribution's own manifest, licence and notice")
PY

if [ "${VERIFY_ONLY:-0}" != "1" ]; then
  mv -f -- "$CANDIDATE" "$OUT_JAR"
  mv -f -- "$CANDIDATE.provenance" "$OUT_JAR.provenance"
fi

echo
echo "=== what the result carries ==="
unzip -p "$OUT_JAR" spark-rapids-jni-version-info.properties | sed 's/^/  jni  /'
unzip -p "$OUT_JAR" cudf-java-version-info.properties | sed 's/^/  cudf /'
sed 's/^/  provenance  /' "$OUT_JAR.provenance"
echo
echo "use it with:  export RAPIDS_JAR=$OUT_JAR"
