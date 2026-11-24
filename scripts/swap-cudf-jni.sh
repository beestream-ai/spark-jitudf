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
#   from the same pinned revisions with the same CUDA toolkit major version (the classifier), and
#   carries both halves of the binding linked to each other; afterwards, that the result has no
#   duplicate entries and still the distribution's manifest, licence and notice. A jar that fails
#   any of that is never moved into place. A .provenance file beside the result records the SHA-256
#   of both inputs and of the output.
#
#   "Linked to each other" is checked the way the JVM will link them, not by searching for bytes.
#   Every native method the JNI jar's classes declare is turned into the symbol the JVM looks up
#   for it (JNI's short name, or the long name with the mangled argument descriptor when the
#   method is overloaded), and that symbol must be a defined, global, default-visibility function in
#   the dynamic symbol table (.dynsym) of one of this architecture's libraries -- read with a small
#   ELF parser here, so the check is the same on every host and needs no binutils. A name that is
#   only text in the file, only an undefined reference, hidden, local, or absent does not count.
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
import collections, hashlib, os, re, struct, sys, zipfile

jni_path, in_path, out_path, target_arch, verify_only = sys.argv[1:6]
ELF_MACHINE = {"amd64": 62, "aarch64": 183}
NEW_BINDING = ("ai/rapids/cudf/ColumnView", "transformMulti")
TOOLKIT = re.compile(rb"Cuda compilation tools, release ([0-9]+)\.([0-9]+)")
errors = []

def elf_symbols(data):
    """Every named symbol of an ELF64 little-endian shared object, classified the way the dynamic
    linker sees it. Raises ValueError for a file that is not one."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    try:
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
        if shoff == 0 or shnum == 0 or shoff + shentsize * shnum > len(data):
            raise ValueError("no section header table")
        sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
        symbols = {}
        for kind, dynamic in ((11, True), (2, False)):  # SHT_DYNSYM, then SHT_SYMTAB
            for sec in sections:
                if sec[1] != kind:
                    continue
                strtab = sections[sec[6]]
                for off in range(sec[4], sec[4] + sec[5], 24):
                    st_name, st_info, st_other, st_shndx = struct.unpack_from("<IBBH", data, off)
                    start = strtab[4] + st_name
                    name = data[start:data.index(b"\0", start)].decode("latin-1")
                    if not name:
                        continue
                    if not dynamic:
                        symbols.setdefault(name, "defined only in the static symbol table, not exported")
                        continue
                    bind, kind_, visibility = st_info >> 4, st_info & 0xF, st_other & 3
                    if st_shndx == 0:
                        status = "an undefined reference, not a definition"
                    elif bind == 0:
                        status = "a local symbol, not exported"
                    elif visibility in (1, 2):
                        status = "hidden, not exported"
                    elif kind_ not in (2, 10):  # STT_FUNC, STT_GNU_IFUNC
                        status = "exported, but not a function"
                    else:
                        status = "exported"
                    if symbols.get(name) != "exported":
                        symbols[name] = status
        if not any(sec[1] == 11 for sec in sections):
            raise ValueError("no dynamic symbol table")
        return symbols
    except struct.error as e:
        raise ValueError(f"truncated ELF file ({e})")

def native_methods(classfile):
    """(class name, [(method, descriptor)]) for the native methods a class file declares."""
    d = classfile
    if d[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("not a class file")
    count, = struct.unpack_from(">H", d, 8)
    pool, i, p = [None] * count, 1, 10
    while i < count:
        tag = d[p]
        if tag == 1:
            n, = struct.unpack_from(">H", d, p + 1)
            pool[i] = d[p + 3:p + 3 + n].decode("utf-8", "surrogatepass")
            p += 3 + n
        elif tag in (3, 4):
            p += 5
        elif tag in (5, 6):
            p += 9
            i += 1
        elif tag in (7, 8, 16, 19, 20):
            pool[i] = struct.unpack_from(">H", d, p + 1)[0]
            p += 3
        elif tag in (9, 10, 11, 12, 17, 18):
            p += 5
        elif tag == 15:
            p += 4
        else:
            raise ValueError(f"unknown constant pool tag {tag}")
        i += 1
    this_class, = struct.unpack_from(">H", d, p + 2)
    name = pool[pool[this_class]]
    p += 6
    interfaces, = struct.unpack_from(">H", d, p)
    p += 2 + 2 * interfaces
    members = []
    for _ in range(2):  # fields, then methods
        n, = struct.unpack_from(">H", d, p)
        p += 2
        members = []
        for _ in range(n):
            access, name_index, desc_index, attributes = struct.unpack_from(">HHHH", d, p)
            p += 8
            for _ in range(attributes):
                length, = struct.unpack_from(">I", d, p + 2)
                p += 6 + length
            members.append((access, pool[name_index], pool[desc_index]))
    return name, [(m, desc) for access, m, desc in members if access & 0x0100]

def jni_mangle(text):
    out = []
    for ch in text:
        if ch == "/":
            out.append("_")
        elif ch == "_":
            out.append("_1")
        elif ch == ";":
            out.append("_2")
        elif ch == "[":
            out.append("_3")
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        else:
            out.append("_0%04x" % ord(ch))
    return "".join(out)

def jni_symbols(cls, method, descriptor):
    """The short and long names the JVM looks up for a native method (JNI spec, "Resolving Native
    Method Names")."""
    short = "Java_" + jni_mangle(cls) + "_" + jni_mangle(method)
    return short, short + "__" + jni_mangle(descriptor[1:descriptor.index(")")])

def toolkit_major(z, names):
    for n in names:
        if n.endswith("/libcudf.so"):
            m = TOOLKIT.search(z.read(n))
            if m:
                return int(m.group(1)), f"{int(m.group(1))}.{int(m.group(2))}"
    return None, None

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
    # The CUDA toolkit the libraries were compiled with -- what the cuda12/cuda13 classifier names.
    # nvcc records it in every object it compiles; mixing majors in one jar mixes runtimes.
    jni_cuda, jni_cuda_text = toolkit_major(jni, sorted(jni_native))
    dist_cuda, dist_cuda_text = toolkit_major(dist, sorted(dist_native))
    if jni_cuda is None or dist_cuda is None:
        fail(f"the CUDA toolkit of {'the JNI' if jni_cuda is None else 'the distribution'}'s libcudf.so cannot be established, "
             "so the classifier cannot be checked")
    elif jni_cuda != dist_cuda:
        fail(f"the JNI libraries were built with CUDA {jni_cuda_text} and the distribution's with CUDA {dist_cuda_text}: "
             f"a cuda{jni_cuda} artifact in a cuda{dist_cuda} distribution")
    else:
        print(f"  CUDA toolkit: {jni_cuda_text} (JNI) and {dist_cuda_text} (distribution), classifier cuda{jni_cuda}")
    # Java and native halves of the binding, linked the way the JVM will link them. The cuDF JNI
    # layer is linked into libcudf.so itself, so exports are gathered across this architecture's
    # libraries rather than from one assumed file.
    exports = {}
    for n in sorted(jni_native):
        try:
            for symbol, status in elf_symbols(jni.read(n)).items():
                if exports.get(symbol) != "exported":
                    exports[symbol] = status
        except ValueError as e:
            fail(f"{n} has no readable ELF dynamic symbol table: {e}")
    unresolved, checked, saw_new_binding = [], 0, False
    for n in jni_names:
        if not n.endswith(".class"):
            continue
        try:
            cls, natives = native_methods(jni.read(n))
        except (ValueError, struct.error, IndexError, UnicodeDecodeError) as e:
            fail(f"{n} could not be read as a class file: {e}")
            continue
        overloads = collections.Counter(m for m, _ in natives)
        for method, descriptor in natives:
            checked += 1
            short, long_ = jni_symbols(cls, method, descriptor)
            if (cls, method) == NEW_BINDING:
                saw_new_binding = True
            if exports.get(long_) == "exported":
                continue
            if overloads[method] == 1 and exports.get(short) == "exported":
                continue
            if overloads[method] > 1 and exports.get(short) == "exported":
                why = "is overloaded, so the JVM needs the long name " + long_ + ", which is not exported"
            else:
                status = exports.get(short) or exports.get(long_)
                if status is None:
                    in_text = any(short.encode() in jni.read(lib) for lib in sorted(jni_native)) if (cls, method) == NEW_BINDING else False
                    status = "present only as text in the file" if in_text else "absent"
                why = f"needs {short}, which is {status}"
            unresolved.append(f"{cls.replace('/', '.')}.{method}{descriptor} {why}")
    if not saw_new_binding:
        fail("ColumnView.class in the JNI jar declares no native transformMulti: it is not the generalized binding")
    if unresolved:
        fail(f"{len(unresolved)} of {checked} native methods would not link: " + "; ".join(unresolved[:4]))
    else:
        print(f"  JNI linkage: all {checked} native methods resolve to exported functions (transformMulti among them)")
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
print("  verified: one architecture, exact native paths, matching revisions and CUDA major, every native")
print("  method linked to an exported function, no duplicate entries, and the distribution's own")
print("  manifest, licence and notice")
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
