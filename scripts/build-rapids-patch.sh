#!/bin/bash
# Writes a patched copy of the released RAPIDS 26.08.1 distribution jar in which this fork's
# changed classes replace the released ones, in the spark404 shim and in spark-shared.
#
# Where the classes come from. They come from a real reactor build of the fork
# (scripts/build-rapids-reactor.sh), not from compiling a handful of sources against a
# pre-extracted classpath. That matters for more than tidiness: the reactor build applies the
# project's own -Xfatal-warnings and -Werror settings, and the first time it was run against this
# work it rejected two unused imports that the subset build had compiled happily for weeks. A
# build that cannot fail cannot tell you anything.
#
# Why still a patched jar rather than `mvn package` of the whole distribution. The distribution jar
# carries a compiled copy of every class for each of 25 supported Spark versions. Only the Spark
# 4.0.4 shim is exercised here, and building the other 24 takes hours on a laptop with no bearing
# on anything measured. The classes that go in are built from the fork's sources by the fork's own
# build, and they replace the released ones at the exact paths the shim loader resolves.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# The fork and branch that carry the fallback hook. Cloning instructions are in the README;
# without this branch there is nothing for the plugin to be called from.
FORK=${RAPIDS_SRC:-$TP/cudf-spark}
RAPIDS_BASE_REF=${RAPIDS_BASE_REF:-origin/release/26.08}
JAR_IN=$TP/rapids-4-spark_2.13-26.08.1.jar
JAR_OUT=$TP/rapids-4-spark_2.13-26.08.1-tornado.jar

# The released jar this patches. Checked here rather than at the cp below, because the reactor
# build above it takes minutes and failing after that with "cp: No such file" tells you nothing.
if [ ! -f "$JAR_IN" ]; then
  echo "missing the released RAPIDS jar: $JAR_IN" >&2
  echo "download it once:" >&2
  echo "  mkdir -p $TP && curl -L -o $JAR_IN \\" >&2
  echo "    https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.13/26.08.1/rapids-4-spark_2.13-26.08.1.jar" >&2
  echo "or set RAPIDS_JAR_STOCK if you keep it elsewhere." >&2
  exit 2
fi

if [ "${SKIP_REACTOR:-false}" != "true" ]; then
  echo "=== reactor build ==="
  "$ROOT/scripts/build-rapids-reactor.sh" package
fi

# The sources this fork changed, relative to the upstream commit it was branched from. Derived
# rather than listed, so that a source added later cannot be silently left out of the jar.
#
# The base is the branch point: where feature/tornadovm-udf-fallback left the release it was cut
# from. Derived rather than written down, so that a source added to the branch later cannot be
# silently left out of the jar -- but derived from the merge-base with the upstream release line,
# not from "commits no remote has", because the branch is published and every one of its commits
# is on a remote. Override with RAPIDS_BASE if the branch is rebased onto something else.
cd "$FORK"
BASE=${RAPIDS_BASE:-$(git merge-base HEAD "$RAPIDS_BASE_REF" 2>/dev/null)}
if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null; then
  echo "cannot determine the branch point of $(git branch --show-current) in $FORK" >&2
  echo "set RAPIDS_BASE to the upstream commit it was branched from" >&2
  exit 2
fi
mapfile -t CHANGED < <(git diff --name-only "$BASE" HEAD -- 'udf-compiler/src/main/**/*.scala' 'sql-plugin/src/main/**/*.scala')
if [ ${#CHANGED[@]} -eq 0 ]; then
  echo "no changed Scala sources found against $BASE" >&2
  exit 2
fi
echo "=== ${#CHANGED[@]} changed sources against $BASE ==="
printf '  %s\n' "${CHANGED[@]}"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/spark404" "$STAGE/spark-shared"

# For each changed source, copy every class file the compiler produced from it: the top-level
# definitions it declares, their companions, and their nested and anonymous classes, all of which
# the compiler names with the declaring class as a prefix.
COPIED=0
for src in "${CHANGED[@]}"; do
  module=${src%%/*}
  classes="$FORK/scala2.13/$module/target/spark404/classes"
  [ -d "$classes" ] || { echo "no reactor output for $module at $classes" >&2; exit 3; }
  pkg=$(grep -m1 '^package ' "$FORK/$src" | sed 's/^package //; s/\r//; s/\./\//g')
  while read -r name; do
    for f in "$classes/$pkg/$name".class "$classes/$pkg/$name"'$'*.class; do
      [ -e "$f" ] || continue
      rel="$pkg/$(basename "$f")"
      cp -f "$f" "$STAGE/spark404/$rel" 2>/dev/null || { mkdir -p "$STAGE/spark404/$pkg" "$STAGE/spark-shared/$pkg"; cp -f "$f" "$STAGE/spark404/$rel"; }
      cp -f "$f" "$STAGE/spark-shared/$rel"
      COPIED=$((COPIED + 1))
    done
  done < <(grep -oE '^(final |sealed |abstract |case )*(class|object|trait) [A-Za-z0-9_]+' "$FORK/$src" | awk '{print $NF}' | sort -u)
done
echo "=== $COPIED class files staged ==="

# Both locations. The distribution puts a class in spark-shared when its compiled form is identical
# for every supported Spark version, and in the sparkNNN shim when it is not; the shim loader adds
# spark-shared to the classloader ahead of the shim, so a class patched only in the shim is
# shadowed by the released copy in spark-shared and never loads.
cp -f "$JAR_IN" "$JAR_OUT"
( cd "$STAGE" && "$JAVA_HOME/bin/jar" uf "$JAR_OUT" spark404 spark-shared )
echo "patched jar: $JAR_OUT"
"$JAVA_HOME/bin/jar" tf "$JAR_OUT" | grep -c 'Tornado' | sed 's/^/Tornado classes in jar: /'
