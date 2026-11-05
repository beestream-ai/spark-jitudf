#!/bin/bash
# An independent source build of the RAPIDS modules this work changed, from the fork's own reactor
# rather than from a patched release jar.
#
# Why the -Dscala.javac.args override. The project sets java.major.version=8 and compiles with
# -Werror under -Xlint:all. Two of those warnings only exist on a JDK newer than the one upstream
# builds 404 with, and both are about upstream's own code:
#
#   options           "source value 8 is obsolete"  -- kills sql-plugin-api immediately.
#   lossy-conversions RapidsHostColumnBuilder.java:441, an int-to-byte compound assignment.
#
# Upstream already does exactly this for its newer Spark profiles: release411 and release420 add
# `-options` to the -Xlint exclusions. release404 does not, so the same property is supplied here
# with the second category added. -Werror stays on, and every other lint category stays on, so
# nothing in the code this work changed is being excused -- these two suppressions are about the
# JDK being newer than the profile expects.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

RAPIDS_SRC=${RAPIDS_SRC:-$TP/cudf-spark}
GOAL=${1:-package}

cd "$RAPIDS_SRC"
echo "=== rapids reactor build: $GOAL ==="
echo "JDK: $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"
echo "HEAD: $(git rev-parse HEAD) on $(git branch --show-current)"

exec mvn -f scala2.13/pom.xml \
  -Dbuildver=404 \
  -pl udf-compiler,sql-plugin -am \
  -DskipTests \
  -Dscala.javac.args='-Xlint:all,-serial,-path,-try,-processing,-options,-lossy-conversions|-Werror' \
  "$GOAL"
