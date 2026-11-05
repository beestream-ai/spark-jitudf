# Shared environment for every run: the toolchain, Spark distribution, patched RAPIDS jar and
# TornadoVM SDK this project is built and measured against.
#
# Every path below can be overridden from the environment, and ROOT is derived from this file's
# own location rather than written out, so a clone anywhere works without editing anything. The
# three that have no sensible default on another machine -- the JDK, the TornadoVM SDK, and where
# the Spark distribution was unpacked -- are the ones to set first; see the README.

# Resolve this file's directory even when it is sourced rather than executed.
export ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# JDK 21. Derived from javac on PATH when not set; TornadoVM needs the Panama preview flags,
# which scripts/run-spark.sh passes.
export JAVA_HOME=${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac 2>/dev/null)")")")}
export PATH="$JAVA_HOME/bin:$PATH"

export TP=${TP:-$ROOT/third_party}
export SPARK_HOME=${SPARK_HOME:-$TP/spark-dist/spark-4.0.4-bin-hadoop3}
export RAPIDS_JAR=${RAPIDS_JAR:-$TP/rapids-4-spark_2.13-26.08.1-tornado.jar}
export RAPIDS_JAR_STOCK=${RAPIDS_JAR_STOCK:-$TP/rapids-4-spark_2.13-26.08.1.jar}

# The built TornadoVM SDK, i.e. the directory holding bin/, lib/, share/java/tornado and
# etc/exportLists. After building TornadoVM with the CUDA backend this is under its dist/.
export TORNADO_SDK=${TORNADO_SDK:-$TP/tornadovm-sdk}

# The two Maven modules. PLUGIN_JAR is what RAPIDS calls into and goes on the spark-submit jars
# list; BENCH_JAR holds the programs that are submitted -- the integration test, the
# microbenchmark, the probes and the MovieLens application.
export JITUDF_VERSION=${JITUDF_VERSION:-0.1.0-SNAPSHOT}
export PLUGIN_JAR=${PLUGIN_JAR:-$ROOT/spark-jitudf-plugin/target/spark-jitudf-plugin-$JITUDF_VERSION-with-deps.jar}
export BENCH_JAR=${BENCH_JAR:-$ROOT/spark-jitudf-bench/target/spark-jitudf-bench-$JITUDF_VERSION.jar}

# NVIDIA's native cosine-similarity UDF, built by scripts/build-native-udf.sh (B3 baseline).
export NATIVE_UDF_JAR=${NATIVE_UDF_JAR:-$TP/native-udf/nvidia-udf-examples.jar}
