# Building, installing and verifying spark-jitudf (TC/cuDF backend)

This guide builds every artifact from public sources, installs them into a stock Spark
distribution, and verifies that a Java UDF runs on the GPU. It was executed end to end from clean
clones on an x86-64 laptop with an RTX 4070 (see [Verified run](#verified-run) at the end). Exact
versions and revisions are in [../COMPATIBILITY.md](../COMPATIBILITY.md).

Nothing here requires a TornadoVM SDK. That is needed only for the legacy runtime backend
([VALIDATION.md](VALIDATION.md#legacy-tornadovm-runtime-backend)).

## 1. Prerequisites

| | Version | Needed for |
|---|---|---|
| Linux, x86-64 or aarch64 | Tested: Fedora 44 (x86-64) and an aarch64 Linux server (Neoverse-N1) | all |
| NVIDIA GPU | Compute capability 7.0 or newer. Tested: 8.9 (RTX 4070 Laptop), 8.0 (A100) | running |
| NVIDIA driver | New enough for the CUDA major of the JNI build (CUDA 12: 525+; CUDA 13: 580+). Tested: 610.57 | running |
| Docker with the NVIDIA container toolkit | Any current version; `docker run --gpus all` must work | the native JNI build |
| JDK | 21 (tested: OpenJDK 21.0.2 and 21.0.13) | all Java builds and Spark |
| Maven | 3.9 or newer (tested 3.9.11), network access to Maven Central | all Java builds |
| git, curl, python3, unzip | any current | build and verification scripts |
| Disk | about 25 GB free: JNI build tree ~6 GB, Maven repository ~3 GB, NVIDIA's build image ~11 GB | the native JNI build |
| CUDA toolkit on the host | **not needed**: the JNI builds inside NVIDIA's own container, and NVRTC is linked into `libcudf.so` | — |

The Spark and RAPIDS versions are fixed by the build, not chosen by you: Spark 4.0.4 (Scala 2.13)
and the RAPIDS Accelerator 26.08.1 distribution.

## 2. Clone

```bash
git clone -b redesign/driver-side-tc-cudf https://github.com/beestream-ai/spark-jitudf
cd spark-jitudf
git clone -b feature/tornadovm-udf-fallback https://github.com/beestream-ai/cudf-spark third_party/cudf-spark
```

To build the exact independently validated sources rather than the branch tips, check out the
revisions listed in [COMPATIBILITY.md](../COMPATIBILITY.md) (`git checkout <sha>` in each clone).
The branch tips after those revisions contain documentation only.

All commands below run from the `spark-jitudf` directory. The scripts read their paths from
[`scripts/env.sh`](../scripts/env.sh); with the layout used here (everything under
`third_party/`), nothing needs to be set except `JAVA_HOME` if `javac` on your `PATH` is not JDK 21.

## 3. Build the generalized cuDF JNI (native, long)

The released `com.nvidia:spark-rapids-jni:26.08.0` exposes `ColumnView.transform` for one input
column and an INT32 result only. The integration needs the multi-input, typed-output, null-aware
form. The fork carries it as a patch plus a build helper that applies it to NVIDIA's pinned sources:

```bash
third_party/cudf-spark/upstream-patches/build-cudf-spark-jni.sh "$PWD/third_party/cudf-spark-jni"
```

What it does (see the script's header for detail):

* clones `https://github.com/NVIDIA/cudf-spark-jni` at `761c184ff0a0dc5b678b1f5f163c579068beb133`,
  the revision NVIDIA's released 26.08.0 artifact names in its own version-info, with the cuDF
  submodule at `ff5b362d7c06ae5837fd7a7337e2ae20895f324d`;
* stages the patch as `patches/0002-tornadovm-transform-generalized-jni.patch`, so NVIDIA's own
  `build/apply-patches` applies it with theirs;
* sets the artifact version to `26.08.0-tcudf`, so NVIDIA's released coordinates are never
  overwritten;
* builds inside NVIDIA's build container (`build/build-in-docker`, CUDA 12.9.1 by default) for the
  compute capability of **this machine's GPU** (native SASS plus PTX for newer GPUs), and
  `mvn install`s `com.nvidia:spark-rapids-jni:26.08.0-tcudf:cuda12` into your local Maven
  repository.

Options (environment variables): `CUDA_VERSION=13.0.2` builds a `cuda13` artifact;
`CMAKE_CUDA_ARCHITECTURES=80` builds for another GPU (then also `ALLOW_ARCH_MISMATCH=1`);
`CPP_PARALLEL_LEVEL` (default 8; each `nvcc` needs 2–3 GiB of RAM); `LOCAL_MAVEN_REPO` installs
somewhere other than `~/.m2/repository`. On aarch64 the classifier is `cuda12-arm64` or
`cuda13-arm64`.

Rerunning it on an existing work directory is safe. It restores only the cuDF files its own patches
changed, after backing them up, and refuses to touch a submodule with unexpected modifications.

Duration: libcudf dominates. Measured times are in [Verified run](#verified-run).

## 4. Download Spark and the released RAPIDS jar

```bash
mkdir -p third_party/spark-dist
curl -fL https://archive.apache.org/dist/spark/spark-4.0.4/spark-4.0.4-bin-hadoop3.tgz \
  -o third_party/spark-4.0.4-bin-hadoop3.tgz
curl -fsL https://archive.apache.org/dist/spark/spark-4.0.4/spark-4.0.4-bin-hadoop3.tgz.sha512 \
  | awk '{print $1"  third_party/spark-4.0.4-bin-hadoop3.tgz"}' | sha512sum -c -
tar xzf third_party/spark-4.0.4-bin-hadoop3.tgz -C third_party/spark-dist

curl -fL -o third_party/rapids-4-spark_2.13-26.08.1.jar \
  https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.13/26.08.1/rapids-4-spark_2.13-26.08.1.jar
```

`archive.apache.org` can be slow (it is the only host still serving Spark 4.0.4). Any copy whose
SHA-512 matches the published checksum is the same file.

On x86-64 the default `rapids-4-spark_2.13-26.08.1.jar` is the CUDA 12 build. On aarch64, or for
CUDA 13, download the matching classifier jar from the same Maven directory
(`-cuda12-arm64`, `-cuda13`, `-cuda13-arm64`) and save it under the same file name. The
distribution's CUDA major must match the JNI you built in step 3; step 6 checks this.

## 5. Patch the RAPIDS planner into the distribution

```bash
./scripts/build-rapids-patch.sh
```

This builds the fork's `udf-compiler` and `sql-plugin` for Spark 4.0.4 with the fork's own Maven
reactor (`scripts/build-rapids-reactor.sh`, offline-capable once dependencies are cached), and
replaces exactly the classes compiled from the fork's changed sources in `spark404/` and
`spark-shared/` of the released jar. Output: `third_party/rapids-4-spark_2.13-26.08.1-tornado.jar`.

The changed sources are derived from `git merge-base HEAD origin/release/26.08`, which is NVIDIA's
`v26.08.1-rc` commit `1253310626da939c5924017d1c21e94d0932e0c1`, the revision the released jar
was built from. If your clone has no `origin/release/26.08` (e.g. a clone of a clone), set
`RAPIDS_BASE=1253310626da939c5924017d1c21e94d0932e0c1`.

## 6. Fold the generalized JNI into the distribution

```bash
./scripts/swap-cudf-jni.sh
```

The RAPIDS distribution jar already contains a cuDF JNI (classes plus `libcudf.so`). Putting a
second one ahead of it on the class path makes the class and native library come from different
jars. The script instead replaces every native path for the target architecture inside the jar,
and refuses the result unless:

* the JNI jar and the distribution are for the same architecture (ELF header and directory) and the
  same CUDA major;
* both were built from the same pinned JNI and cuDF revisions;
* every native method the JNI classes declare resolves to an exported function in the new
  libraries, and nothing that linked in NVIDIA's release is lost
  (`ColumnView.transformMulti` with descriptor `([JLjava/lang/String;ZIIZI)J` must be among them).

It finds the JNI jar in `~/.m2/repository/com/nvidia/spark-rapids-jni/26.08.0-tcudf/`; pass the
jar path as the first argument to use another one. Output:
`third_party/rapids-4-spark_2.13-26.08.1-tornado-jni.jar` with a `.provenance` file recording the
SHA-256 of both inputs and the output. `VERIFY_ONLY=1 ./scripts/swap-cudf-jni.sh` re-checks an
existing output without rewriting it.

## 7. Build the provider

```bash
mvn package
```

This builds `spark-jitudf-plugin` (shaded: `spark-jitudf-plugin-0.1.0-SNAPSHOT-with-deps.jar`) and
`spark-jitudf-bench`, and runs the unit tests and the 153-case provider coverage corpus (148
plugin tests and 5 bench tests; no GPU needed). It resolves the TC compiler
(`io.github.beehive-lab:tc-*:8.2.0-jdk21`) from Maven Central and the generalized JNI from step 3
from your local repository. `mvn package -DskipTests` skips the tests.

## 8. Run the quick start

```bash
export SPARK_HOME=$PWD/third_party/spark-dist/spark-4.0.4-bin-hadoop3
"$SPARK_HOME/bin/spark-shell" --master 'local[2]' \
  --jars third_party/rapids-4-spark_2.13-26.08.1-tornado-jni.jar,spark-jitudf-plugin/target/spark-jitudf-plugin-0.1.0-SNAPSHOT-with-deps.jar,spark-jitudf-bench/target/spark-jitudf-bench-0.1.0-SNAPSHOT.jar \
  --driver-java-options '-Dtornado.spark.backend=cudf-transform' \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.tornadovm.enabled=true \
  --conf spark.rapids.cudfVersionOverride=true \
  < docs/examples/quickstart.scala
```

`spark.rapids.cudfVersionOverride=true` is required because the distribution now carries cuDF
`26.08.0-tcudf` rather than the released `26.08.0`, and RAPIDS checks that version at startup.

The script is fed on standard input; `spark-shell`'s `-i`/`-I` preload options do not run it
reliably when standard input is not a terminal.

What to look for in the output:

* `total_steps` is the same number in the GPU and the CPU runs (131434424).
* The executed plan (`AdaptiveSparkPlan isFinalPlan=true`) contains `gpu_collatz(id#…)` in the GPU
  operators (`GpuHashAggregate … gpubasicsum(gpu_collatz(id#0L) …)`), and no `GpuCpuBridge` or
  `GpuRowBasedScalaUDF`. Calling `explain()` before the query has run shows only AQE's initial
  plan, which does not yet contain GPU operators; check the executed plan of the query that ran.
* `provider decision for collatz: TORNADOVM_ELIGIBLE (cudf-transform…): [LONG] -> INT`.
* The `sign_label` query returns `non-positive`/`positive` labels, its decision is
  `TORNADOVM_INELIGIBLE(the return type string is not a fixed-width numeric type)`, and its plan
  evaluates the UDF on the CPU through `GpuCpuBridge(... sign_label ...)`.

The recorded output of the verified run is in [Verified run](#verified-run).

## 9. Use it in your own job

| Requirement | How |
|---|---|
| Jars on the driver and executors | `--jars <rapids-…-tornado-jni.jar>,<spark-jitudf-plugin-…-with-deps.jar>,<your job jar>`. The plugin jar is needed on executors because the compiled UDF is a `RapidsUDF` class from it. |
| RAPIDS | `spark.plugins=com.nvidia.spark.SQLPlugin`, plus your usual RAPIDS GPU settings |
| Enable the provider | `spark.rapids.sql.tornadovm.enabled=true` |
| Select the TC/cuDF backend | `-Dtornado.spark.backend=cudf-transform` in the **driver**'s Java options |
| cuDF version check | `spark.rapids.cudfVersionOverride=true` |
| Floating-point UDFs | `spark.rapids.sql.incompatibleOps.enabled=true`, set explicitly (see the configuration reference and [SUPPORTED_UDFS.md](SUPPORTED_UDFS.md)) |

Register UDFs as usual (`spark.udf().register(name, udf, returnType)`) and call them from SQL or the
DataFrame API. `ai.beestream.jitudf.TornadoUdfProvider.decisions()` on the driver returns the
decision for every UDF the provider saw, with the reason for each refusal.

For a cluster, the same jars and settings apply; this has been tested with Spark's `local` and
`local-cluster` masters only (separate driver and executor JVMs on one machine).

## Configuration reference

Verified against the sources at the revisions in [COMPATIBILITY.md](../COMPATIBILITY.md).

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `spark.rapids.sql.tornadovm.enabled` | Spark conf (fork) | `false` | Offer UDFs that have no GPU implementation to the provider. With `false`, the fork behaves as the release it was cut from. Can be changed per session (`spark.conf.set`). |
| `spark.rapids.sql.tornadovm.provider` | Spark conf (fork) | `ai.beestream.jitudf.TornadoUdfProvider` | Provider class, loaded through the context class loader. |
| `tornado.spark.backend` | driver JVM system property | `tornado-runtime` | `cudf-transform` selects the TC/cuDF backend described here. `tornado-runtime` is the legacy backend that executes TornadoVM in executors and needs a TornadoVM SDK. |
| `spark.rapids.sql.enabled` | Spark conf (RAPIDS) | `true` | RAPIDS master switch. |
| `spark.rapids.sql.incompatibleOps.enabled` | Spark conf (RAPIDS) | RAPIDS: `true` when unset. **Provider: must be set explicitly** | Required for any UDF whose reachable code uses floating point. The provider reads the key with `false` as its fallback, so an unset key (which RAPIDS itself treats as `true`) still declines floating-point UDFs. Set it to `true` explicitly to compile them. |
| `spark.rapids.sql.variableFloatAgg.enabled` | Spark conf (RAPIDS) | `true` | Must not be `false` for floating-point LIST sums (a segmented reduction adds in a different order from the Java loop). |
| `spark.rapids.sql.rowBasedUDF.enabled` | Spark conf (RAPIDS) | `false` | Run declined UDFs with RAPIDS' row-based operator (`gpu_row_based_<name>`) inside the GPU projection instead of the CPU bridge. Keeps compiled siblings on the GPU next to receiver-sensitive UDFs. |
| `spark.rapids.sql.expression.cpuBridge.enabled` | Spark conf (RAPIDS) | `true` | RAPIDS CPU bridge for expressions without a GPU implementation. |
| `spark.rapids.cudfVersionOverride` | Spark conf (RAPIDS) | `false` | Must be `true` with the `26.08.0-tcudf` JNI. |
| `spark.rapids.sql.explain` | Spark conf (RAPIDS) | `NOT_ON_GPU` | Logs every expression that stays on the CPU and why (`!Expression <ScalaUDF> ... cannot run on GPU because ...`). `NONE` silences it. |
| `tornado.spark.metrics` | system property | `false` | Per-batch timing records in the provider (diagnostics). |
| `LIBCUDF_KERNEL_CACHE_PATH` | environment, executors | cuDF default | Where cuDF persists NVRTC-compiled kernels. A warm cache removes executor-side compilation on later runs. |

## Verifying an installation

| Check | How | Expected |
|---|---|---|
| Distribution carries the generalized JNI | `VERIFY_ONLY=1 ./scripts/swap-cudf-jni.sh` | `verified: …` and `transformMulti among them` |
| Right cuDF loaded at run time | RAPIDS startup log line `RAPIDS Accelerator JNI build: …` | `version -> 26.08.0-tcudf` |
| Provider compiled the UDF | `TornadoUdfProvider.decisions()` | `TORNADOVM_ELIGIBLE (cudf-transform…)` |
| GPU execution selected | `df.queryExecution.executedPlan` after running the query, or the Spark UI SQL tab | `gpu_<udfname>` in a GPU operator; no `GpuCpuBridge` / `GpuRowBasedScalaUDF` for that UDF |
| End-to-end suite | `RAPIDS_JAR=$PWD/third_party/rapids-4-spark_2.13-26.08.1-tornado-jni.jar NO_TORNADO_OPTS=1 ./scripts/run-spark.sh ai.beestream.jitudf.bench.CudfTransformIntegrationTest 200000` | `=== CHECKS: 79, FAILURES: 0 ===` (see [VALIDATION.md](VALIDATION.md)) |

## Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| `RAPIDS Accelerator ... using cudf 26.08.0-tcudf` and the plugin refuses to start | cuDF version check | Set `spark.rapids.cudfVersionOverride=true`. |
| `UnsatisfiedLinkError: ...transformMulti` or `NoSuchMethodError: ColumnView.transform([Lai/rapids/cudf/ColumnView;...` | The distribution's cuDF JNI is NVIDIA's released one, or a second JNI jar is on the class path ahead of it | Use the jar produced by `swap-cudf-jni.sh` and nothing else carrying `ai/rapids/cudf`. `VERIFY_ONLY=1` re-checks it. |
| `swap-cudf-jni.sh` refuses: CUDA major differs | JNI built with CUDA 12 but distribution is CUDA 13, or vice versa | Rebuild the JNI with the matching `CUDA_VERSION`, or download the distribution with the matching classifier. `ALLOW_CUDA_MAJOR_MISMATCH=1` exists for diagnosis only and is recorded in the provenance file. |
| `swap-cudf-jni.sh` refuses: architecture | x86-64 JNI against an aarch64 distribution or the reverse | Build the JNI on (or for, with `TARGET_ARCH`) the target architecture; use the matching distribution classifier. |
| JNI build ends `BUILD FAILURE ... Could not transfer artifact ... Connection reset` | A transient network error resolving a Maven plugin from Maven Central, after the native compile | Rerun the same `build-cudf-spark-jni.sh` command on the same work directory. The rerun is safe, and ccache makes the native part much faster. |
| `no kernel image is available for execution on the device` | The JNI was built for a newer GPU than the one running it | Rebuild with `CMAKE_CUDA_ARCHITECTURES` set to the oldest compute capability you run on. |
| Plan shows `GpuCpuBridge` / `gpu_row_based_<name>` / a CPU `Project` for your UDF | The provider declined it (expected for unsupported UDFs) | Read `TornadoUdfProvider.decisions().get(name)`; the reason names the construct. See [SUPPORTED_UDFS.md](SUPPORTED_UDFS.md). |
| Decision `TORNADOVM_ELIGIBLE` but the plan has a CPU `Project` | A receiver-sensitive sibling UDF in the same projection | Expected with default settings; `spark.rapids.sql.rowBasedUDF.enabled=true` keeps the projection on the GPU. |
| No decision recorded for the UDF at all | `spark.rapids.sql.tornadovm.enabled` not set, the plugin jar is missing on the driver, or RAPIDS translated the UDF itself (`spark.rapids.sql.udfCompiler.enabled`) | Check the configuration and the `--jars` list. |
| Decision mentions `tornado-runtime` or the job needs `TORNADO_SDK` | `-Dtornado.spark.backend=cudf-transform` missing from the driver options | Add it to `--driver-java-options`. |
| Floating-point UDF declined | `spark.rapids.sql.incompatibleOps.enabled` is not explicitly `true` (unset counts as `false` for the provider) | Set it to `true` if CUDA math results (a few ULP from Java) are acceptable. |
| Query fails with `Transform UDF evaluation failed with error DIVISION_BY_ZERO` | The UDF divided by zero on some row | Expected: Java would have thrown `ArithmeticException`. |
| Query fails with `... error UNKNOWN_ERROR` | A null reached a Java unboxing (e.g. `Integer` → `int`) | Expected: Java would have thrown `NullPointerException`. The diagnostic type differs; see [SUPPORTED_UDFS.md](SUPPORTED_UDFS.md#error-semantics-summarized). |
| NVRTC compilation error in an executor log | The compiled CUDA failed to compile on that executor (toolchain mismatch or a defect) | Not a fallback: the task fails. Report it with the provider decision and the driver log. |
| `ClassNotFoundException: ai.beestream.jitudf...` on executors | Plugin jar not shipped to executors | Put it on `--jars`, not only on the driver class path. |
| Scala/Spark errors at startup | A Spark other than 4.0.4/Scala 2.13 | Use Spark 4.0.4 built for Scala 2.13; the patched planner classes are the `spark404` shim. |

## Verified run

Executed on 2026-11-25 from fresh clones of both repositories at the validated revisions, in a new
directory, with an **empty Maven local repository** (`-Dmaven.repo.local` / `LOCAL_MAVEN_REPO`
pointing into that directory) and the documentation's file layout. Host: Fedora 44, x86-64, RTX 4070
Laptop GPU (compute capability 8.9, driver 595.71.05), 20 threads, 30 GiB RAM, OpenJDK 21.0.2,
Maven 3.9.11, Docker 29.5.2.

| Step | Result | Time |
|---|---|---|
| §3 JNI build (`CPP_PARALLEL_LEVEL=6`) | First attempt compiled all native code, then failed resolving a Maven plugin (`Connection reset` from Maven Central). The rerun of the same command restored the submodule as designed and ended `BUILD SUCCESS`, `== installed com.nvidia:spark-rapids-jni:26.08.0-tcudf:cuda12`. | 50 min + 3 min (the host's ccache already held objects from earlier builds of the same revision; an empty ccache takes longer) |
| §4 Downloads | RAPIDS jar SHA-1 equal to Maven Central's; Spark tarball SHA-512 `OK` against Apache's published checksum | archive.apache.org: ~50 KB/s |
| §5 Planner patch | `BUILD SUCCESS`, 10 changed sources, 46 class files; the resulting jar is entry-for-entry identical to the validated distribution | 4 min 47 s (including Maven downloads) |
| §6 JNI swap | `CUDA toolkit: 12.9 (JNI) and 12.9 (distribution)`, `667 of 667 native methods resolve … transformMulti among them`, `verified` | not timed (minutes) |
| §7 `mvn package` | 148/148 plugin tests, 5/5 bench tests, coverage matrix 102/42/9; the two jars are file-for-file identical in content to the validated jars | 1 min 6 s |
| §8 Quick start | `total_steps = 131434424` on GPU and CPU; executed plan `GpuHashAggregate … gpubasicsum(gpu_collatz(id#0L) …)`, no bridge; `TORNADOVM_ELIGIBLE (cudf-transform…): [LONG] -> INT`; `sign_label` via `GpuCpuBridge`, `TORNADOVM_INELIGIBLE(the return type string is not a fixed-width numeric type)` | < 1 min |
| [VALIDATION.md](VALIDATION.md) suites | cuDF integration 79/79, differential 259/259, separate executors 39/39 (task retry and executor replacement included), benchmark 22/22 checks with every configuration path-verified; all exit 0, all `GPU-EXCLUSIVE: YES` | ~2 min total |

Defects in the earlier instructions that this run found and that are corrected above:
* Spark must be unpacked into `third_party/spark-dist/` for `scripts/env.sh` to find it.
* `spark-shell -I`/`-i` do not run the script non-interactively; feed it on standard input.
* `explain()` before execution shows no GPU operators; the executed plan does.
* `RAPIDS_JAR` defaults to the distribution without the generalized JNI and must be set for the
  TC/cuDF suites.
* The disk requirement was overstated.
