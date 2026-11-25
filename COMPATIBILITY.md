# Compatibility manifest

The single reference for which versions, revisions and artifacts work together. Everything below
was verified against the sources and the independent validation of 2026-11-25.

## Source revisions

| Repository | Branch | Independently validated source revision | Tag |
|---|---|---|---|
| [beestream-ai/spark-jitudf](https://github.com/beestream-ai/spark-jitudf) | `redesign/driver-side-tc-cudf` | `82d505c547b413c57d84b82db70af4c6c6e8c086` | `tc-cudf-review-20261125` |
| [beestream-ai/cudf-spark](https://github.com/beestream-ai/cudf-spark) | `feature/tornadovm-udf-fallback` | `477c7dde10f88c3704414e2fe393dc2bcc702e1a` | `tc-cudf-review-20261125` |

The tags mark the source revisions that were built and validated on both GPUs. Commits after them
on the same branches change documentation and repository metadata (README files, the docs in this
repository, `NOTICE`, the fork's `TORNADOVM_FALLBACK.md` and `upstream-patches/README.md`), plus one
comment-only rewording in five Java files of this repository. Every compiled class of the plugin,
its tests and the bench module was verified byte-identical to the tagged revision after that
change. No Scala, C++, patch or build-script file changes after the tags.

## Component versions

| Component | Version | Source |
|---|---|---|
| JDK | 21 (tested 21.0.2 on x86-64, 21.0.13 on aarch64) | any OpenJDK 21 build |
| Maven | 3.9+ (tested 3.9.11) | |
| Apache Spark | 4.0.4, Scala 2.13, `spark-4.0.4-bin-hadoop3` | `https://archive.apache.org/dist/spark/spark-4.0.4/` |
| RAPIDS Accelerator | 26.08.1 (`com.nvidia:rapids-4-spark_2.13:26.08.1`) with the fork's `spark404`/`spark-shared` classes patched in | Maven Central + `scripts/build-rapids-patch.sh` |
| RAPIDS fork base | NVIDIA `v26.08.1-rc`, `1253310626da939c5924017d1c21e94d0932e0c1` (the revision recorded in the released jar) | `release/26.08` |
| cuDF JNI (generalized) | `com.nvidia:spark-rapids-jni:26.08.0-tcudf`, built from NVIDIA `cudf-spark-jni` `761c184ff0a0dc5b678b1f5f163c579068beb133` with cuDF `ff5b362d7c06ae5837fd7a7337e2ae20895f324d` plus `upstream-patches/cudf-transform-generalized-jni.patch` | `cudf-spark/upstream-patches/build-cudf-spark-jni.sh` |
| TornadoVM TC compiler | `8.2.0-jdk21` (`io.github.beehive-lab:tc-runtime`, `tc-frontend`, `tc-ir`, `tc-tornado`; `tornado-api`, `tornado-runtime` at compile time). 8.2.0 is the first release with `TcCompiler.compileDeviceFunction`. | Maven Central |
| TornadoVM SDK | 8.2.0 with the CUDA backend, **only** for the legacy runtime backend | TornadoVM release |

The generalized JNI patch applies to the pinned revisions above. It has **not** been ported to or
tested against newer cuDF or cudf-spark-jni releases.

## CUDA and architecture

| Platform | JNI build | JNI classifier | RAPIDS distribution classifier | Native arch check |
|---|---|---|---|---|
| x86-64, CUDA 12 | `CUDA_VERSION=12.9.1` (default) | `cuda12` | default jar (CUDA 12) | `amd64` |
| aarch64, CUDA 13 | `CUDA_VERSION=13.0.2` | `cuda13-arm64` | `cuda13-arm64` jar (CUDA 13.3) | `aarch64` |

`scripts/swap-cudf-jni.sh` refuses a JNI whose CUDA major or architecture differs from the
distribution's, or whose native methods do not all link. The JNI is built for the compute
capability of the build machine's GPU unless `CMAKE_CUDA_ARCHITECTURES` says otherwise. PTX JITs
forward only, so build for the oldest GPU you run on.

## Required configuration

| Setting | Value |
|---|---|
| `spark.plugins` | `com.nvidia.spark.SQLPlugin` |
| `spark.rapids.sql.tornadovm.enabled` | `true` |
| driver JVM option | `-Dtornado.spark.backend=cudf-transform` |
| `spark.rapids.cudfVersionOverride` | `true` (the distribution carries cuDF `26.08.0-tcudf`) |
| jars on driver and executors | the swapped RAPIDS distribution, `spark-jitudf-plugin-0.1.0-SNAPSHOT-with-deps.jar`, the job jar |

Optional settings are in [docs/INSTALL.md](docs/INSTALL.md#configuration-reference).

## Tested environments

| | RTX 4070 Laptop | A100 80GB PCIe |
|---|---|---|
| Architecture / compute capability | x86-64 / 8.9 | aarch64 (Neoverse-N1) / 8.0 |
| Driver | 595.71.05 | 610.57.04 |
| CUDA (JNI / distribution) | 12.9 / 12 | 13.0 / 13.3 |
| JDK | 21.0.2 | 21.0.13 |
| Spark masters | `local[2]`, `local-cluster[2,1,4096]` | `local[2]`, `local-cluster[2,1,8192]` |

Independent validation of the source revisions above (2026-11-25). The counts are checks inside
each suite, each comparing against CPU Spark and the physical plan; see
[docs/VALIDATION.md](docs/VALIDATION.md).

| Suite | RTX 4070 | A100 |
|---|---:|---:|
| Plugin unit tests / bench unit tests | 148/148, 5/5 | (Java, not GPU-specific) |
| Spark differential (`CodexDifferentialTest`) | 259/259 | 259/259 |
| cuDF integration (`CudfTransformIntegrationTest`) | 79/79 | 79/79 |
| Separate executors (`SeparateExecutorTest`) | 39/39 | 42/42 |
| Legacy runtime backend (`IntegrationTest`) | 30/30 | 30/30 |

The separate-executor total varies with how many polls the executor-replacement wait takes; the
checks themselves are the same.

Not tested: multi-node clusters, multiple GPUs per executor, Spark versions other than 4.0.4,
RAPIDS versions other than 26.08.1, other cuDF revisions.

## Documentation

| | |
|---|---|
| [README.md](README.md) | Overview, architecture, quick start |
| [docs/INSTALL.md](docs/INSTALL.md) | Build, install, configure, verify, troubleshoot |
| [docs/SUPPORTED_UDFS.md](docs/SUPPORTED_UDFS.md) | Supported UDF domain |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Test suites and benchmark |
| [cudf-spark `TORNADOVM_FALLBACK.md`](https://github.com/beestream-ai/cudf-spark/blob/feature/tornadovm-udf-fallback/TORNADOVM_FALLBACK.md) | The RAPIDS fork's changes |
| [cudf-spark `upstream-patches/README.md`](https://github.com/beestream-ai/cudf-spark/blob/feature/tornadovm-udf-fallback/upstream-patches/README.md) | The generalized cuDF transform JNI binding |
