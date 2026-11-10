# spark-jitudf

**GPU-compiled Spark UDFs, through TornadoVM.**

Runs an ordinary Spark JVM UDF on a GPU, after the RAPIDS `udf-compiler` has declined to translate
it.

A UDF containing a loop is the clearest case. RAPIDS builds Catalyst expression trees, and an
expression tree cannot hold a loop, so the UDF goes to the CPU — even though the arithmetic inside
it is exactly what a GPU is good at. This project compiles that UDF's **bytecode** into a TornadoVM
kernel instead, and runs it over the cuDF column buffers **where they already are**: no copy to the
host, no copy between device buffers, and no TornadoVM array header written into memory libcudf
owns.

The UDF is not rewritten, and the query is not written differently:

```java
spark.udf().register("looped", new LoopUdf(), DataTypes.DoubleType);
spark.sql("SELECT id, looped(a, b) FROM numbers");   // runs on the GPU
```

Nothing in the benchmark module calls this project's API, because there is no API to call: a UDF
reaches the GPU through `spark.udf().register` and the RAPIDS plugin, and Spark itself is
unmodified.

## Why

For a UDF the plugin cannot translate, **turning RAPIDS on can make an application slower than not
using the GPU at all.** Each batch leaves the GPU, becomes rows, is scored in the JVM, and returns.
That is the right thing for the plugin to do — correct answers beat refusing — and it is the cost
this removes.

The measurements, what separates the figures that can be quoted, and what the evaluation does *not*
establish are in [spark-jitudf-benchmarking](https://github.com/beestream-ai/spark-jitudf-benchmarking).

## What is here

| Path | |
|---|---|
| `spark-jitudf-plugin/` | The module RAPIDS calls into: UDF resolution, eligibility analysis, ASM lowering, kernel generation, cuDF buffer adapters, caches. |
| `spark-jitudf-bench/` | The runnable end: the end-to-end test, the microbenchmark, the MovieLens application the evaluation measures, the probes, and the UDFs they all use. |
| `scripts/` | `env.sh` (paths, all overridable), `run-spark.sh` (launcher), `build-rapids-patch.sh`, `build-rapids-reactor.sh`. |
| [spark-jitudf-benchmarking](https://github.com/beestream-ai/spark-jitudf-benchmarking) | The benchmark drivers, the correctness checks, the results and the evaluation. |

## What you need

| | |
|---|---|
| GPU | NVIDIA, compute capability 7.0+, 8 GiB or more |
| Driver | 525 or newer, for CUDA 12 |
| JDK | 21 |
| Maven | 3.9 or newer |
| TornadoVM | **8.1.0 or newer**, with the CUDA backend. Earlier versions lack the device-view API the cuDF buffers are bound through. The quickest way is SDKMAN: `sdk install tornadovm 8.1.0-jdk21-cuda`, which gives you a `TORNADO_SDK` directory directly. Otherwise [build it](https://tornadovm.readthedocs.io/en/latest/installation.html). |
| Spark | 4.0.4, Scala 2.13, the stock binary distribution. Unmodified. |
| RAPIDS | The released 26.08.1 jar, with the fork's classes patched in (below) |

Only `mvn package` is needed to run the unit tests; everything else is needed to reach a GPU.

## Getting it running

One other repository is needed: [`beestream-ai/cudf-spark`](https://github.com/beestream-ai/cudf-spark)
on branch `feature/tornadovm-udf-fallback`, a fork of NVIDIA's RAPIDS plugin that makes the UDF
compiler say *why* it declined a UDF and lets something else try. Without it there is no hook for
this plugin to be called from, and the released jar will not do.

```bash
# the three that have no sensible default on another machine
export JAVA_HOME=/path/to/jdk-21
export TORNADO_SDK=/path/to/tornadovm/dist/tornadovm-*-cuda   # holds bin/, share/java/tornado
export SPARK_HOME=/path/to/spark-4.0.4-bin-hadoop3

mkdir -p third_party

# Spark, if you do not have it
curl -L https://archive.apache.org/dist/spark/spark-4.0.4/spark-4.0.4-bin-hadoop3.tgz \
  | tar xz -C third_party

# the released RAPIDS jar the fork's classes are patched into
curl -L -o third_party/rapids-4-spark_2.13-26.08.1.jar \
  https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.13/26.08.1/rapids-4-spark_2.13-26.08.1.jar

# the fork
git clone --branch feature/tornadovm-udf-fallback \
  https://github.com/beestream-ai/cudf-spark third_party/cudf-spark

./scripts/build-rapids-patch.sh     # the fork's changes into the released RAPIDS jar
mvn package                         # both modules, and the plugin's 33 unit tests
./scripts/run-spark.sh ai.beestream.jitudf.bench.IntegrationTest
```

The last command must end `CHECKS: 27, FAILURES: 0`.

Every path above can be overridden from the environment; `scripts/env.sh` lists them and derives
the rest from the repository root, so a clone anywhere works without editing anything.

On an SDKMAN install, `tornado --devices` fails with `Module tornado.clblast not found`. That is a
quirk of that package's own launcher and does not affect anything here: `run-spark.sh` builds its
own module path and asks only for `tornado.runtime`, `tornado.drivers.common` and
`tornado.drivers.cuda`. Use `RapidsSmokeTest` below to check your setup instead.

It is **off by default**. Nothing changes until `spark.rapids.sql.tornadovm.enabled=true` is set.

## Trying it

Each of these is `./scripts/run-spark.sh <class>`. They need the build above; none needs a dataset.

| Class (under `ai.beestream.jitudf.bench`) | What it shows |
|---|---|
| `IntegrationTest` | The whole thing: a UDF with a loop reaching a kernel, a string-returning UDF correctly staying on the CPU, an expression RAPIDS already handled keeping its path, and every accelerated value compared row by row against the same UDF on the CPU. Ends `CHECKS: 27, FAILURES: 0`. |
| `RapidsSmokeTest` | That RAPIDS itself initialises and puts a plain query on the GPU. Run this first if the integration test fails — it separates "RAPIDS is not working here" from "the integration is not working here". |
| `Benchmark` | End-to-end timings of one query under each configuration, on whatever machine you are on. |
| `ListStress` | The `ArrayType`/cuDF `LIST` path on its own, at a range of sizes. |
| `PlanDump` | The executed physical plan, so you can see which operator the UDF became. |
| `ArrayUdfProbe`, `WrapperProbe` | What Spark actually hands and wraps a Java UDF in. Diagnostics, for when a UDF you wrote is refused. |
| `AlsProbe` | Whether Spark's own ALS prediction UDF reaches a kernel unmodified. |

To try your own UDF, copy one from `spark-jitudf-bench/src/main/java/ai/beestream/jitudf/bench/udfs/`
— `LoopUdf` is the simplest — register it, and run a query over it. `TornadoUdfProvider.decisions()`
then says what was compiled and what was refused, and a refusal names the construct that caused it
(`LOOP`, `UNSUPPORTED_MATH`, `OBJECT_ACCESS`, and so on). `IntegrationTest` prints that log at the
end of its run, so it doubles as a worked example of how to read it.

The `movielens` classes are the measured application; they need the MovieLens 25M dataset, and
their data preparation and measurement protocol are in
[spark-jitudf-benchmarking](https://github.com/beestream-ai/spark-jitudf-benchmarking).

## What it handles

Fixed-width numeric columns (`int`, `long`, `float`, `double`) in any arity, and `ArrayType`
columns over the same element types — a cuDF `LIST`, which the kernel reads as offsets plus a
flattened child, with rows free to differ in length. Loops, branches, `java.lang.Math`, and static
helper methods in the UDF's own class.

## What it does not do

Nulls, structs, strings, decimals, and anything with state, exceptions or allocation. All of these
are refused **while planning**, so the query simply keeps the path it already had — a UDF is never
sent to a kernel on the hope that it will work.

Planning-time refusal is not executor-time fallback: once a plan holds a kernel, a failure during
execution fails the task.

## Status

Research code. One application, one dataset, Spark `local` mode. No cluster, multi-GPU or
production claim. [spark-jitudf-benchmarking](https://github.com/beestream-ai/spark-jitudf-benchmarking) §8 has the full list, including the claims
that were withdrawn under review.

## License

spark-jitudf is released under the [Apache License, Version 2.0](LICENSE). Copyright (c) 2026
Beestream Ltd. Every source file carries the matching header.

Nothing third-party is included here; what the build and the evaluation fetch from elsewhere is
listed in [NOTICE](NOTICE). In particular, the RAPIDS changes this depends on live in
[`beestream-ai/cudf-spark`](https://github.com/beestream-ai/cudf-spark), stay under that project's
own Apache 2.0 licence and keep its file headers; nothing here relicenses them, and no file here
ships in NVIDIA's package namespace.
