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

For a UDF the plugin cannot translate, **turning RAPIDS on makes the measured application 1.44x to
1.76x slower than not using the GPU at all.** Each batch leaves the GPU, becomes rows, is scored in
the JVM, and returns. That is the right thing for the plugin to do — correct answers beat refusing
— and it is the cost this removes.

Against that baseline, on a MovieLens similarity query at 199,800 candidate pairs:
**4.89x faster** [4.67, 5.27] for the query as written, or **2.43x** [2.23, 2.66] against a CPU
fallback given the whole machine. Both numbers, what separates them, and what the evaluation does
*not* establish are in [EVALUATION.md](EVALUATION.md).

## What is here

| Path | |
|---|---|
| `spark-jitudf-plugin/` | The module RAPIDS calls into: UDF resolution, eligibility analysis, ASM lowering, kernel generation, cuDF buffer adapters, caches. |
| `spark-jitudf-bench/` | The runnable end: the end-to-end test, the microbenchmark, the MovieLens application the evaluation measures, the probes, and the UDFs they all use. |
| `scripts/` | `env.sh` (paths, all overridable), `run-spark.sh` (launcher), `build-rapids-patch.sh`, `tvm-gate.sh`, the benchmark driver and the correctness checker. |
| `evaluation/` | The result data and figures behind EVALUATION.md. |
| [EVALUATION.md](EVALUATION.md) | What was measured, the numbers, and their limits. |
| [REPRODUCING.md](REPRODUCING.md) | How to re-run all of it. |

## Getting it running

One other repository is needed: the RAPIDS fork branch that makes the UDF compiler say *why* it
declined a UDF and lets something else try. Apache Spark is unmodified — the stock 4.0.4 binary
distribution is what runs here.

```bash
git clone https://github.com/NVIDIA/cudf-spark third_party/cudf-spark
git -C third_party/cudf-spark checkout feature/tornadovm-udf-fallback

export JAVA_HOME=/path/to/jdk-21
export TORNADO_SDK=/path/to/tornadovm/dist/tornadovm-*-cuda
export SPARK_HOME=/path/to/spark-4.0.4-bin-hadoop3

./scripts/build-rapids-patch.sh     # the fork's changes into the released RAPIDS jar
mvn package                         # both modules, and the plugin's 33 unit tests
./scripts/run-spark.sh io.github.beehivelab.jitudf.bench.IntegrationTest
```

The last command must end `CHECKS: 27, FAILURES: 0`. Full prerequisites, the MovieLens data and
the measurement protocol are in [REPRODUCING.md](REPRODUCING.md).

It is **off by default**. Nothing changes until `spark.rapids.sql.tornadovm.enabled=true` is set.

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

Research code. One application, one dataset, one laptop GPU, Spark `local` mode. No cluster,
multi-GPU or production claim. See [EVALUATION.md](EVALUATION.md) §8 for the full list, including
the claims that were withdrawn under review.

## License

Apache License 2.0; the full text is in [LICENSE](LICENSE). Every source file carries the matching
header.

The RAPIDS changes this depends on stay under that project's own Apache 2.0 licence and keep its
file headers; nothing here relicenses them, and no file here ships in NVIDIA's package namespace.
