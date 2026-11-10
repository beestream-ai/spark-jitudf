/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitudf.bench.movielens;

import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import ai.beestream.jitudf.bench.udfs.CosineSimilarityUdf;

import scala.collection.Seq;

/**
 * "More like this" over the MovieLens Tag Genome.
 *
 * <p>
 * For each of the most-rated <i>seed</i> movies, score every movie in a bounded candidate pool by
 * the cosine similarity of their 1128-dimensional tag-relevance vectors, then return the top-K most
 * similar titles per seed. The vectors are the dataset's own tag relevance values; the candidate
 * pool is the most-rated movies, which is a policy that needs no model, no randomness and no
 * Cartesian product over the whole catalogue.
 * </p>
 *
 * <p>
 * The scoring function is an ordinary JVM UDF ({@link CosineSimilarityUdf}) and is <b>identical in
 * every configuration</b>. Only the engine configuration changes: plain Spark, RAPIDS with the UDF
 * on its CPU bridge, or RAPIDS with the UDF compiled to a GPU kernel.
 * </p>
 *
 * <p>
 * Shape of the query, which is what makes this an application rather than a microbenchmark: a
 * Parquet scan, two joins to attach feature vectors, a filter, the UDF projection, a ranking window
 * and a top-K selection. The UDF is one operator among several, so the end-to-end number is bounded
 * by everything around it as well.
 * </p>
 */
public final class MovieSimilarityApp {

    public static void main(String[] args) {
        String dataDir = arg(args, 0, "third_party/ml-prepared");
        int seeds = Integer.parseInt(arg(args, 1, "50"));
        int poolSize = Integer.parseInt(arg(args, 2, "200"));
        int topK = Integer.parseInt(arg(args, 3, "10"));
        String mode = arg(args, 4, "tornado");
        int dims = Integer.parseInt(arg(args, 5, "0")); // 0 = use the full genome

        SparkSession spark = session(mode);
        Dataset<Row> ranked = buildQuery(spark, dataDir, seeds, poolSize, topK, dims);

        long t0 = System.nanoTime();
        List<Row> out = ranked.collectAsList();
        double millis = (System.nanoTime() - t0) / 1e6;

        String plan = ranked.queryExecution().executedPlan().toString();
        System.out.println("=== PHYSICAL PLAN (" + mode + ") ===");
        for (String line : plan.split("\n")) {
            System.out.println("PLAN| " + line);
        }
        System.out.printf("=== %s: %d rows in %.1f ms ===%n", mode, out.size(), millis);
        for (int i = 0; i < Math.min(5, out.size()); i++) {
            System.out.println("TOPK| " + out.get(i));
        }
        ai.beestream.jitudf.bench.IntegrationTest.printDecisions();
        spark.stop();
    }

    /**
     * The query under test. Deliberately built once and shared by every configuration so that no
     * baseline can differ in partitioning, candidate selection, ordering or semantics.
     */
    /**
     * What Spark actually gave us, as opposed to what was requested.
     *
     * <p>
     * {@code defaultParallelism} is the number Spark derives from the master string, so it is the
     * check that catches a master that parsed differently from the one intended. Task concurrency
     * is not reported here because a session cannot know it: it is recovered per stage from the
     * event log, where overlapping task intervals are what concurrency actually means.
     * </p>
     */
    public static java.util.Map<String, String> effectiveResources(SparkSession spark) {
        java.util.Map<String, String> m = new java.util.LinkedHashMap<>();
        m.put("master", spark.sparkContext().master());
        m.put("appId", spark.sparkContext().applicationId());
        m.put("defaultParallelism", String.valueOf(spark.sparkContext().defaultParallelism()));
        m.put("executorCores", spark.conf().getOption("spark.executor.cores").getOrElse(() -> ""));
        m.put("shufflePartitions", spark.conf().get("spark.sql.shuffle.partitions", ""));
        m.put("availableProcessors", String.valueOf(Runtime.getRuntime().availableProcessors()));
        return m;
    }

    public static Dataset<Row> buildQuery(SparkSession spark, String dataDir, int seeds, int poolSize, int topK, int dims) {
        // Read with an explicit non-nullable schema. Spark's Parquet reader marks every field
        // nullable regardless of what was written, and a nullable column is refused by the kernel
        // compiler because a null has no defined meaning inside a kernel. Declaring the schema here
        // is sound because PrepareData checked it: no nulls, every vector the full genome length,
        // and it fails rather than writes if that is ever untrue.
        // Spark's Parquet reader marks every field nullable and overrides any schema given to it,
        // so nullability has to be re-established with an expression the analyser can reason about.
        // This normalisation says what the data already is: PrepareData verified there are no null
        // vectors and no null elements, and refuses to write if that is ever untrue. It runs once
        // over the 13,816 movie rows, before any pairing, and it is applied identically in every
        // configuration, so no baseline is advantaged by it.
        // The native baseline is typed on float and is a different UDF, so it reads the float copy
        // of the same vectors. Everything else about the query -- candidate policy, joins, window,
        // ranking, ordering -- is identical, which is the only way the comparison means anything.
        String featureExpr = (NATIVE.get() || FLOAT32.get())
                ? "transform(coalesce(features_float, array()), x -> coalesce(x, cast(0.0 as float))) AS features"
                : "transform(coalesce(features, array()), x -> coalesce(x, 0.0D)) AS features";
        Dataset<Row> vectors = spark.read().parquet(dataDir + "/movie_vectors") //
                .selectExpr("movieId", "title", "genres", "nRatings", featureExpr);
        if (dims > 0) {
            // A reduced feature dimension, taken as a prefix of the fixed tag order so that every
            // row is cut at the same tags and the comparison stays meaningful.
            vectors = vectors.selectExpr("movieId", "title", "genres", "nRatings", "slice(features, 1, " + dims + ") as features");
        }
        vectors.createOrReplaceTempView("movie_vectors");

        // Deterministic: ordered by popularity, ties broken by movieId, so every run and every
        // configuration sees exactly the same pool and the same seeds.
        // The candidate pool. A global LIMIT makes Spark produce it in a single partition, which
        // means the join and the scoring that follow run in one task however many threads the
        // driver has. That is Spark's own plan for this query and it is what every configuration
        // gets, so it does not bias the comparison -- but it does mean the CPU baseline is one
        // core's work rather than the whole machine's. `-Dapp.repartition=N` re-partitions the
        // pool so the alternative can be measured instead of argued about; it changes the plan, so
        // runs that use it are a separate experiment and are never mixed into the headline table.
        Dataset<Row> pool = spark.sql("SELECT * FROM movie_vectors ORDER BY nRatings DESC, movieId ASC LIMIT " + poolSize);
        int repartition = Integer.getInteger("app.repartition", 0);
        if (repartition > 0) {
            pool = pool.repartition(repartition);
        }
        pool.createOrReplaceTempView("pool");
        // Seeds come from the same ordering and stay a single small broadcast relation whether or
        // not the pool was re-partitioned, so the build side of the join is unchanged either way.
        spark.sql("SELECT * FROM pool ORDER BY nRatings DESC, movieId ASC LIMIT " + seeds).createOrReplaceTempView("seeds");

        // The score is computed in an inner select and the window orders by that column, rather
        // than repeating the expression inside OVER(...). Written the obvious way -- the scoring
        // expression in both the projection and the ORDER BY -- Catalyst materialises it twice, as
        // `score` and again as a hidden `_w1`, so every candidate pair is scored twice. That is
        // uniform across configurations and so does not bias the comparison, but it doubles the
        // scoring work in all of them and makes the measured cost of scoring wrong by 2x.
        //
        // `app.duplicate.score=true` restores the two-evaluation form, which is how the duplication
        // is demonstrated rather than asserted.
        boolean duplicate = Boolean.getBoolean("app.duplicate.score");
        String orderBy = duplicate ? score("sf", "cf") : "score";
        return spark.sql( //
                "SELECT seedId, seedTitle, candId, candTitle, score FROM (" //
                        + "  SELECT *, row_number() OVER (PARTITION BY seedId ORDER BY " + orderBy + " DESC, candId ASC) AS rk FROM (" //
                        + "    SELECT s.movieId AS seedId, s.title AS seedTitle, c.movieId AS candId, c.title AS candTitle, " //
                        + "           " + score("s.features", "c.features") + " AS score, s.features AS sf, c.features AS cf " //
                        + "    FROM seeds s JOIN pool c ON s.movieId <> c.movieId" //
                        + "  )" //
                        + ") WHERE rk <= " + topK + " ORDER BY seedId ASC, rk ASC");
    }

    /**
     * The schema the prepared Parquet is known to satisfy, kept as the written contract that
     * {@link PrepareData} enforces. It is not passed to the reader: Spark's Parquet source calls
     * {@code asNullable} on any schema given to it, so nullability has to be re-established with an
     * expression instead. See {@link #buildQuery}.
     */
    public static final StructType VECTOR_SCHEMA = new StructType() //
            .add("movieId", DataTypes.IntegerType, false) //
            .add("title", DataTypes.StringType, false) //
            .add("genres", DataTypes.StringType, false) //
            .add("nRatings", DataTypes.IntegerType, false) //
            .add("features", DataTypes.createArrayType(DataTypes.DoubleType, false), false);

    /** True when this process is measuring NVIDIA's native UDF rather than the JVM one. */
    private static final ThreadLocal<Boolean> NATIVE = ThreadLocal.withInitial(() -> false);

    /** True when this process scores with the float32 JVM UDF, for the precision-matched run. */
    private static final ThreadLocal<Boolean> FLOAT32 = ThreadLocal.withInitial(() -> false);

    /** True when this process scores with a pure SQL expression instead of a UDF. */
    private static final ThreadLocal<Boolean> SQL = ThreadLocal.withInitial(() -> false);

    /**
     * The scoring expression for this configuration.
     *
     * <p>
     * The SQL form computes exactly what {@link CosineSimilarityUdf} computes, including its
     * zero-denominator rule, using only built-in array functions. It exists to answer "what if the
     * user simply rewrote the function in SQL?" -- which is the alternative anyone would suggest,
     * and which is only available when the function can be expressed that way at all.
     * </p>
     */
    static String score(String a, String b) {
        if (SQL.get()) {
            String dot = "aggregate(zip_with(" + a + ", " + b + ", (x, y) -> x * y), 0D, (acc, v) -> acc + v)";
            String na = "sqrt(aggregate(" + a + ", 0D, (acc, x) -> acc + x * x))";
            String nb = "sqrt(aggregate(" + b + ", 0D, (acc, x) -> acc + x * x))";
            return "(CASE WHEN (" + na + ") * (" + nb + ") = 0D THEN 0D ELSE (" + dot + ") / ((" + na + ") * (" + nb + ")) END)";
        }
        return (NATIVE.get() ? "cosine_native" : FLOAT32.get() ? "cosine_f32" : "cosine") + "(" + a + ", " + b + ")";
    }

    public static SparkSession session(String mode) {
        boolean rapids = !"cpu".equals(mode) && !"sqlcpu".equals(mode);
        boolean tornadoFloat = "float32".equals(mode);
        if ("sqlcpu".equals(mode)) {
            SQL.set(true);
        }
        boolean tornado = "tornado".equals(mode) || "float32".equals(mode);
        SparkSession.Builder b = SparkSession.builder() //
                .appName("movielens-similarity-" + mode);
        // One authority for the master setting, and it is the launcher.
        //
        // This used to read .master(System.getProperty("app.master", "local[2]")), which an
        // explicit builder call wins over spark-submit's --master. The effect was that
        // BENCH_MASTER never reached this application: every run that believed it was sweeping
        // CPU threads actually ran local[2]. Any conclusion drawn from such a sweep is void.
        //
        // Now the builder only sets a master when one is given explicitly, so with no property
        // the session inherits --master from spark-submit. Benchmark verifies afterwards that the
        // session Spark actually built matches what was asked for, and fails the run otherwise --
        // a configuration that silently disagrees with its description is worse than no run.
        String master = System.getProperty("app.master", "");
        if (!master.isEmpty()) {
            b = b.master(master);
        }
        b = b //
                .config("spark.sql.shuffle.partitions", System.getProperty("app.partitions", "2")) //
                .config("spark.rapids.sql.enabled", String.valueOf(rapids)) //
                .config("spark.rapids.sql.tornadovm.enabled", String.valueOf(tornado));
        if (rapids) {
            b = b.config("spark.plugins", "com.nvidia.spark.SQLPlugin") //
                    .config("spark.rapids.sql.udfCompiler.enabled", "true") //
                    .config("spark.rapids.sql.explain", System.getProperty("app.explain", "NONE")) //
                    .config("spark.rapids.memory.gpu.pool", "ASYNC") //
                    .config("spark.rapids.memory.gpu.allocFraction", "0.3") //
                    .config("spark.rapids.memory.gpu.minAllocFraction", "0.05") //
                    .config("spark.rapids.memory.gpu.maxAllocFraction", "0.5") //
                    .config("spark.rapids.memory.gpu.reserve", "805306368") //
                    .config("spark.rapids.sql.batchSizeBytes", System.getProperty("app.batchBytes", "268435456"))
                    // Per-operator timing is gated behind the metrics level. DEBUG is only used for
                    // the profiling runs, never the headline timing runs, and its own cost is
                    // measured against an uninstrumented control.
                    .config("spark.rapids.sql.metrics.level", System.getProperty("app.metricsLevel", "MODERATE"));
        }
        // Spark's event log is the only record of where elapsed time actually went: when each stage
        // started and ended, and how many tasks ran in it. An attribution built from operator
        // counters cannot tell concurrent task-seconds from critical-path seconds; the event log
        // can. Off unless asked for, because it writes a file per run.
        String eventLog = System.getProperty("app.eventLog", "");
        if (!eventLog.isEmpty()) {
            new java.io.File(eventLog).mkdirs();
            b = b.config("spark.eventLog.enabled", "true")
                    .config("spark.eventLog.dir", new java.io.File(eventLog).toURI().toString())
                    .config("spark.eventLog.compress", "false");
        }
        SparkSession spark = b.getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        // The same UDF object in every JVM-UDF configuration. Nothing about it changes between
        // B0, B1 and B2 -- only the engine configuration does.
        spark.udf().register("cosine", new CosineSimilarityUdf(), DataTypes.DoubleType);
        if ("sql".equals(mode)) {
            SQL.set(true);
        }
        if ("float32".equals(mode)) {
            // The same float vectors NVIDIA's UDF reads, scored by an ordinary JVM UDF that this
            // integration compiles. The only remaining difference from B3 is who wrote the kernel.
            FLOAT32.set(true);
            spark.udf().register("cosine_f32", new ai.beestream.jitudf.bench.udfs.CosineSimilarityFloatUdf(), DataTypes.FloatType);
        }
        if ("native".equals(mode)) {
            NATIVE.set(true);
            // NVIDIA's own class. It implements RapidsUDF, so RAPIDS calls its evaluateColumnar
            // directly and the JVM call() path is never used on the GPU plan.
            // Supplied on the classpath rather than vendored here, so this repository ships no
            // classes in NVIDIA's package namespace: spark-jitudf-benchmarking's scripts/build-native-udf.sh builds it from
            // NVIDIA's own RAPIDS-accelerated-UDFs examples. Absent, only this mode is lost.
            spark.udf().register("cosine_native", nvidiaCosineUdf(), DataTypes.FloatType);
        }
        return spark;
    }

    /**
     * NVIDIA's own cosine-similarity UDF (B3), loaded reflectively because it is not a build
     * dependency of this repository. It implements {@code RapidsUDF}, so RAPIDS calls its
     * {@code evaluateColumnar} directly and the JVM {@code call()} path is never used on the GPU
     * plan.
     */
    @SuppressWarnings("unchecked")
    private static UDF2<Seq<Float>, Seq<Float>, Float> nvidiaCosineUdf() {
        String name = "com.nvidia.spark.rapids.udf.java.CosineSimilarity";
        try {
            return (UDF2<Seq<Float>, Seq<Float>, Float>) Class.forName(name)
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(name + " is not on the classpath; build it with "
                    + "spark-jitudf-benchmarking's scripts/build-native-udf.sh, or run a mode other than 'native'", e);
        }
    }

    private static String arg(String[] args, int i, String fallback) {
        return args.length > i ? args[i] : fallback;
    }

    private MovieSimilarityApp() {
    }
}
