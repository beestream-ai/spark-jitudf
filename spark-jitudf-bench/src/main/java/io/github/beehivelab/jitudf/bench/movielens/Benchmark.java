/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf.bench.movielens;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * One configuration of the MovieLens similarity application, measured in its own JVM.
 *
 * <p>
 * A separate process per configuration is deliberate. The RAPIDS plugin is installed when the
 * SparkSession is created, so toggling it inside one JVM would leave the two configurations sharing
 * a warmed JIT, a warmed allocator and a warmed GPU context. Running each in a fresh process costs
 * start-up time that is excluded from the measurement anyway, and removes a whole class of
 * cross-contamination. The driving script decides the order.
 * </p>
 *
 * <p>
 * Everything this writes is raw. One row per timed run, never a summary: a median can be computed
 * later from the rows, but rows cannot be recovered from a median.
 * </p>
 */
public final class Benchmark {

    public static void main(String[] args) throws IOException {
        String dataDir = System.getProperty("app.data", "third_party/ml-prepared");
        String mode = System.getProperty("app.mode", "tornado");
        int seeds = Integer.getInteger("app.seeds", 50);
        int pool = Integer.getInteger("app.pool", 200);
        int topK = Integer.getInteger("app.topk", 10);
        int dims = Integer.getInteger("app.dims", 0);
        int warmups = Integer.getInteger("app.warmups", 2);
        int runs = Integer.getInteger("app.runs", 5);
        String outDir = System.getProperty("app.out", "application-evaluation/results");
        String tag = System.getProperty("app.tag", "default");

        Files.createDirectories(Path.of(outDir));
        String config = String.format(Locale.ROOT, "%s_s%d_p%d_d%d_b%s_part%s", mode, seeds, pool, dims, //
                System.getProperty("app.batchBytes", "268435456"), System.getProperty("app.partitions", "2"));

        SparkSession spark = MovieSimilarityApp.session(mode);

        // Refuse to measure a configuration that is not the one asked for.
        //
        // app.expectMaster is set by the launcher to the same string it passes to spark-submit. If
        // the session Spark built disagrees, every number this process would produce is mislabelled,
        // so it is better to produce none. This exists because the opposite happened: a builder
        // default silently overrode the launcher and a whole thread sweep measured local[2].
        java.util.Map<String, String> effective = MovieSimilarityApp.effectiveResources(spark);
        String expectMaster = System.getProperty("app.expectMaster", "");
        if (!expectMaster.isEmpty() && !expectMaster.equals(effective.get("master"))) {
            spark.stop();
            throw new IllegalStateException("effective master " + effective.get("master")
                    + " does not match requested " + expectMaster
                    + "; refusing to record a run under a configuration it did not use");
        }
        System.out.println("EFFECTIVE " + effective);

        long pairs = (long) seeds * (pool - 1);

        // Cold: the first execution of this configuration in this JVM. For the TornadoVM
        // configuration it also pays for bytecode analysis, kernel generation and NVRTC, which is
        // why it is recorded on its own line rather than averaged into the warm runs.
        // With attribution on, each execution is measured on its own: the integration's cumulative
        // counters are zeroed first, so what they hold afterwards belongs to that execution alone.
        // The reset and the readback both happen outside the timed region.
        boolean attribute = Boolean.getBoolean("app.attribution");
        java.util.List<Attribution.Sample> samples = new java.util.ArrayList<>();

        if (attribute) {
            Attribution.reset();
        }
        long w0 = System.currentTimeMillis();
        long cls0 = Attribution.classesLoaded();
        long jit0 = Attribution.jitMillis();
        long t0 = System.nanoTime();
        Dataset<Row> coldQ = MovieSimilarityApp.buildQuery(spark, dataDir, seeds, pool, topK, dims);
        double coldBuildMillis = (System.nanoTime() - t0) / 1e6;
        java.util.List<Row> coldOut = coldQ.collectAsList();
        double coldMillis = (System.nanoTime() - t0) / 1e6;
        long coldRows = coldOut.size();
        recordEvidence(outDir, tag, mode, seeds, pool, dims, "cold", 0, coldMillis, coldOut, coldQ, effective, dataDir);
        if (attribute) {
            long[] br = Attribution.bridgeNanos(coldQ);
            samples.add(new Attribution.Sample("cold", 0, coldMillis, coldRows, Attribution.udfTotals(), Attribution.compileTotals(), br[0], br[1], w0, System.currentTimeMillis(), coldBuildMillis,
                    Attribution.classesLoaded() - cls0, Attribution.jitMillis() - jit0));
        }

        for (int i = 0; i < warmups; i++) {
            run(spark, dataDir, seeds, pool, topK, dims);
        }

        double[] millis = new double[runs];
        long rows = 0;
        for (int i = 0; i < runs; i++) {
            if (attribute) {
                Attribution.reset();
            }
            long w = System.currentTimeMillis();
            long cls = Attribution.classesLoaded();
            long jit = Attribution.jitMillis();
            long t = System.nanoTime();
            Dataset<Row> rq = MovieSimilarityApp.buildQuery(spark, dataDir, seeds, pool, topK, dims);
            double buildMillis = (System.nanoTime() - t) / 1e6;
            java.util.List<Row> out = rq.collectAsList();
            millis[i] = (System.nanoTime() - t) / 1e6;
            rows = out.size();
            // Evidence for THIS execution, captured after the timer stops. Previously the warm
            // rows borrowed the plan and output of a separate untimed query run afterwards, which
            // cannot show what any timed query actually did.
            recordEvidence(outDir, tag, mode, seeds, pool, dims, "warm", i, millis[i], out, rq, effective, dataDir);
            if (attribute) {
                long[] br = Attribution.bridgeNanos(rq);
                samples.add(new Attribution.Sample("warm", i, millis[i], rows, Attribution.udfTotals(), Attribution.compileTotals(), br[0], br[1], w, System.currentTimeMillis(), buildMillis,
                        Attribution.classesLoaded() - cls, Attribution.jitMillis() - jit));
            }
        }

        // The plan of a query that has actually executed: adaptive execution rewrites it as it runs,
        // so the plan of an unexecuted Dataset is the plan before any GPU rule touched it.
        Dataset<Row> q = MovieSimilarityApp.buildQuery(spark, dataDir, seeds, pool, topK, dims);
        List<Row> out = q.collectAsList();
        String plan = q.queryExecution().executedPlan().toString();
        Files.writeString(Path.of(outDir, "plan_" + config + ".txt"), plan);

        // The full result, for comparison against the CPU reference. Deterministic order.
        // The query returns rows ordered by seedId ASC, rk ASC, so a row's position within its
        // seed is its rank. Writing it explicitly means the correctness check compares the ranking
        // itself, not just the set of scores.
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(Path.of(outDir, "output_" + config + ".csv"), StandardCharsets.UTF_8))) {
            w.println("seedId,rk,candId,score");
            int previousSeed = Integer.MIN_VALUE;
            int rk = 0;
            for (Row r : out) {
                int seedId = r.getInt(0);
                rk = seedId == previousSeed ? rk + 1 : 1;
                previousSeed = seedId;
                // The native baseline returns float, the JVM UDF returns double; read whichever.
                w.printf(Locale.ROOT, "%d,%d,%d,%.17g%n", seedId, rk, r.getInt(2), ((Number) r.get(4)).doubleValue());
            }
        }

        // Operator metrics, harvested outside every timed region. Diagnostics, not a budget --
        // see PlanMetrics for why they must not be stacked.
        if (Boolean.getBoolean("app.metrics")) {
            PlanMetrics.write(Path.of(outDir, "plan_metrics.csv"), tag, mode, seeds, pool, dims,
                    PlanMetrics.collect(q.queryExecution().executedPlan()));
        }

        String udfPath = classify(plan);
        String metrics = metricsSummary();

        Path csv = Path.of(outDir, "runs.csv");
        boolean fresh = !Files.exists(csv);
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(csv, StandardCharsets.UTF_8, //
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (fresh) {
                w.println("tag,mode,seeds,pool,dims,batch_bytes,partitions,pairs,phase,run_index,millis,rows,udf_path");
            }
            w.printf(Locale.ROOT, "%s,%s,%d,%d,%d,%s,%s,%d,cold,0,%.3f,%d,%s%n", tag, mode, seeds, pool, dims, //
                    System.getProperty("app.batchBytes", "268435456"), System.getProperty("app.partitions", "2"), pairs, coldMillis, coldRows, udfPath);
            for (int i = 0; i < runs; i++) {
                w.printf(Locale.ROOT, "%s,%s,%d,%d,%d,%s,%s,%d,warm,%d,%.3f,%d,%s%n", tag, mode, seeds, pool, dims, //
                        System.getProperty("app.batchBytes", "268435456"), System.getProperty("app.partitions", "2"), pairs, i, millis[i], rows, udfPath);
            }
        }

        System.out.printf("RESULT mode=%s pairs=%d dims=%d cold=%.1fms warm=%s rows=%d udf=%s%n", //
                mode, pairs, dims, coldMillis, java.util.Arrays.toString(millis), rows, udfPath);
        System.out.println("METRICS " + metrics);
        System.out.println(compilePhases());
        if (Boolean.getBoolean("app.attribution")) {
            // The application id ties each row to its Spark event log, so the event-log timeline and
            // our own instrumentation can be compared run for run instead of by order.
            Attribution.write(Path.of(outDir, "attribution.csv"), tag, mode, seeds, pool, spark.sparkContext().applicationId(), samples);
            System.out.println("ATTRIBUTION rows=" + samples.size() + " -> " + outDir + "/attribution.csv");
        }
        spark.stop();
    }

    /**
     * One immutable record per timed execution: its own digest, its own executed plan, the
     * configuration Spark actually used and the identity of the code and data that produced it.
     *
     * <p>
     * Filenames carry tag, mode, size, phase, run index and the Spark application id, so two
     * experiments cannot collide, and {@link Provenance#writeNew} refuses to overwrite rather than
     * silently replacing an earlier artifact. All of this runs after the timer has stopped.
     * </p>
     */
    private static void recordEvidence(String outDir, String tag, String mode, int seeds, int pool, int dims, String phase, int index, double elapsedMillis,
            java.util.List<Row> out, Dataset<Row> q, java.util.Map<String, String> effective, String dataDir) {
        if (!Boolean.parseBoolean(System.getProperty("app.provenance", "true"))) {
            return;
        }
        String appId = effective.getOrDefault("appId", "unknown");
        String runId = String.format(Locale.ROOT, "%s_%s_s%d_p%d_d%d_%s_%s%d", tag, mode, seeds, pool, dims, appId, phase, index);
        Path dir = Path.of(outDir, "provenance");
        String plan = Provenance.executedPlan(q);
        Provenance.writeNew(dir.resolve("plan_" + runId + ".txt"), plan);

        java.util.Map<String, String> rec = new java.util.LinkedHashMap<>();
        rec.put("runId", runId);
        rec.put("tag", tag);
        rec.put("mode", mode);
        rec.put("seeds", String.valueOf(seeds));
        rec.put("pool", String.valueOf(pool));
        rec.put("pairs", String.valueOf((long) seeds * (pool - 1)));
        rec.put("dims", String.valueOf(dims));
        rec.put("phase", phase);
        rec.put("runIndex", String.valueOf(index));
        rec.put("elapsedMs", String.format(Locale.ROOT, "%.3f", elapsedMillis));
        rec.put("rows", String.valueOf(out.size()));
        rec.put("outputDigest", Provenance.digest(out));
        rec.put("planDigest", Provenance.textHash(plan));
        // The path this execution actually took, read from its own executed plan.
        rec.put("scoringPath", classify(plan));
        rec.putAll(effective);
        rec.put("dataset", Provenance.datasetIdentity(dataDir));
        rec.putAll(Provenance.sourceIdentity(System.getProperty("app.repoRoot", ".")));
        rec.put("integrationJar", Provenance.fileHash(Path.of(System.getProperty("app.integrationJar", ""))));
        rec.put("harnessJar", Provenance.fileHash(Path.of(System.getProperty("app.harnessJar", ""))));
        rec.put("wallClockMs", String.valueOf(System.currentTimeMillis()));
        Provenance.appendManifest(dir.resolve("manifest.jsonl"), rec);
    }

    private static long run(SparkSession spark, String dataDir, int seeds, int pool, int topK, int dims) {
        return MovieSimilarityApp.buildQuery(spark, dataDir, seeds, pool, topK, dims).collectAsList().size();
    }

    /** Where the UDF actually ran, read from the executed plan rather than assumed. */
    static String classify(String plan) {
        // Order matters: "gpu_cosine" is a prefix of "gpu_cosine_native", so the native case has to
        // be tested first or it is misreported as a TornadoVM kernel.
        boolean nativeKernel = plan.contains("gpu_cosine_native");
        if (plan.contains("gpu_cosine_f32")) {
            return plan.contains("GpuCpuBridge") ? "MIXED_KERNEL_AND_BRIDGE" : "TORNADOVM_KERNEL_F32";
        }
        boolean kernel = !nativeKernel && plan.contains("gpu_cosine");
        boolean bridge = plan.contains("GpuCpuBridge");
        boolean gpuProject = plan.contains("GpuProject");
        if (nativeKernel && !bridge) {
            return "NVIDIA_NATIVE_KERNEL";
        }
        if (nativeKernel) {
            return "MIXED_NATIVE_AND_BRIDGE";
        }
        if (kernel && !bridge) {
            return "TORNADOVM_KERNEL";
        }
        if (bridge && !kernel) {
            return "RAPIDS_CPU_BRIDGE";
        }
        if (kernel) {
            return "MIXED_KERNEL_AND_BRIDGE";
        }
        return gpuProject ? "GPU_NATIVE_NO_UDF" : "CPU";
    }

    private static String metricsSummary() {
        try {
            Class<?> m = Class.forName("io.github.beehivelab.jitudf.TornadoUdfMetrics");
            return String.valueOf(m.getMethod("summary").invoke(null));
        } catch (ReflectiveOperationException e) {
            return "no TornadoVM metrics (configuration does not use the kernel path)";
        }
    }

    /**
     * The one-off cost of turning the UDF into a kernel, split by phase.
     *
     * <p>
     * Reported on its own line rather than folded into the cold run's time. The cold run also pays
     * for Spark's own JIT warm-up, a cold page cache and GPU context creation, none of which this
     * integration causes and all of which the RAPIDS baseline pays too; calling the whole of the
     * cold excess "compilation" would be wrong.
     * </p>
     */
    private static String compilePhases() {
        try {
            Class<?> m = Class.forName("io.github.beehivelab.jitudf.CompilePhases");
            return String.valueOf(m.getMethod("summary").invoke(null));
        } catch (ReflectiveOperationException e) {
            return "COMPILE-PHASES none (configuration does not compile a kernel)";
        }
    }

    private Benchmark() {
    }
}
