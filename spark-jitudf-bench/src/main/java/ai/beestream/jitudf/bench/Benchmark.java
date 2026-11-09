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
package ai.beestream.jitudf.bench;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.bench.udfs.CosineSimilarityUdf;
import ai.beestream.jitudf.bench.udfs.DataDependentLoopUdf;
import ai.beestream.jitudf.bench.udfs.HelperLoopUdf;
import ai.beestream.jitudf.bench.udfs.LoopUdf;
import ai.beestream.jitudf.bench.udfs.MathUdf;
import ai.beestream.jitudf.bench.udfs.StringUdf;

/**
 * End-to-end Spark SQL timings for the same query under each configuration, on this laptop.
 *
 * <p>
 * Every number written here is a wall-clock measurement of a real Spark action over real data. The
 * query is forced to completion with an aggregation that consumes every row, so nothing is left
 * lazy and no partial result is timed; the aggregation also keeps the batch on the GPU after the
 * UDF, which is the point of returning a cuDF column rather than rows.
 * </p>
 *
 * <p>
 * Configurations:
 * </p>
 * <ul>
 * <li><b>B0</b> plain Spark, RAPIDS off: the UDF runs row by row on the CPU, as it would without
 * any of this.</li>
 * <li><b>B1</b> RAPIDS on, this integration off: RAPIDS declines the UDF and evaluates it through
 * its own CPU bridge inside a GPU plan.</li>
 * <li><b>B3</b> RAPIDS on, this integration on: the UDF is compiled to a kernel and reads the cuDF
 * buffers in place through device views.</li>
 * <li><b>B5</b> the same arithmetic written as a Catalyst expression, where the workload can be
 * expressed that way at all: what RAPIDS alone can already do, for scale.</li>
 * </ul>
 */
public final class Benchmark {

    private static final int WARMUP_RUNS = 3;
    private static final int MEASURED_RUNS = 9;

    /**
     * How many rows are compared element by element against the CPU, on top of the whole-column
     * summaries. An aggregate agreeing is necessary, not sufficient: two columns that differ in two
     * rows by equal and opposite amounts have the same sum.
     */
    private static final int ELEMENTWISE_ROWS = 100_000;

    /**
     * Seeds the order the configurations are measured in.
     *
     * <p>
     * Running B0, B1, B3, B5 in that order every time confounds the configuration with its position
     * in the sequence: later runs see a warmer page cache, a hotter JIT and a different thermal
     * state, and whatever that is worth accrues to the same configuration every time. The order is
     * therefore permuted per workload, size and batch size, deterministically from this seed so
     * that a run can be repeated exactly, and the order actually used is written into the CSV.
     * </p>
     */
    private static final long ORDER_SEED = Long.getLong("bench.order.seed", 20261030L);

    /**
     * One row per timed run, kept raw.
     *
     * <p>
     * A median is a summary, and a summary cannot be re-examined. Writing every observation means a
     * later question about spread, drift or an outlier can be answered from the data rather than by
     * running the whole thing again and hoping the machine was in the same state.
     * </p>
     */
    private static final List<String[]> rawRuns = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        Path csv = Path.of(args.length > 0 ? args[0] : "benchmarks/results.csv");
        Path breakdownCsv = csv.resolveSibling("breakdown.csv");
        Files.createDirectories(csv.getParent());

        // 2 M rows was dropped after a control showed it unmeasurable on this machine: B0 has RAPIDS
        // disabled, so the batch size cannot affect it, yet two of its 2 M cells moved by more than
        // 15% across that setting. At 8 M and above the same control agrees to within 3%.
        int[] sizes = { 8_000_000, 16_000_000 };
        int[] batchBytes = { 64 * 1024 * 1024, 256 * 1024 * 1024 };

        List<String[]> rows = new ArrayList<>();
        List<String[]> breakdown = new ArrayList<>();

        for (int batchSizeBytes : batchBytes) {
            SparkSession spark = session(batchSizeBytes);
            try {
                for (int size : sizes) {
                    // The control. No UDF at all, just arithmetic RAPIDS compiles itself, so B1 is
                    // RAPIDS doing what it is good at rather than RAPIDS carrying a UDF it cannot
                    // compile. Without this the suite measures the plugin only on the one workload
                    // it declares out of scope, which says nothing about the plugin and nothing
                    // about whether this harness can detect a GPU speedup at all. It also checks
                    // that B3 leaves a UDF-free query alone: with nothing to compile, enabling this
                    // integration must change neither the plan nor the time.
                    String controlSql = "cbrt(a * 1.5 + b * 0.25 - c) + sqrt(a*a + b*b + c*c) * 0.5";
                    measure(spark, rows, breakdown, "W0-sql-no-udf-control", controlSql, controlSql, size, batchSizeBytes);
                    measure(spark, rows, breakdown, "W1-fixed-width-math", "tornado_math(a, b, c)", "cbrt(a * 1.5 + b * 0.25 - c) + sqrt(a*a + b*b + c*c) * 0.5", size, batchSizeBytes);
                    measure(spark, rows, breakdown, "W2-counted-loop", "tornado_loop(a, b)", null, size, batchSizeBytes);
                    measure(spark, rows, breakdown, "W2b-data-dependent-loop", "tornado_ddloop(a, b)", null, size, batchSizeBytes);
                    measure(spark, rows, breakdown, "W2c-loop-static-helper", "tornado_helper(a, b)", null, size, batchSizeBytes);
                    measure(spark, rows, breakdown, "W3-list-cosine-similarity", "tornado_cosine(vec_a, vec_b)", null, size, batchSizeBytes);

                    // Three UDFs in one projection, all three of them eligible. Each becomes its own
                    // generated class and its own cached plan, and the batch is bound three times.
                    // The question is whether the per-kernel cost is paid three times over or
                    // whether the batch, which is already resident, is what matters.
                    measure(spark, rows, breakdown, "W4-three-kernels", //
                            "tornado_math(a, b, c) + tornado_loop(a, b) + tornado_helper(a, b)", null, size, batchSizeBytes);

                    // The realistic mixed case: one UDF that compiles to a kernel beside one that
                    // nothing can compile, so a column still has to make the round trip whatever
                    // happens. This is the Amdahl case, and it is the one a user is most likely to
                    // actually have.
                    measure(spark, rows, breakdown, "W5-kernel-plus-bridged", //
                            "tornado_loop(a, b) + length(tornado_string(a))", null, size, batchSizeBytes);
                }
            } finally {
                spark.stop();
            }
        }

        write(csv, "workload,rows,partitions,batch_size_bytes,mode,runs,median_ms,min_ms,max_ms,p10_ms,p90_ms,rows_per_sec,correctness,physical_plan,notes", rows);
        write(breakdownCsv, "workload,rows,batch_size_bytes,mode,stage,median_us,min_us,max_us,count,method_of_measurement", breakdown);
        write(csv.resolveSibling("runs.csv"), "workload,rows,batch_size_bytes,mode,run_index,millis,checksum", rawRuns);
        System.out.println("wrote " + csv.toAbsolutePath() + ", " + breakdownCsv.toAbsolutePath() + " and " + csv.resolveSibling("runs.csv").toAbsolutePath());
    }

    static SparkSession session(int batchSizeBytes) {
        SparkSession spark = SparkSession.builder() //
                .appName("tornado-rapids-benchmark") //
                .master("local[2]") //
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin") //
                .config("spark.rapids.sql.enabled", "true") //
                .config("spark.rapids.sql.udfCompiler.enabled", "true") //
                .config("spark.rapids.sql.explain", "NONE") //
                .config("spark.rapids.memory.gpu.pool", "ASYNC") //
                .config("spark.rapids.memory.gpu.allocFraction", "0.35") //
                .config("spark.rapids.memory.gpu.minAllocFraction", "0.05") //
                .config("spark.rapids.memory.gpu.maxAllocFraction", "0.6") //
                .config("spark.rapids.memory.gpu.reserve", "805306368") //
                .config("spark.rapids.sql.batchSizeBytes", String.valueOf(batchSizeBytes)) //
                .config("spark.sql.shuffle.partitions", "2") //
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        spark.udf().register("tornado_loop", new LoopUdf(64), DataTypes.DoubleType);
        spark.udf().register("tornado_ddloop", new DataDependentLoopUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_helper", new HelperLoopUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_math", new MathUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_cosine", new CosineSimilarityUdf(), DataTypes.DoubleType);
        // W5 needs a UDF that nothing can compile, so that one column of the projection is forced
        // through the CPU bridge no matter what else the query does.
        spark.udf().register("tornado_string", new StringUdf(), DataTypes.StringType);
        return spark;
    }

    static Dataset<Row> data(SparkSession spark, int rows) {
        // vec_a varies in length by row and vec_b does not, so the LIST workload has to read the
        // offsets rather than assume a stride. if(...) keeps both columns non-nullable, which a
        // remainder would not.
        return spark.range(0, rows, 1, 2) //
                .selectExpr("cast(id as double) * 0.5 + 1.0 as a", //
                        "cast(id as double) * 0.25 + 2.0 as b", //
                        "cast(id as double) * 0.125 + 3.0 as c") //
                .selectExpr("a", "b", "c", //
                        "if(a < 500000.0, array(a, b, c), array(a, b, c, a + b)) as vec_a", //
                        "array(c, b, a) as vec_b");
    }

    /** Everything one configuration produced, kept so that comparisons happen after all of them ran. */
    private record ModeResult(String mode, double coldMillis, double[] millis, double[] checksums, double[] summary, double[] sample, String plan, List<String[]> breakdown) {
    }

    /**
     * {@code -Dbench.workloads=W0,W4,W5} measures only those, so a question about one workload does
     * not require re-measuring every other one. Modes within a cell are still all measured together,
     * which is what makes a cell's own comparisons valid; comparing across runs is not.
     */
    private static final String ONLY = System.getProperty("bench.workloads", "");

    private static void measure(SparkSession spark, List<String[]> rows, List<String[]> breakdown, String workload, String udfExpression, String catalystEquivalent, int size,
            int batchSizeBytes) {
        if (!ONLY.isEmpty() && Arrays.stream(ONLY.split(",")).noneMatch(w -> workload.startsWith(w.trim()))) {
            return;
        }
        List<String> modes = new ArrayList<>(List.of("B0-cpu-no-rapids", "B1-rapids-cpu-bridge", "B3-tornadovm-deviceview", "B5-catalyst-expression"));
        if (catalystEquivalent == null) {
            modes.remove("B5-catalyst-expression"); // this workload has no Catalyst equivalent, which is why it is here
        }
        Collections.shuffle(modes, new Random(ORDER_SEED + Objects.hash(workload, size, batchSizeBytes)));
        String order = "order=" + String.join(">", modes) + " seed=" + ORDER_SEED;

        Map<String, ModeResult> results = new LinkedHashMap<>();
        for (String mode : modes) {
            results.put(mode, runMode(spark, workload, mode, mode.equals("B5-catalyst-expression") ? catalystEquivalent : udfExpression, size, batchSizeBytes));
        }

        ModeResult reference = results.get("B0-cpu-no-rapids");
        for (String mode : List.of("B0-cpu-no-rapids", "B1-rapids-cpu-bridge", "B3-tornadovm-deviceview", "B5-catalyst-expression")) {
            ModeResult result = results.get(mode);
            if (result == null) {
                continue;
            }
            double[] sorted = result.millis().clone();
            Arrays.sort(sorted);
            double median = median(sorted);
            double coldMillis = result.coldMillis();
            String correctness = compare(reference, result);

            rows.add(new String[] { workload, String.valueOf(size), "2", String.valueOf(batchSizeBytes), mode + "-FIRST-RUN", "1", f(coldMillis), f(coldMillis), f(coldMillis),
                    f(coldMillis), f(coldMillis), f(size / (coldMillis / 1000.0)), "NOT_CHECKED", "not-inspected",
                    "first run of this configuration in this session: includes Spark planning, JIT and, for B3, kernel generation and NVRTC. It does NOT isolate compilation; "
                            + "the per-plan first-batch figures in breakdown.csv are the closer measurement of that." });

            String notes = order;
            if (reference != null) {
                double[] referenceSorted = reference.millis().clone();
                Arrays.sort(referenceSorted);
                notes += String.format(Locale.ROOT, " speedup_vs_B0=%.2fx", median(referenceSorted) / median);
            }
            ModeResult bridge = results.get("B1-rapids-cpu-bridge");
            if (bridge != null) {
                double[] bridgeSorted = bridge.millis().clone();
                Arrays.sort(bridgeSorted);
                notes += String.format(Locale.ROOT, " speedup_vs_B1=%.2fx", median(bridgeSorted) / median);
            }

            rows.add(new String[] { workload, String.valueOf(size), "2", String.valueOf(batchSizeBytes), mode, String.valueOf(MEASURED_RUNS), f(median), f(sorted[0]),
                    f(sorted[sorted.length - 1]), f(percentile(sorted, 10)), f(percentile(sorted, 90)), f(size / (median / 1000.0)), correctness, result.plan(), notes });

            for (int i = 0; i < result.millis().length; i++) {
                rawRuns.add(new String[] { workload, String.valueOf(size), String.valueOf(batchSizeBytes), mode, String.valueOf(i), f(result.millis()[i]),
                        String.format(Locale.ROOT, "%.17g", result.checksums()[i]) });
            }
            breakdown.addAll(result.breakdown());
            System.out.printf("%-26s %-26s rows=%d batch=%dMB median=%.1f ms (%s)%n", workload, mode, size, batchSizeBytes / (1024 * 1024), median, correctness);
        }
    }

    private static ModeResult runMode(SparkSession spark, String workload, String mode, String expression, int size, int batchSizeBytes) {
        configure(spark, mode);
        resetMetrics();

        long coldStart = System.nanoTime();
        runOnce(spark, size, expression);
        double coldMillis = (System.nanoTime() - coldStart) / 1e6;

        for (int i = 0; i < WARMUP_RUNS; i++) {
            runOnce(spark, size, expression);
        }
        if (mode.equals("B3-tornadovm-deviceview")) {
            resetMetrics(); // keep only the steady-state per-batch figures
        }

        double[] millis = new double[MEASURED_RUNS];
        double[] checksums = new double[MEASURED_RUNS];
        for (int i = 0; i < MEASURED_RUNS; i++) {
            long start = System.nanoTime();
            checksums[i] = runOnce(spark, size, expression);
            millis[i] = (System.nanoTime() - start) / 1e6;
        }

        // The breakdown is snapshotted here, before any further query runs. Everything below adds
        // batches of its own, and a batch that was never timed has no business in a table of
        // per-batch timings.
        List<String[]> breakdown = new ArrayList<>();
        if (mode.equals("B3-tornadovm-deviceview")) {
            collectBreakdown(breakdown, workload, size, batchSizeBytes);
        }

        double[] summary = summarise(spark, size, expression);
        double[] sample = sample(spark, size, expression);
        String plan = planSummary(spark, size, expression);
        return new ModeResult(mode, coldMillis, millis, checksums, summary, sample, plan, breakdown);
    }

    /**
     * How this configuration's output compares with the CPU's, over every row rather than over one
     * aggregate of them.
     */
    private static String compare(ModeResult reference, ModeResult result) {
        for (int i = 1; i < result.checksums().length; i++) {
            if (!agrees(result.checksums()[0], result.checksums()[i])) {
                return "UNSTABLE_ACROSS_RUNS(run0=" + result.checksums()[0] + " run" + i + "=" + result.checksums()[i] + ")";
            }
        }
        if (reference == null || reference == result) {
            return reference == result ? "REFERENCE" : "NO_B0_REFERENCE";
        }
        if (!agrees(reference.checksums()[0], result.checksums()[0])) {
            return "DIFFERS_SUM(" + result.checksums()[0] + " vs " + reference.checksums()[0] + ")";
        }
        String[] names = { "count", "sum", "sumOfSquares", "min", "max" };
        for (int i = 0; i < names.length; i++) {
            if (!agrees(reference.summary()[i], result.summary()[i])) {
                return "DIFFERS_" + names[i] + "(" + result.summary()[i] + " vs " + reference.summary()[i] + ")";
            }
        }
        for (int i = 0; i < Math.min(reference.sample().length, result.sample().length); i++) {
            if (!agrees(reference.sample()[i], result.sample()[i])) {
                return "DIFFERS_ROW_" + i + "(" + result.sample()[i] + " vs " + reference.sample()[i] + ")";
            }
        }
        return "MATCHES_B0_ALL_ROWS";
    }

    /** count, sum, sum of squares, min and max of the UDF output over every row. Untimed. */
    private static double[] summarise(SparkSession spark, int size, String expression) {
        Row row = data(spark, size).selectExpr("(" + expression + ") as v") //
                .selectExpr("count(v)", "sum(v)", "sum(v * v)", "min(v)", "max(v)").head();
        double[] values = new double[5];
        for (int i = 0; i < 5; i++) {
            values[i] = row.isNullAt(i) ? Double.NaN : ((Number) row.get(i)).doubleValue();
        }
        return values;
    }

    /** The first rows' outputs in a fixed order, so that two configurations can be compared row by row. */
    private static double[] sample(SparkSession spark, int size, String expression) {
        List<Row> collected = spark.range(0, Math.min(size, ELEMENTWISE_ROWS), 1, 2) //
                .selectExpr("id", "cast(id as double) * 0.5 + 1.0 as a", "cast(id as double) * 0.25 + 2.0 as b", "cast(id as double) * 0.125 + 3.0 as c") //
                .selectExpr("id", "a", "b", "c", "if(a < 500000.0, array(a, b, c), array(a, b, c, a + b)) as vec_a", "array(c, b, a) as vec_b") //
                .selectExpr("id", "(" + expression + ") as v").orderBy("id").collectAsList();
        double[] values = new double[collected.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = collected.get(i).isNullAt(1) ? Double.NaN : collected.get(i).getDouble(1);
        }
        return values;
    }

    private static boolean agrees(double expected, double actual) {
        if (Double.isNaN(expected) && Double.isNaN(actual)) {
            return true;
        }
        double scale = Math.max(1.0, Math.abs(expected));
        return Math.abs(expected - actual) / scale < 1e-9;
    }

    private static void configure(SparkSession spark, String mode) {
        switch (mode) {
            case "B0-cpu-no-rapids" -> {
                spark.conf().set("spark.rapids.sql.enabled", "false");
                spark.conf().set("spark.rapids.sql.tornadovm.enabled", "false");
            }
            case "B1-rapids-cpu-bridge" -> {
                spark.conf().set("spark.rapids.sql.enabled", "true");
                spark.conf().set("spark.rapids.sql.tornadovm.enabled", "false");
            }
            default -> {
                spark.conf().set("spark.rapids.sql.enabled", "true");
                spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
            }
        }
    }

    /** One full query: every row is read, the UDF runs on every row, and the sum consumes them all. */
    private static double runOnce(SparkSession spark, int size, String expression) {
        Row row = data(spark, size).selectExpr("sum(" + expression + ") as total").head();
        return row.isNullAt(0) ? Double.NaN : row.getDouble(0);
    }

    private static String planSummary(SparkSession spark, int size, String expression) {
        Dataset<Row> query = data(spark, size).selectExpr("sum(" + expression + ") as total");
        // Adaptive execution rewrites the plan as the query runs, so the executed plan of a query
        // that has not run yet is the plan before any GPU rule touched it. It has to be this very
        // Dataset that runs, too: head() builds a limit query of its own and leaves this one
        // unexecuted, which is exactly how a plan that did run on the GPU still reads as CPU-only.
        query.collect();
        String plan = query.queryExecution().executedPlan().toString();
        List<String> markers = new ArrayList<>();
        if (plan.contains("GpuProject")) {
            markers.add("GpuProject");
        }
        if (plan.contains("GpuCpuBridge")) {
            markers.add("GpuCpuBridge");
        }
        if (plan.contains("gpu_tornado")) {
            markers.add("GpuScalaUDF");
        }
        if (plan.contains("GpuHashAggregate")) {
            markers.add("GpuHashAggregate");
        }
        if (plan.contains("Project ") && !plan.contains("GpuProject")) {
            markers.add("CpuProject");
        }
        return markers.isEmpty() ? "cpu-only" : String.join("+", markers);
    }

    private static void collectBreakdown(List<String[]> breakdown, String workload, int size, int batchSizeBytes) {
        try {
            Class<?> metrics = Class.forName("ai.beestream.jitudf.TornadoUdfMetrics");
            @SuppressWarnings("unchecked")
            List<Object> warm = (List<Object>) metrics.getMethod("warmSamples").invoke(null);
            @SuppressWarnings("unchecked")
            List<Object> cold = (List<Object>) metrics.getMethod("coldSamples").invoke(null);
            addStages(breakdown, workload, size, batchSizeBytes, "B3-warm", warm);
            addStages(breakdown, workload, size, batchSizeBytes, "B3-first-batch-per-plan", cold);
        } catch (ReflectiveOperationException e) {
            breakdown.add(new String[] { workload, String.valueOf(size), String.valueOf(batchSizeBytes), "B3", "all", "NOT_MEASURED", "NOT_MEASURED", "NOT_MEASURED", "0", e.toString() });
        }
    }

    private static void addStages(List<String[]> breakdown, String workload, int size, int batchSizeBytes, String mode, List<Object> samples) {
        if (samples.isEmpty()) {
            return;
        }
        String[] stages = { "columnAccessNanos", "outputAllocNanos", "planLookupNanos", "bindNanos", "inputSyncNanos", "executeNanos", "outputSyncNanos", "wrapNanos" };
        // Bytes rather than microseconds, kept in the same table because they are per-batch facts
        // about the same batches: what was bound, and what -- if anything -- was copied.
        for (String counter : new String[] { "inputBytesBound", "outputBytesBound", "copyInBytes", "copyOutBytes", "kernelNanos" }) {
            double[] values = new double[samples.size()];
            for (int i = 0; i < samples.size(); i++) {
                try {
                    values[i] = (long) samples.get(i).getClass().getMethod(counter).invoke(samples.get(i));
                } catch (ReflectiveOperationException e) {
                    values[i] = Double.NaN;
                }
            }
            Arrays.sort(values);
            breakdown.add(new String[] { workload, String.valueOf(size), String.valueOf(batchSizeBytes), mode, counter, f(median(values)), f(values[0]), f(values[values.length - 1]),
                    String.valueOf(values.length), counter.endsWith("Bytes") || counter.endsWith("Bound") ? "bytes; copyIn/copyOut from TornadoVM's profiler, -1 when not profiling"
                            : "nanoseconds of device kernel time from TornadoVM's profiler, -1 when not profiling" });
        }
        // execute minus kernel: what a batch spends inside the execution plan that is not device
        // kernel time -- queue submission, the plan's own bookkeeping, and the wait. It is derived
        // here rather than left to whoever reads the table, because leaving it to them is how it
        // came to be left out of the overhead total in the first place.
        double[] gap = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            try {
                long execute = (long) samples.get(i).getClass().getMethod("executeNanos").invoke(samples.get(i));
                long kernel = (long) samples.get(i).getClass().getMethod("kernelNanos").invoke(samples.get(i));
                gap[i] = kernel < 0 ? Double.NaN : (execute - kernel) / 1000.0;
            } catch (ReflectiveOperationException e) {
                gap[i] = Double.NaN;
            }
        }
        Arrays.sort(gap);
        breakdown.add(new String[] { workload, String.valueOf(size), String.valueOf(batchSizeBytes), mode, "executeMinusKernel", f(median(gap)), f(gap[0]), f(gap[gap.length - 1]),
                String.valueOf(gap.length), "executeNanos - kernelNanos in microseconds; NaN when the profiler is off. Part of non-kernel overhead." });

        for (String stage : stages) {
            double[] micros = new double[samples.size()];
            for (int i = 0; i < samples.size(); i++) {
                try {
                    long nanos = (long) samples.get(i).getClass().getMethod(stage).invoke(samples.get(i));
                    micros[i] = nanos / 1000.0;
                } catch (ReflectiveOperationException e) {
                    micros[i] = Double.NaN;
                }
            }
            Arrays.sort(micros);
            breakdown.add(new String[] { workload, String.valueOf(size), String.valueOf(batchSizeBytes), mode, stage.replace("Nanos", ""), f(median(micros)), f(micros[0]),
                    f(micros[micros.length - 1]), String.valueOf(micros.length), "System.nanoTime around the phase; GPU work bracketed by a device synchronize" });
        }
    }

    private static void resetMetrics() {
        try {
            Class.forName("ai.beestream.jitudf.TornadoUdfMetrics").getMethod("reset").invoke(null);
        } catch (ReflectiveOperationException e) {
            // not on the classpath in the CPU-only configurations
        }
    }

    private static double median(double[] sorted) {
        int n = sorted.length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double percentile(double[] sorted, int p) {
        int index = Math.min(sorted.length - 1, Math.max(0, (int) Math.round((p / 100.0) * (sorted.length - 1))));
        return sorted[index];
    }

    private static String f(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    private static void write(Path path, String header, List<String[]> rows) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING))) {
            out.println(header);
            for (String[] row : rows) {
                out.println(String.join(",", Arrays.stream(row).map(Benchmark::escape).toList()));
            }
        }
    }

    private static String escape(String value) {
        return value.contains(",") ? "\"" + value + "\"" : value;
    }
}
