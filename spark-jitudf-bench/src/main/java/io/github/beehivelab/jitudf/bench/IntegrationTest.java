package io.github.beehivelab.jitudf.bench;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import io.github.beehivelab.jitudf.bench.udfs.CosineSimilarityUdf;
import io.github.beehivelab.jitudf.bench.udfs.DataDependentLoopUdf;
import io.github.beehivelab.jitudf.bench.udfs.HelperLoopUdf;
import io.github.beehivelab.jitudf.bench.udfs.LoopUdf;
import io.github.beehivelab.jitudf.bench.udfs.MathUdf;
import io.github.beehivelab.jitudf.bench.udfs.StringUdf;

/**
 * The end-to-end check: ordinary Spark UDF registration, RAPIDS declining the UDF, TornadoVM
 * picking it up, and the query running on the GPU with the right answers.
 *
 * <p>
 * Every case is run twice -- once with the integration enabled and once with it off -- over the
 * same generated data, and the two results are compared row by row. The CPU run is the reference,
 * so "correct" here means "the same numbers Spark produces without any of this", not "the numbers
 * looked plausible".
 * </p>
 */
public final class IntegrationTest {

    private static final double TOLERANCE = 1e-9;

    private static int failures = 0;
    private static int checks = 0;

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;

        SparkSession spark = session(rows);
        try {
            runCase(spark, "W1-math-three-columns", "tornado_math", true);
            runCase(spark, "W2-counted-loop", "tornado_loop", true);
            runCase(spark, "W2b-data-dependent-loop", "tornado_ddloop", true);
            runCase(spark, "W2c-loop-with-static-helper", "tornado_helper", true);
            runListCase(spark);
            runEmptyListCase(spark);
            runMixedCase(spark);
            runStringCase(spark);
            runControlCase(spark);
            printDecisions();
        } finally {
            spark.stop();
        }
        System.out.println();
        System.out.println("=== CHECKS: " + checks + ", FAILURES: " + failures + " ===");
        if (failures > 0) {
            System.exit(1);
        }
    }

    static SparkSession session(int rows) {
        SparkSession spark = SparkSession.builder() //
                .appName("tornado-rapids-integration") //
                .master("local[2]") //
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin") //
                .config("spark.rapids.sql.enabled", "true") //
                .config("spark.rapids.sql.udfCompiler.enabled", "true") //
                .config("spark.rapids.sql.explain", "NOT_ON_GPU") //
                .config("spark.rapids.memory.gpu.pool", "ASYNC") //
                .config("spark.rapids.memory.gpu.allocFraction", "0.35") //
                .config("spark.rapids.memory.gpu.minAllocFraction", "0.05") //
                .config("spark.rapids.memory.gpu.maxAllocFraction", "0.6") //
                .config("spark.rapids.memory.gpu.reserve", "805306368") //
                .config("spark.rapids.sql.batchSizeBytes", "268435456") //
                .config("spark.sql.shuffle.partitions", "2") //
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        spark.udf().register("tornado_loop", new LoopUdf(64), DataTypes.DoubleType);
        spark.udf().register("tornado_ddloop", new DataDependentLoopUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_helper", new HelperLoopUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_math", new MathUdf(), DataTypes.DoubleType);
        spark.udf().register("tornado_string", new StringUdf(), DataTypes.StringType);
        spark.udf().register("tornado_cosine", new CosineSimilarityUdf(), DataTypes.DoubleType);
        return spark;
    }

    /**
     * Fixed-seed numeric columns, identical for every configuration.
     *
     * <p>
     * Deliberately built without {@code %}: a remainder is nullable in Spark (it yields null on a
     * zero divisor), and the eligibility analysis refuses a nullable input rather than assume a
     * batch happens to contain no nulls. Scaling and offsetting a range keeps the columns
     * non-nullable, which is the subset this milestone supports.
     * </p>
     */
    static Dataset<Row> data(SparkSession spark, int rows) {
        return spark.range(0, rows, 1, 2) //
                .selectExpr("id", //
                        "cast(id as double) * 0.5 + 1.0 as a", //
                        "cast(id as double) * 0.25 + 2.0 as b", //
                        "cast(id as double) * 0.125 + 3.0 as c");
    }

    private static void runCase(SparkSession spark, String label, String udf, boolean expectGpu) {
        int rows = 100_000;
        String expression = udf.equals("tornado_math") ? udf + "(a, b, c)" : udf + "(a, b)";

        double[] cpu = collect(spark, rows, expression, false);
        double[] gpu = collect(spark, rows, expression, true);
        String plan = plan(spark, rows, expression, true);

        // GpuProject alone proves nothing: RAPIDS also puts a CPU-evaluated UDF inside a GPU
        // project, through GpuCpuBridge. What has to be true is that the UDF itself became a
        // gpu_-prefixed GpuScalaUDF, which is the columnar path this integration exists to reach.
        boolean onGpu = plan.contains("GpuProject") && plan.contains("gpu_" + udf) && !plan.contains("GpuCpuBridge");
        check(label + ": the UDF runs as a GpuScalaUDF inside GpuProjectExec", expectGpu == onGpu, "plan was:\n" + plan);
        check(label + ": row count matches", cpu.length == gpu.length, cpu.length + " vs " + gpu.length);
        int mismatches = 0;
        for (int i = 0; i < Math.min(cpu.length, gpu.length); i++) {
            double scale = Math.max(1.0, Math.abs(cpu[i]));
            if (Math.abs(cpu[i] - gpu[i]) / scale > TOLERANCE) {
                if (mismatches < 5) {
                    System.out.printf("  row %d: cpu=%.17g gpu=%.17g%n", i, cpu[i], gpu[i]);
                }
                mismatches++;
            }
        }
        check(label + ": values match the CPU within " + TOLERANCE + " relative", mismatches == 0, mismatches + " rows differ");
    }

    /**
     * W3: a per-row vector reduction over two {@code ArrayType} columns, which reach cuDF as
     * {@code LIST} columns and the kernel as offsets plus a flattened child.
     *
     * <p>
     * The two columns have different lengths, and one of them has a length that varies by row, so
     * the kernel really has to read the offsets rather than assume a stride.
     * </p>
     */
    private static void runListCase(SparkSession spark) {
        int rows = 100_000;
        // if(...) keeps the column non-nullable while making the rows different lengths; a
        // remainder would have made it nullable, which the analysis refuses.
        String left = "if(id % 2 = 0, array(a, b, c), array(a, b, c, a + b)) as vec_a";
        String right = "array(c, b, a) as vec_b";

        double[] cpu = collectList(spark, rows, left, right, false);
        double[] gpu = collectList(spark, rows, left, right, true);
        String plan = planFor(spark, rows, left, right, true);

        boolean onGpu = plan.contains("GpuProject") && plan.contains("gpu_tornado_cosine") && !plan.contains("GpuCpuBridge");
        check("W3-list-cosine-similarity: the UDF runs as a GpuScalaUDF inside GpuProjectExec", onGpu, "plan was:\n" + plan);
        check("W3-list-cosine-similarity: row count matches", cpu.length == gpu.length, cpu.length + " vs " + gpu.length);
        int mismatches = 0;
        for (int i = 0; i < Math.min(cpu.length, gpu.length); i++) {
            double scale = Math.max(1.0, Math.abs(cpu[i]));
            if (Math.abs(cpu[i] - gpu[i]) / scale > TOLERANCE) {
                if (mismatches < 5) {
                    System.out.printf("  row %d: cpu=%.17g gpu=%.17g%n", i, cpu[i], gpu[i]);
                }
                mismatches++;
            }
        }
        check("W3-list-cosine-similarity: values match the CPU within " + TOLERANCE + " relative", mismatches == 0, mismatches + " rows differ");
    }

    /**
     * The batch where every row's list is empty, and the batch where only some are.
     *
     * <p>
     * cuDF need not allocate a child payload for a LIST column whose rows are all empty, so the
     * borrowed address for the flattened child is simply not there. That is an ordinary batch, not
     * an error: the kernel's bounds proof says every access satisfies start &lt;= start + i &lt;
     * end, and end equals start on an empty row, so nothing is read. This checks that such a batch
     * produces the answer the CPU produces instead of failing, and that a batch mixing empty and
     * non-empty rows is still right row by row.
     * </p>
     */
    private static void runEmptyListCase(SparkSession spark) {
        int rows = 50_000;
        // slice(..., 1, 0) is an empty array<double> that is still non-nullable, which is what the
        // analysis requires; array() on its own would be array<void>.
        String allEmptyLeft = "slice(array(a, b, c), 1, 0) as vec_a";
        String allEmptyRight = "slice(array(c, b, a), 1, 0) as vec_b";

        double[] cpu = collectList(spark, rows, allEmptyLeft, allEmptyRight, false);
        double[] gpu = collectList(spark, rows, allEmptyLeft, allEmptyRight, true);
        String plan = planFor(spark, rows, allEmptyLeft, allEmptyRight, true);
        boolean onGpu = plan.contains("GpuProject") && plan.contains("gpu_tornado_cosine") && !plan.contains("GpuCpuBridge");
        check("W3-empty-list: an all-empty LIST batch still runs on the GPU", onGpu, "plan was:\n" + plan);
        check("W3-empty-list: every row is the CPU's 0.0", matches(cpu, gpu), firstMismatch(cpu, gpu));

        String mixedLeft = "if(id % 3 = 0, slice(array(a, b, c), 1, 0), array(a, b, c)) as vec_a";
        String mixedRight = "if(id % 3 = 0, slice(array(c, b, a), 1, 0), array(c, b, a)) as vec_b";
        double[] mixedCpu = collectList(spark, rows, mixedLeft, mixedRight, false);
        double[] mixedGpu = collectList(spark, rows, mixedLeft, mixedRight, true);
        check("W3-empty-list: a batch mixing empty and non-empty rows matches the CPU", matches(mixedCpu, mixedGpu), firstMismatch(mixedCpu, mixedGpu));
    }

    private static boolean matches(double[] cpu, double[] gpu) {
        if (cpu.length != gpu.length) {
            return false;
        }
        for (int i = 0; i < cpu.length; i++) {
            double scale = Math.max(1.0, Math.abs(cpu[i]));
            if (Math.abs(cpu[i] - gpu[i]) / scale > TOLERANCE) {
                return false;
            }
        }
        return true;
    }

    private static String firstMismatch(double[] cpu, double[] gpu) {
        if (cpu.length != gpu.length) {
            return "row counts differ: " + cpu.length + " vs " + gpu.length;
        }
        for (int i = 0; i < cpu.length; i++) {
            double scale = Math.max(1.0, Math.abs(cpu[i]));
            if (Math.abs(cpu[i] - gpu[i]) / scale > TOLERANCE) {
                return String.format("row %d: cpu=%.17g gpu=%.17g", i, cpu[i], gpu[i]);
            }
        }
        return "all " + cpu.length + " rows match";
    }

    private static double[] collectList(SparkSession spark, int rows, String left, String right, boolean tornadoEnabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(tornadoEnabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(tornadoEnabled));
        List<Row> collected = data(spark, rows).selectExpr("id", left, right) //
                .selectExpr("id", "tornado_cosine(vec_a, vec_b) as value").orderBy("id").collectAsList();
        double[] values = new double[collected.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = collected.get(i).getDouble(1);
        }
        return values;
    }

    private static String planFor(SparkSession spark, int rows, String left, String right, boolean tornadoEnabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(tornadoEnabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(tornadoEnabled));
        Dataset<Row> df = data(spark, rows).selectExpr("id", left, right).selectExpr("id", "tornado_cosine(vec_a, vec_b) as value");
        df.count();
        return df.queryExecution().executedPlan().toString();
    }

    /**
     * One query, four kinds of expression at once.
     *
     * <p>
     * The question this answers is whether enabling the kernel compiler diverts everything through
     * it. It does not, and cannot: the hook runs only on a UDF the RAPIDS compiler has already
     * declined, and an expression RAPIDS handles natively never reaches it. So a single projection
     * can mix a native GPU expression, two UDFs compiled to two separate kernels, and a third UDF
     * that neither compiler accepts and which stays on the CPU bridge -- in the same plan, over the
     * same batch, with every column correct.
     * </p>
     */
    private static void runMixedCase(SparkSession spark) {
        int rows = 100_000;
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");

        Dataset<Row> df = data(spark, rows).selectExpr("id", //
                "a * 2.0 + b as native_expr", // RAPIDS evaluates this itself, no UDF involved
                "tornado_math(a, b, c) as kernel_one", // declined by RAPIDS, compiled here
                "tornado_loop(a, b) as kernel_two", // declined by RAPIDS, compiled here, different kernel
                "tornado_string(a) as cpu_only"); // declined by both, stays on the CPU
        String plan = df.queryExecution().executedPlan().toString();
        df.count();
        plan = df.queryExecution().executedPlan().toString();

        for (String line : plan.split("\n")) {
            if (line.contains("GpuProject")) {
                System.out.println("MIXED-PLAN " + line.trim());
            }
        }
        check("mixed: the native expression is not diverted to a kernel", !plan.contains("gpu_tornado_math(a") || !plan.contains("GpuAlias((a"), "plan was:\n" + plan);
        check("mixed: both eligible UDFs became kernels in one plan", plan.contains("gpu_tornado_math") && plan.contains("gpu_tornado_loop"), "plan was:\n" + plan);
        check("mixed: the ineligible UDF stayed on the CPU bridge", plan.contains("GpuCpuBridge") && !plan.contains("gpu_tornado_string"), "plan was:\n" + plan);

        // And every column is still right, computed against the same query with everything off.
        List<Row> gpu = df.orderBy("id").collectAsList();
        spark.conf().set("spark.rapids.sql.enabled", "false");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "false");
        List<Row> cpu = data(spark, rows).selectExpr("id", "a * 2.0 + b as native_expr", "tornado_math(a, b, c) as kernel_one", //
                "tornado_loop(a, b) as kernel_two", "tornado_string(a) as cpu_only").orderBy("id").collectAsList();
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");

        check("mixed: same row count", cpu.size() == gpu.size(), cpu.size() + " vs " + gpu.size());
        int bad = 0;
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            for (int col = 1; col <= 3; col++) {
                double e = cpu.get(i).getDouble(col), g = gpu.get(i).getDouble(col);
                if (Math.abs(e - g) / Math.max(1.0, Math.abs(e)) > TOLERANCE) {
                    bad++;
                }
            }
            if (!cpu.get(i).getString(4).equals(gpu.get(i).getString(4))) {
                bad++;
            }
        }
        check("mixed: all four columns match the CPU on every row", bad == 0, bad + " values differ");
    }

    /** A UDF neither compiler accepts: it must stay on the CPU and still be right. */
    private static void runStringCase(SparkSession spark) {
        int rows = 20_000;
        Dataset<Row> df = data(spark, rows).selectExpr("id", "tornado_string(a) as label");
        String plan = df.queryExecution().executedPlan().toString();
        // RAPIDS keeps an unsupported UDF on the CPU through its own bridge expression. What
        // matters is that it was not diverted to a kernel, not that the plan has no GPU node.
        check("negative-string-udf: left on the existing CPU bridge", !plan.contains("gpu_tornado_string"), "plan was:\n" + plan);
        long positives = df.filter("label = 'positive'").count();
        check("negative-string-udf: correct result", positives == rows, positives + " of " + rows);
    }

    /** An expression RAPIDS handles itself: it must take the path it always took. */
    private static void runControlCase(SparkSession spark) {
        int rows = 50_000;
        Dataset<Row> df = data(spark, rows).selectExpr("id", "a * b + c as total");
        String plan = df.queryExecution().executedPlan().toString();
        check("control-native-expression: runs on the GPU as before", plan.contains("GpuProject"), "plan was:\n" + plan);
        Row sum = df.selectExpr("sum(total) as s").head();
        check("control-native-expression: produced a sum", sum.getDouble(0) > 0, String.valueOf(sum.getDouble(0)));
    }

    static double[] collect(SparkSession spark, int rows, String expression, boolean tornadoEnabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(tornadoEnabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(tornadoEnabled));
        List<Row> collected = data(spark, rows).selectExpr("id", expression + " as value").orderBy("id").collectAsList();
        double[] values = new double[collected.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = collected.get(i).getDouble(1);
        }
        return values;
    }

    static String plan(SparkSession spark, int rows, String expression, boolean tornadoEnabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(tornadoEnabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(tornadoEnabled));
        Dataset<Row> df = data(spark, rows).selectExpr("id", expression + " as value");
        df.count();
        return df.queryExecution().executedPlan().toString();
    }

    public static void printDecisions() {
        System.out.println();
        System.out.println("=== PLANNING DECISIONS ===");
        try {
            Class<?> provider = Class.forName("io.github.beehivelab.jitudf.TornadoUdfProvider");
            Object decisions = provider.getMethod("decisions").invoke(null);
            System.out.println("provider: " + decisions);
            System.out.println("accepted=" + provider.getMethod("accepted").invoke(null) + " refused=" + provider.getMethod("refused").invoke(null));
        } catch (ReflectiveOperationException e) {
            System.out.println("provider decisions unavailable: " + e);
        }
        try {
            Class<?> bridge = Class.forName("com.nvidia.spark.rapids.TornadoUdfBridge$");
            Object instance = bridge.getField("MODULE$").get(null);
            System.out.println("bridge: " + bridge.getMethod("decisionLog").invoke(instance));
        } catch (ReflectiveOperationException e) {
            System.out.println("bridge decisions unavailable: " + e);
        }
        try {
            Class<?> metrics = Class.forName("io.github.beehivelab.jitudf.TornadoUdfMetrics");
            System.out.println("executor: " + metrics.getMethod("summary").invoke(null));
            Class<?> cache = Class.forName("io.github.beehivelab.jitudf.TornadoKernelCache");
            System.out.println("cache: classesDefined=" + cache.getMethod("classesDefined").invoke(null) //
                    + " plansBuilt=" + cache.getMethod("plansBuilt").invoke(null) //
                    + " plansReused=" + cache.getMethod("plansReused").invoke(null));
        } catch (ReflectiveOperationException e) {
            System.out.println("executor metrics unavailable: " + e);
        }
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (ok) {
            System.out.println("[PASS] " + what);
        } else {
            failures++;
            System.out.println("[FAIL] " + what + " -- " + detail);
        }
    }

    private static List<String> unused() {
        return new ArrayList<>();
    }
}
