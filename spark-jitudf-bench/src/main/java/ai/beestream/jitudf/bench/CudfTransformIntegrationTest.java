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

import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * The NVIDIA-aligned path, end to end on a GPU: driver-side TC compilation into cuDF's own
 * {@code transform}, with no TornadoVM runtime anywhere.
 *
 * <p>
 * This is the test that distinguishes the two architectures. {@link IntegrationTest} proves the
 * TornadoVM runtime path; this proves the one NVIDIA asked for. The UDF is an ordinary Java
 * {@code UDF1} containing a data-dependent loop -- the shape the RAPIDS UDF compiler must decline,
 * because a Catalyst expression tree has no back edge -- registered the way any user would register
 * it, and not rewritten.
 * </p>
 *
 * <p>
 * Three things have to hold at once for this to mean anything, so all three are checked: the answer
 * matches plain CPU Spark exactly, the physical plan shows the GPU path rather than a fallback, and
 * the object RAPIDS was given is a {@code CudfTransformUdf} carrying CUDA source rather than a
 * TornadoVM kernel. The last matters because a passing result with the wrong plan is the usual way
 * this kind of test lies.
 * </p>
 */
public final class CudfTransformIntegrationTest {

    private static int checks = 0;
    private static int failures = 0;

    /**
     * Collatz steps: a data-dependent loop over a long, returning an int.
     *
     * <p>
     * The accumulator is a {@code long} because {@code 3x + 1} leaves the int range well inside the
     * inputs used here, so an int version would silently compute something else. The aggregate of
     * this function over 1..50,000,000 is 8,600,532,084, which is the oracle the benchmark uses.
     * </p>
     */
    public static final class CollatzUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long seed) {
            long x = seed;
            int steps = 0;
            while (x != 1) {
                x = (x & 1L) == 1L ? 3 * x + 1 : x >>> 1;
                steps++;
            }
            return steps;
        }
    }

    /** Set bits of a packed activity bitmap: the one near-fit in NVIDIA's customer survey. */
    public static final class ActiveDaysUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long bitmap) {
            return Long.bitCount(bitmap);
        }
    }

    /** Configured through its constructor: the common shape, reached by folding the final field. */
    public static final class BoundedCollatzUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        private final int rounds;

        public BoundedCollatzUdf(int rounds) {
            this.rounds = rounds;
        }

        @Override
        public Integer call(Long seed) {
            long x = seed;
            int steps = 0;
            for (int i = 0; i < rounds && x != 1; i++) {
                x = (x & 1L) == 1L ? 3 * x + 1 : x >>> 1;
                steps++;
            }
            return steps;
        }
    }

    /** The customer near-fit: per-day activity flags packed into an array<bigint>, counted per row. */
    public static final class ActivityBitmapUdf implements UDF1<scala.collection.Seq<Long>, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(scala.collection.Seq<Long> arr) {
            int acc = 0;
            for (int i = 0; i < arr.length(); i++) {
                acc += Long.bitCount((Long) arr.apply(i));
            }
            return acc;
        }
    }

    /** Integer division and remainder by constants: refused outright until the obsolete gate went. */
    public static final class DigitSumUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            long x = v;
            int sum = 0;
            while (x > 0) {
                sum += (int) (x % 10);
                x = x / 10;
            }
            return sum;
        }
    }

    /**
     * Divides by a value taken from the input, which no constant analysis can prove non-zero. It
     * compiles because cudf can be told the row failed: the device function returns a cudf error
     * code. Here the divisor is never zero, so the codes are all SUCCESS and the answer must equal
     * the CPU's.
     */
    public static final class VariableDivisorUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            long d = (v & 0xFFL) + 1;
            return (int) (v / d);
        }
    }

    /** Divides by a value that really is zero on some rows: the query must fail, not answer. */
    public static final class ZeroDivisorUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            long d = v % 1000;
            return (int) (v / d);
        }
    }

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;

        // Chosen before Spark starts: the provider reads it when it compiles.
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        System.out.println("backend = " + TornadoUdfProvider.backend());

        SparkSession spark = IntegrationTest.session(rows);
        try {
            spark.udf().register("cudf_collatz", new CollatzUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_active_days", new ActiveDaysUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_bounded", new BoundedCollatzUdf(24), DataTypes.IntegerType);
            spark.udf().register("cudf_bitmap_list", new ActivityBitmapUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_digit_sum", new DigitSumUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_var_div", new VariableDivisorUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_zero_div", new ZeroDivisorUdf(), DataTypes.IntegerType);

            runCase(spark, "C1-collatz-loop", "cudf_collatz", rows);
            runCase(spark, "C2-activity-bitmap", "cudf_active_days", rows);
            runCase(spark, "C3-constructor-configured", "cudf_bounded", rows);
            runListCase(spark, rows);
            // Integer division and remainder, by a constant and by a value from the row.
            runCase(spark, "C5-integer-division", "cudf_digit_sum", rows);
            runCase(spark, "C6-variable-divisor", "cudf_var_div", rows);
            runZeroDivisorCase(spark, rows);
            checkProviderProducedCuda();
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

    /**
     * Non-null, non-negative longs with enough spread to make the loop data-dependent.
     *
     * <p>
     * Built without {@code %}, which is nullable in Spark and would have the analysis refuse the
     * column before any of this was reached.
     * </p>
     */
    private static Dataset<Row> data(SparkSession spark, int rows) {
        return spark.range(1, rows + 1L, 1, 2).selectExpr("id", "id * 7 + 1 as seed");
    }

    private static void runCase(SparkSession spark, String label, String udf, int rows) {
        System.out.println();
        System.out.println("--- " + label + " ---");

        // The reference: the same query with this integration switched off entirely.
        List<Row> cpu = collect(spark, rows, udf, false);
        List<Row> gpu = collect(spark, rows, udf, true);

        check(label + ": row count", cpu.size() == gpu.size(), cpu.size() + " vs " + gpu.size());
        int mismatches = 0;
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            if (cpu.get(i).getInt(1) != gpu.get(i).getInt(1)) {
                if (mismatches++ < 3) {
                    System.out.println("  row " + i + ": cpu=" + cpu.get(i).getInt(1) + " gpu=" + gpu.get(i).getInt(1));
                }
            }
        }
        // Integer arithmetic, so this is bit-exactness and not a tolerance.
        check(label + ": every row identical to CPU Spark", mismatches == 0, mismatches + " rows differ");

        String plan = plan(spark, rows, udf, true);
        // RAPIDS names a GPU-evaluated UDF `gpu_<name>`, so its presence is the positive evidence
        // that the UDF itself ran on the device and not merely that the surrounding plan did.
        check(label + ": the UDF itself runs on the GPU", plan.contains("gpu_" + udf), firstLines(plan));
        // GpuCpuBridge is the fallback that evaluates the UDF on the CPU inside a GPU plan, and
        // GpuRowBasedScalaUDF is the row-by-row path. Either means the UDF did not go through cudf.
        // GpuColumnarToRow is not a fallback: it is how any GPU plan hands rows back to collect().
        check(label + ": no CPU fallback for the UDF", !plan.contains("GpuCpuBridge") && !plan.contains("GpuRowBasedScalaUDF"), firstLines(plan));
    }

    /**
     * A fold over an array&lt;bigint&gt; column: the pattern NVIDIA's survey called its only near-fit,
     * and the one this architecture was expected to be unable to run.
     */
    private static void runListCase(SparkSession spark, int rows) {
        System.out.println();
        System.out.println("--- C4-activity-bitmap-list ---");
        String expression = "cudf_bitmap_list(array(seed, seed * 3 + 1, seed * 7)) as result";

        spark.conf().set("spark.rapids.sql.enabled", "false");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "false");
        List<Row> cpu = data(spark, rows).selectExpr("id", expression).orderBy("id").collectAsList();
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
        Dataset<Row> df = data(spark, rows).selectExpr("id", expression);
        List<Row> gpu = df.orderBy("id").collectAsList();

        int mismatches = 0;
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            if (cpu.get(i).getInt(1) != gpu.get(i).getInt(1)) {
                if (mismatches++ < 3) {
                    System.out.println("  row " + i + ": cpu=" + cpu.get(i).getInt(1) + " gpu=" + gpu.get(i).getInt(1));
                }
            }
        }
        check("C4-activity-bitmap-list: every row identical to CPU Spark", mismatches == 0 && cpu.size() == gpu.size(), mismatches + " rows differ");
        String plan = plan(spark, rows, "cudf_bitmap_list", true);
        check("C4-activity-bitmap-list: no CPU fallback", !plan.contains("GpuCpuBridge") && !plan.contains("GpuRowBasedScalaUDF"), firstLines(plan));
    }

    /**
     * The other half of admitting a divisor that cannot be proved non-zero: when it really is zero,
     * the query has to fail. On the CPU Java raises ArithmeticException; on the GPU the device
     * function returns cudf::errc::DIVISION_BY_ZERO, the kernel reduces it into one code for the
     * launch, and cudf raises evaluation_error for the call. Different exceptions, but the same
     * outcome, and the outcome is the thing that must not change: a wrong number instead of a
     * failure would be the worst result this integration could produce.
     */
    private static void runZeroDivisorCase(SparkSession spark, int rows) {
        System.out.println();
        System.out.println("--- C7-divide-by-zero ---");
        boolean cpuFailed = failsWith(spark, rows, "cudf_zero_div", false);
        boolean gpuFailed = failsWith(spark, rows, "cudf_zero_div", true);
        check("C7-divide-by-zero: CPU Spark fails the query", cpuFailed, "the CPU did not fail, so there is nothing to match");
        check("C7-divide-by-zero: the GPU path fails it too, rather than answering", gpuFailed, "the GPU produced a result where the CPU raised");
    }

    /** Whether the query throws, which is what both paths are supposed to do here. */
    private static boolean failsWith(SparkSession spark, int rows, String udf, boolean enabled) {
        try {
            collect(spark, rows, udf, enabled);
            return false;
        } catch (Exception e) {
            Throwable root = e;
            while (root.getCause() != null && root.getCause() != root) {
                root = root.getCause();
            }
            System.out.println("  " + (enabled ? "gpu" : "cpu") + " raised " + root.getClass().getName() + ": "
                    + String.valueOf(root.getMessage()).lines().findFirst().orElse(""));
            return true;
        }
    }

    private static List<Row> collect(SparkSession spark, int rows, String udf, boolean enabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(enabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(enabled));
        return data(spark, rows).selectExpr("id", udf + "(seed) as result").orderBy("id").collectAsList();
    }

    private static String plan(SparkSession spark, int rows, String udf, boolean enabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(enabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(enabled));
        // The list UDF takes an array, so its call is spelled differently; passing `seed` to it
        // would be a type error rather than a plan to inspect.
        String call = udf.equals("cudf_bitmap_list") ? udf + "(array(seed, seed * 3 + 1, seed * 7))" : udf + "(seed)";
        Dataset<Row> df = data(spark, rows).selectExpr("id", call + " as result");
        df.collect();
        return df.queryExecution().executedPlan().toString();
    }

    /**
     * That the UDF really went through cuDF's transform, and not through the other backend.
     *
     * <p>
     * The provider records one decision per UDF. A decision tagged {@code cudf-transform} is the
     * only evidence that separates this run from {@link IntegrationTest}'s, since both produce
     * correct answers on a GPU plan.
     * </p>
     */
    private static void checkProviderProducedCuda() {
        System.out.println();
        System.out.println("--- backend ---");
        long viaCudf = TornadoUdfProvider.decisions().values().stream().filter(d -> d.contains("cudf-transform")).count();
        check("the provider compiled through cudf::transform", viaCudf >= 4, "decisions: " + TornadoUdfProvider.decisions());
    }

    private static void printDecisions() {
        System.out.println();
        System.out.println("--- planning decisions ---");
        TornadoUdfProvider.decisions().forEach((name, decision) -> System.out.println("  " + name + " -> " + decision));
    }

    private static String firstLines(String plan) {
        return plan.lines().limit(4).reduce("", (a, b) -> a + "\n    " + b);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }
}
