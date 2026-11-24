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
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.api.java.UDF5;
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

    // -----------------------------------------------------------------------------------------
    // Several inputs, typed outputs, and nullable columns: the three the widened binding adds.
    // -----------------------------------------------------------------------------------------

    /** Two INT32 columns and an INT32 result, with a loop so this is not a Catalyst expression. */
    public static final class TwoIntUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer a, Integer b) {
            int acc = a;
            for (int i = 0; i < 8; i++) {
                acc = acc * 31 + b + i;
            }
            return acc;
        }
    }

    /** INT32 and FLOAT32 in, FLOAT64 out: three different widths in one call. */
    public static final class MixedWidthUdf implements UDF2<Integer, Float, Double> {
        private static final long serialVersionUID = 1L;

        @Override
        public Double call(Integer a, Float b) {
            // sqrt is correctly rounded on both sides, and the sum of exact values is exact, so
            // the comparison against CPU Spark is equality rather than a tolerance.
            return Math.sqrt(a.doubleValue() * a.doubleValue() + b.doubleValue() * b.doubleValue());
        }
    }

    /** Three fixed-width inputs of three types, and an INT64 result. */
    public static final class ThreeInputUdf implements UDF3<Long, Integer, Double, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Integer b, Double c) {
            long acc = a;
            for (int i = 0; i < 6; i++) {
                acc = acc * 31 + b + c.longValue();
            }
            return acc;
        }
    }

    /** One INT64 in, one INT64 out: a result that does not fit the old INT32-only binding. */
    public static final class LongMixUdf implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long v) {
            long z = v;
            for (int i = 0; i < 4; i++) {
                z ^= z >>> 30;
                z *= 0xBF58476D1CE4E5B9L;
            }
            return z ^ (z >>> 27);
        }
    }

    /** Answers for a null instead of propagating it: a mask copy would get every null row wrong. */
    public static final class DefaultingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long value) {
            if (value == null) {
                return -1;
            }
            return value.intValue();
        }
    }

    /** Returns null for a null input, and also for a non-null input it rejects. */
    public static final class NullableResultUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long value) {
            if (value == null) {
                return null;
            }
            if ((value & 1L) == 1L) {
                return null;
            }
            return (int) (value / 2L);
        }
    }

    /** Two nullable columns, either of which may be absent on any given row. */
    public static final class CoalescingUdf implements UDF2<Long, Integer, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Integer b) {
            if (a == null && b == null) {
                return null;
            }
            long av = a == null ? 0L : a;
            long bv = b == null ? 0L : b.longValue();
            return av * 3L + bv;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Arities above three, and booleans.
    //
    // Nothing in the compiler, the generator or the binding ever counted to three: every one of
    // them iterates the input list. These cases exist because "nothing counts to three" is a claim
    // about code that was only ever exercised with one, two and three columns, and the way to
    // settle it is to run four and five.
    //
    // The boolean cases are different in kind: BOOL8 is a type the integration genuinely did not
    // carry, and admitting it is why PrimitiveKind has a fifth constant.
    // -------------------------------------------------------------------------------------------

    /** Four inputs, three different widths among them. */
    public static final class FourInputUdf implements UDF4<Long, Integer, Integer, Double, Double> {
        private static final long serialVersionUID = 1L;

        @Override
        public Double call(Long a, Integer b, Integer c, Double d) {
            double acc = d;
            for (int i = 0; i < 8; i++) {
                acc = acc * 1.5 + (a % (b + 1)) - c;
            }
            return acc;
        }
    }

    /** Five inputs. Integer arithmetic only, so the comparison with CPU Spark is exact. */
    public static final class FiveInputUdf implements UDF5<Integer, Integer, Long, Integer, Integer, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Integer a, Integer b, Long c, Integer d, Integer e) {
            long acc = c;
            acc = acc * 31L + a;
            acc = acc * 31L + b;
            acc = acc * 31L + d;
            acc = acc * 31L + e;
            return acc;
        }
    }

    /** A boolean result: the output type the binding could not ask for before. */
    public static final class IsLuckyUdf implements UDF1<Long, Boolean> {
        private static final long serialVersionUID = 1L;

        @Override
        public Boolean call(Long v) {
            long x = v;
            int digits = 0;
            while (x > 0) {
                digits += (int) (x % 10L);
                x /= 10L;
            }
            return (digits % 7) == 0;
        }
    }

    /** A boolean input as well as a boolean result. */
    public static final class BothBooleanUdf implements UDF2<Boolean, Long, Boolean> {
        private static final long serialVersionUID = 1L;

        @Override
        public Boolean call(Boolean flag, Long v) {
            return flag ? (v % 3L) == 0L : (v % 5L) == 0L;
        }
    }

    /** A nullable boolean result, which is a BOOL8 column with a validity mask of its own. */
    public static final class NullableBooleanUdf implements UDF1<Long, Boolean> {
        private static final long serialVersionUID = 1L;

        @Override
        public Boolean call(Long v) {
            if (v == null) {
                return null;
            }
            return (v % 4L) == 1L;
        }
    }

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;
        configuredRows = rows;

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
            spark.udf().register("cudf_two_int", new TwoIntUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_mixed_width", new MixedWidthUdf(), DataTypes.DoubleType);
            spark.udf().register("cudf_three_input", new ThreeInputUdf(), DataTypes.LongType);
            spark.udf().register("cudf_long_mix", new LongMixUdf(), DataTypes.LongType);
            spark.udf().register("cudf_defaulting", new DefaultingUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_nullable_result", new NullableResultUdf(), DataTypes.IntegerType);
            spark.udf().register("cudf_coalescing", new CoalescingUdf(), DataTypes.LongType);
            spark.udf().register("cudf_four_input", new FourInputUdf(), DataTypes.DoubleType);
            spark.udf().register("cudf_five_input", new FiveInputUdf(), DataTypes.LongType);
            spark.udf().register("cudf_is_lucky", new IsLuckyUdf(), DataTypes.BooleanType);
            spark.udf().register("cudf_both_boolean", new BothBooleanUdf(), DataTypes.BooleanType);
            spark.udf().register("cudf_nullable_boolean", new NullableBooleanUdf(), DataTypes.BooleanType);

            runCase(spark, "C1-collatz-loop", "cudf_collatz", rows);
            runCase(spark, "C2-activity-bitmap", "cudf_active_days", rows);
            runCase(spark, "C3-constructor-configured", "cudf_bounded", rows);
            runListCase(spark, rows);
            // Integer division and remainder, by a constant and by a value from the row.
            runCase(spark, "C5-integer-division", "cudf_digit_sum", rows);
            runCase(spark, "C6-variable-divisor", "cudf_var_div", rows);
            runZeroDivisorCase(spark, rows);

            // Several input columns.
            runValueCase(spark, "C8-two-int-inputs", "cudf_two_int", "cudf_two_int(a, b)", false);
            runValueCase(spark, "C9-mixed-widths-to-float64", "cudf_mixed_width", "cudf_mixed_width(a, f)", false, true);
            runValueCase(spark, "C10-three-inputs-to-int64", "cudf_three_input", "cudf_three_input(c, a, d)", false, true);
            // An output type the old binding could not ask for.
            runValueCase(spark, "C11-int64-output", "cudf_long_mix", "cudf_long_mix(c)", false);
            // Nullable columns, through TC's null-aware mode.
            runValueCase(spark, "C12-nullable-in-non-null-out", "cudf_defaulting", "cudf_defaulting(nseed)", true);
            runValueCase(spark, "C13-nullable-in-nullable-out", "cudf_nullable_result", "cudf_nullable_result(nseed)", true);
            runValueCase(spark, "C14-two-nullable-inputs", "cudf_coalescing", "cudf_coalescing(nseed, nb)", true);

            // Arities the binding has never been run at.
            runValueCase(spark, "C15-four-inputs", "cudf_four_input", "cudf_four_input(c, a, b, d)", false, true);
            runValueCase(spark, "C16-five-inputs", "cudf_five_input", "cudf_five_input(a, b, c, a, b)", false);
            // A boolean result column, and a boolean input column.
            runValueCase(spark, "C17-boolean-output", "cudf_is_lucky", "cudf_is_lucky(c)", false);
            runValueCase(spark, "C18-boolean-input-and-output", "cudf_both_boolean", "cudf_both_boolean(t, c)", false);
            runValueCase(spark, "C19-nullable-boolean-output", "cudf_nullable_boolean", "cudf_nullable_boolean(nseed)", true);

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

    /**
     * Columns for the multi-input cases, all of them non-nullable.
     *
     * <p>
     * Built from {@code &}, {@code *} and {@code +} only. Spark's {@code %} is nullable, because
     * it yields null rather than raising for a zero divisor, and one nullable column would send
     * these cases down the null-aware path instead of the one they are meant to exercise.
     * </p>
     */
    private static Dataset<Row> wideData(SparkSession spark, int rows) {
        return spark.range(1, rows + 1L, 1, 2).selectExpr(
                "id",
                "cast(id & 1023 as int) as a",
                "cast((id >> 3) & 255 as int) as b",
                "id * 7 + 1 as c",
                "cast(id & 15 as float) as f",
                "cast(id & 31 as double) + 0.5 as d",
                // A boolean input column. cast(0/1 as boolean) rather than a comparison, so the
                // column is a plain non-nullable BOOL8 rather than the result of an expression
                // RAPIDS might choose to evaluate on the CPU.
                "cast(id & 1 as boolean) as t");
    }

    /**
     * The same rows with nulls in them: one value in eight absent from {@code nseed}, one in four
     * from {@code nb}, so every combination of present and absent occurs.
     */
    private static Dataset<Row> nullableData(SparkSession spark, int rows) {
        return spark.range(1, rows + 1L, 1, 2).selectExpr(
                "id",
                "case when (id & 7) = 0 then null else id * 7 + 1 end as nseed",
                "case when (id & 3) = 0 then null else cast(id & 1023 as int) end as nb");
    }

    /**
     * One workload, compared value by value against CPU Spark and checked for the GPU path.
     *
     * <p>
     * The comparison is {@link java.util.Objects#equals} on the boxed values rather than a typed
     * getter, so a null on one side and a number on the other is a mismatch rather than an
     * exception -- which is the thing these cases exist to detect. For the floating-point cases the
     * values compared are exact: the device function is compiled strict (no contracted
     * multiply-adds) and {@code sqrt} is correctly rounded on both sides.
     * </p>
     */
    private static void runValueCase(SparkSession spark, String label, String udf, String call, boolean nullable) {
        runValueCase(spark, label, udf, call, nullable, false);
    }

    /**
     * @param floatingPoint
     *        whether the UDF touches float or double, in which case RAPIDS's own opt-in for
     *        operations that may not match Spark's CPU bit for bit has to be set. The plugin
     *        refuses such a UDF without it, deliberately: TornadoVM's CUDA backend is allowed to
     *        contract a multiply and an add into an fma, and Java is not, so a result can differ in
     *        the last ulp. The comparison below is still exact equality -- a device function is
     *        compiled strict, so no contraction happens on this path -- but the gate is about what
     *        may happen, not about what did, and turning it off here would be deciding that for
     *        every user.
     */
    private static void runValueCase(SparkSession spark, String label, String udf, String call, boolean nullable, boolean floatingPoint) {
        System.out.println();
        System.out.println("--- " + label + " ---");
        String previous = spark.conf().get("spark.rapids.sql.incompatibleOps.enabled", "false");
        if (floatingPoint) {
            spark.conf().set("spark.rapids.sql.incompatibleOps.enabled", "true");
        }
        try {
            runValueCaseBody(spark, label, udf, call, nullable);
        } finally {
            spark.conf().set("spark.rapids.sql.incompatibleOps.enabled", previous);
        }
    }

    private static void runValueCaseBody(SparkSession spark, String label, String udf, String call, boolean nullable) {

        List<Row> cpu = collectCall(spark, rowsOf(spark, nullable), call, false);
        List<Row> gpu = collectCall(spark, rowsOf(spark, nullable), call, true);

        check(label + ": row count", cpu.size() == gpu.size(), cpu.size() + " vs " + gpu.size());
        int mismatches = 0;
        int nulls = 0;
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            Object expected = cpu.get(i).get(1);
            Object actual = gpu.get(i).get(1);
            if (expected == null) {
                nulls++;
            }
            if (!java.util.Objects.equals(expected, actual)) {
                if (mismatches++ < 3) {
                    System.out.println("  row " + i + ": cpu=" + expected + " gpu=" + actual);
                }
            }
        }
        check(label + ": every row identical to CPU Spark", mismatches == 0, mismatches + " rows differ");
        if (nullable) {
            // Without this the "identical" check above could pass on a column that happens to have
            // no nulls at all, which would say nothing about null handling.
            check(label + ": CPU Spark really produced nulls to match", nulls > 0 || label.contains("non-null-out"),
                    "no null rows in the reference, so the comparison proves nothing about nulls");
        }

        String plan = planFor(spark, rowsOf(spark, nullable), call);
        check(label + ": the UDF itself runs on the GPU", plan.contains("gpu_" + udf), firstLines(plan));
        check(label + ": no CPU fallback for the UDF", !plan.contains("GpuCpuBridge") && !plan.contains("GpuRowBasedScalaUDF"), firstLines(plan));
    }

    /** The row count the session was built with, so the two datasets stay the same size. */
    private static int configuredRows = 0;

    private static Dataset<Row> rowsOf(SparkSession spark, boolean nullable) {
        return nullable ? nullableData(spark, configuredRows) : wideData(spark, configuredRows);
    }

    private static List<Row> collectCall(SparkSession spark, Dataset<Row> source, String call, boolean enabled) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(enabled));
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(enabled));
        return source.selectExpr("id", call + " as result").orderBy("id").collectAsList();
    }

    private static String planFor(SparkSession spark, Dataset<Row> source, String call) {
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
        spark.conf().set("spark.rapids.sql.enabled", "true");
        Dataset<Row> df = source.selectExpr("id", call + " as result");
        df.collect();
        return df.queryExecution().executedPlan().toString();
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
        check("the provider compiled through cudf::transform", viaCudf >= 13, "decisions: " + TornadoUdfProvider.decisions());
        long viaNullAware = TornadoUdfProvider.decisions().values().stream().filter(d -> d.contains("null-aware")).count();
        // The three nullable cases must have gone through the null-aware compilation, not through
        // some other route that happens to produce the right numbers.
        check("the nullable columns were compiled null-aware", viaNullAware >= 3, "decisions: " + TornadoUdfProvider.decisions());
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
