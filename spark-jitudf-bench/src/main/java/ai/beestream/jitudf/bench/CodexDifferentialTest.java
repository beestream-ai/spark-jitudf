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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.expressions.SparkUserDefinedFunction;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * Differential Spark queries for the independent review's counterexamples, on a real GPU.
 *
 * <p>
 * Every case runs the same query twice in one session: once with RAPIDS and this integration
 * switched off, which is CPU Spark executing the registered Java/Scala UDF exactly as an executor
 * would (after serialization), and once with both on. The two must agree on every value and every
 * null. And each case states the execution path it must take: an accepted UDF must appear as
 * {@code gpu_<name>} with no {@code GpuCpuBridge}/{@code GpuRowBasedScalaUDF}; a UDF that has to
 * fall back must <em>not</em> appear as a GPU UDF. Output equality alone is never taken as proof of
 * either path, and an unexpected path fails the run.
 * </p>
 *
 * <p>
 * Where Java raises, both runs must fail, and for the expected reason: the CPU with the Java
 * exception class, the GPU with the error cuDF reports for it. A linkage error, an NVRTC failure or
 * a missing library is a different failure and fails the check.
 * </p>
 */
public final class CodexDifferentialTest {

    private static int checks;
    private static int failures;

    private CodexDifferentialTest() {
    }

    // ------------------------------------------------------------------------------ the UDFs

    public static final class BitCountUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    /** Java's ActiveDays over Spark's Seq, the canonical fold. */
    public static final class ActiveDaysListUdf implements UDF1<scala.collection.Seq<Long>, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(scala.collection.Seq<Long> arr) {
            int acc = 0;
            for (int i = 0; i < arr.length(); i++) {
                acc += Long.bitCount(arr.apply(i));
            }
            return acc;
        }
    }

    /** A wrapping long sum: overflow and negatives. */
    public static final class LongSumListUdf implements UDF1<scala.collection.Seq<Long>, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(scala.collection.Seq<Long> arr) {
            long acc = 0;
            for (int i = 0; i < arr.length(); i++) {
                acc += arr.apply(i) * 3 - 1;
            }
            return acc;
        }
    }

    /** Codex 2: the same fold with i += 2. */
    public static final class StrideTwoListUdf implements UDF1<scala.collection.Seq<Long>, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(scala.collection.Seq<Long> arr) {
            int acc = 0;
            for (int i = 0; i < arr.length(); i += 2) {
                acc += Long.bitCount(arr.apply(i));
            }
            return acc;
        }
    }

    /** Codex 4: an Integer cast to Boolean. */
    public static final class BadCastUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return ((Boolean) (Object) x).booleanValue() ? 21 : 42;
        }
    }

    /** Codex 5: a final transient field, 9 on the driver and 0 on every executor. */
    public static final class TransientFieldUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final transient int factor;

        public TransientFieldUdf() {
            this.factor = 9;
        }

        @Override
        public Integer call(Integer x) {
            return factor + x;
        }
    }

    /** Codex 6. */
    public static final class FloorDivUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer a, Integer b) {
            return Math.floorDiv(a, b);
        }
    }

    public static final class FloorModLongUdf implements UDF2<Long, Long, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Long b) {
            return Math.floorMod(a, b);
        }
    }

    /** Codex 7: floating point inside an integer signature. */
    public static final class SinInIntUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return (int) (Math.sin((double) x) * 1_000_000_000.0);
        }
    }

    /** Java narrowing at the boundaries: NaN to 0, infinities and out-of-range values saturate. */
    public static final class NarrowingUdf implements UDF1<Double, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Double x) {
            return (long) (int) (double) x * 31 + (long) (double) x;
        }
    }

    // ------------------------------------------------------------------------------ second review

    /** Codex 2-A: the parent implements UDF2<Integer,Integer,Integer> returning x + 1 ... */
    public static class BoxedParentUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return x + 1;
        }
    }

    /** ... and the child adds call(int,int) returning x + 2, which Spark never calls. */
    public static class PrimitiveOverloadChildUdf extends BoxedParentUdf {
        private static final long serialVersionUID = 1L;

        public int call(int x, int y) {
            return x + 2;
        }
    }

    public static final class InheritingChildUdf extends BoxedParentUdf {
        private static final long serialVersionUID = 1L;
    }

    public abstract static class GenericBaseUdf<T> implements UDF1<T, Integer> {
        private static final long serialVersionUID = 1L;
    }

    /** The bridge call(Object)Object is in this class and forwards to call(Long). */
    public static final class BridgedChildUdf extends GenericBaseUdf<Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return Long.bitCount(x) * 3;
        }

        public Integer call(Integer x) {
            return -1;
        }
    }

    /** Codex 2-B: readResolve rebuilds the UDF from a system property of whichever JVM runs it. */
    public static final class ResolvedFactorUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int factor;

        public ResolvedFactorUdf(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private Object readResolve() {
            return new ResolvedFactorUdf(Integer.getInteger("review.factor", 9));
        }
    }

    /** Constructor constants: a long, a boolean and an int. */
    public static final class MaskedUdf implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;
        private final long mask;
        private final boolean negate;
        private final int shift;

        public MaskedUdf(long mask, boolean negate, int shift) {
            this.mask = mask;
            this.negate = negate;
            this.shift = shift;
        }

        @Override
        public Long call(Long x) {
            long v = (x & mask) >>> shift;
            return negate ? -v : v;
        }
    }

    /** writeObject streams factor * 100: executors see 900 for a driver instance holding 9. */
    public static final class WrittenFactorUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int factor;

        public WrittenFactorUdf(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
            java.io.ObjectOutputStream.PutField fields = out.putFields();
            fields.put("factor", factor * 100);
            out.writeFields();
        }
    }

    /** Codex 2-C. */
    public static final class CondFloorModUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorMod(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondFloorDivUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorDiv(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondFloorModLongUdf implements UDF2<Long, Long, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long x, Long y) {
            long d;
            if (x > 1000) {
                d = 1000L;
            } else if (x > 0) {
                d = y;
            } else {
                d = -7L;
            }
            return Math.floorMod(x, d);
        }
    }

    public static final class ByMinusOneUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return x / -1 + x % -1 + Math.floorDiv(x, -1);
        }
    }

    public static final class OrOneUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return x / (y | 1);
        }
    }

    public static final class ListCondDivUdf implements UDF1<scala.collection.Seq<Long>, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(scala.collection.Seq<Long> arr) {
            long acc = 0;
            for (int i = 0; i < arr.length(); i++) {
                long v = arr.apply(i);
                acc += 1000L / (v > 5 ? v : 3L);
            }
            return acc;
        }
    }

    /** A Scala-shaped function with a primitive parameter: Spark short-circuits a null to null. */
    public static long plusOne(long x) {
        return x + 1;
    }

    // ------------------------------------------------------------------------------ the run

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 100_000;
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        SparkSession spark = IntegrationTest.session(rows);
        try {
            spark.udf().register("bitcount", new BitCountUdf(), DataTypes.IntegerType);
            spark.udf().register("active_days", new ActiveDaysListUdf(), DataTypes.IntegerType);
            spark.udf().register("long_sum", new LongSumListUdf(), DataTypes.LongType);
            spark.udf().register("stride_two", new StrideTwoListUdf(), DataTypes.IntegerType);
            spark.udf().register("bad_cast", new BadCastUdf(), DataTypes.IntegerType);
            spark.udf().register("transient_field", new TransientFieldUdf(), DataTypes.IntegerType);
            spark.udf().register("floor_div", new FloorDivUdf(), DataTypes.IntegerType);
            spark.udf().register("floor_mod_long", new FloorModLongUdf(), DataTypes.LongType);
            spark.udf().register("sin_in_int", new SinInIntUdf(), DataTypes.IntegerType);
            spark.udf().register("narrowing", new NarrowingUdf(), DataTypes.LongType);

            // Codex 1: a Scala function that adds one to a Java UDF's result. Registered through
            // SparkUserDefinedFunction so that ScalaUDF.function is exactly this lambda, as it is
            // for a function written in Scala.
            BitCountUdf inner = new BitCountUdf();
            scala.Function1<Object, Object> composed = (scala.Function1<Object, Object> & Serializable) x -> 1 + inner.call((Long) x);
            spark.udf().register("composed", new SparkUserDefinedFunction(composed, DataTypes.IntegerType,
                    seq(scala.Option.apply(Encoders.LONG())), scala.Option.apply(Encoders.INT()), scala.Option.apply("composed"), true, true));
            // A capture-free Scala-shaped function with a primitive parameter.
            scala.Function1<Object, Object> scalaShape = (scala.Function1<Object, Object> & Serializable) x -> plusOne((Long) x);
            scala.Function1<Long, Long> primitive = (scala.Function1<Long, Long> & Serializable) CodexDifferentialTest::plusOne;
            spark.udf().register("plus_one", new SparkUserDefinedFunction(primitive, DataTypes.LongType,
                    seq(scala.Option.apply(Encoders.scalaLong())), scala.Option.apply(Encoders.scalaLong()), scala.Option.apply("plus_one"), false, true));
            Objects.requireNonNull(scalaShape);

            Dataset<Row> base = spark.range(0, rows, 1, 2).selectExpr("id", "id * 7 + 1 as seed");

            System.out.println("\n=== Finding 1: UDF resolution ===");
            // RAPIDS's own udf-compiler would otherwise translate the simplest of these into Catalyst
            // (plus_one becomes `seed + 1`) or screen them out (OBJECT_ACCESS) before the provider is
            // asked, and then this would be testing RAPIDS rather than the resolver.
            spark.conf().set("spark.rapids.sql.udfCompiler.enabled", "false");
            try {
                expectGpu(spark, "F1 genuine Java UDF behind Spark's adapter", "bitcount", base, "bitcount(seed)", false);
                expectFallback(spark, "F1 composed wrapper x -> 1 + inner(x)", "composed", base, "composed(seed)");
                String reason = TornadoUdfProvider.decisions().getOrDefault("composed", "");
                check("F1 composed wrapper: declined by the resolver, not by RAPIDS's screen", reason.contains("forward") || reason.contains("adapter"), reason);
                expectGpu(spark, "F1 capture-free Scala-shaped function", "plus_one", base, "plus_one(seed)", false);
                expectGpu(spark, "F1 primitive parameter over a nullable column (Spark returns null)", "plus_one",
                        base.selectExpr("id", "CASE WHEN (id & 3) = 0 THEN NULL ELSE seed END AS seed"), "plus_one(seed)", false);
            } finally {
                spark.conf().set("spark.rapids.sql.udfCompiler.enabled", "true");
            }

            System.out.println("\n=== Findings 2 and 3: LIST folds ===");
            String mixed = "slice(array(seed, seed * 3 + 1, seed * 7), 1, CAST(id & 3 AS INT))";
            expectGpu(spark, "F3 mixed [], [x], [x,y], [x,y,z] in one batch", "active_days", base, "active_days(" + mixed + ")", false);
            expectGpu(spark, "F3 a batch of nothing but empty lists", "active_days", base, "active_days(slice(array(seed), 1, 0))", false);
            expectGpu(spark, "F3 wrapping long sum with negatives and overflow", "long_sum", base,
                    "long_sum(slice(array(seed * 3037000499, -seed, 9223372036854775807, -9223372036854775807 - 1), 1, CAST(id & 3 AS INT) + 1))", false);
            expectGpu(spark, "F3 nullable LIST type with no null rows", "active_days", base,
                    "active_days(CASE WHEN id < 0 THEN NULL ELSE array(seed, seed + 1) END)", false);
            expectBothFail(spark, "F3 a null LIST row: Java dereferences it", "active_days", base,
                    "active_days(CASE WHEN (id & 7) = 0 THEN NULL ELSE array(seed, seed + 1) END)", "java.lang.NullPointerException", "NullListException");
            expectFallback(spark, "F2 stride two", "stride_two", base, "stride_two(" + mixed + ")");

            System.out.println("\n=== Finding 4: casts ===");
            expectBothFail(spark, "F4 Integer cast to Boolean throws in Java", "bad_cast", base, "bad_cast(CAST(seed AS INT))", "java.lang.ClassCastException",
                    "java.lang.ClassCastException");
            requireNotGpu(spark, "F4 the invalid cast is not compiled", "bad_cast", base, "bad_cast(CAST(seed AS INT))");

            System.out.println("\n=== Finding 5: serialized state ===");
            expectGpu(spark, "F5 transient field takes the executor's value", "transient_field", base, "transient_field(CAST(seed AS INT))", false);

            System.out.println("\n=== Finding 6: floorDiv / floorMod ===");
            Dataset<Row> ints = base.selectExpr("id",
                    "CAST(CASE WHEN (id & 15) = 0 THEN -2147483648 ELSE seed - 50000 END AS INT) AS a",
                    "CAST(CASE WHEN (id & 15) = 0 THEN -1 ELSE (id & 7) - 3 END AS INT) AS b");
            expectGpu(spark, "F6 floorDiv, signs and MIN_VALUE/-1, no zero divisor", "floor_div", ints.where("b <> 0"), "floor_div(a, b)", false);
            expectBothFail(spark, "F6 floorDiv by zero fails the query", "floor_div", ints, "floor_div(a, b)", "java.lang.ArithmeticException", "DIVISION_BY_ZERO");
            Dataset<Row> longs = base.selectExpr("id", "seed * 1000003 - 99999999 AS a", "CASE WHEN (id & 7) = 0 THEN -1 ELSE (id & 7) - 4 END AS b");
            expectGpu(spark, "F6 floorMod long, non-zero divisors", "floor_mod_long", longs.where("b <> 0"), "floor_mod_long(a, b)", false);
            expectBothFail(spark, "F6 floorMod long by zero fails the query", "floor_mod_long", longs, "floor_mod_long(a, b)", "java.lang.ArithmeticException",
                    "DIVISION_BY_ZERO");

            System.out.println("\n=== Finding 7: floating point in an integer signature ===");
            expectFallback(spark, "F7 sin inside INT->INT, incompatibleOps off", "sin_in_int", base, "sin_in_int(CAST(seed AS INT))");
            spark.conf().set("spark.rapids.sql.incompatibleOps.enabled", "true");
            try {
                Dataset<Row> doubles = base.selectExpr("id", "CASE id & 7 WHEN 0 THEN double('NaN') WHEN 1 THEN double('Infinity') WHEN 2 THEN double('-Infinity') "
                        + "WHEN 3 THEN 1e300 WHEN 4 THEN -1e300 WHEN 5 THEN 2147483647.9 WHEN 6 THEN -2147483648.9 ELSE (seed - 50000) / 7.0 END AS x");
                expectGpu(spark, "F7 double narrowing boundaries, bit-exact", "narrowing", doubles, "narrowing(x)", false);
            } finally {
                spark.conf().set("spark.rapids.sql.incompatibleOps.enabled", "false");
            }

            secondReview(spark, base);

            System.out.println("\n--- planning decisions ---");
            TornadoUdfProvider.decisions().forEach((name, decision) -> System.out.println("  " + name + " -> " + decision));
        } finally {
            spark.stop();
        }
        System.out.println();
        System.out.println("=== CHECKS: " + checks + ", FAILURES: " + failures + " ===");
        if (failures > 0) {
            System.exit(1);
        }
    }

    /** Codex's second review, through ordinary registration; see the fixtures above. */
    private static void secondReview(SparkSession spark, Dataset<Row> base) {
        spark.udf().register("inherited_overload", new PrimitiveOverloadChildUdf(), DataTypes.IntegerType);
        spark.udf().register("inheriting_child", new InheritingChildUdf(), DataTypes.IntegerType);
        spark.udf().register("bridged_child", new BridgedChildUdf(), DataTypes.IntegerType);
        BitCountUdf popcount = new BitCountUdf();
        UDF1<Long, Integer> popcountInterface = popcount;
        UDF1<Long, Integer> typedForwarder = x -> popcountInterface.call(x);
        spark.udf().register("typed_forwarder", typedForwarder, DataTypes.IntegerType);
        UDF1<Long, Integer> javaLambda = x -> Long.bitCount(x) + 1;
        spark.udf().register("java_lambda", javaLambda, DataTypes.IntegerType);
        scala.Function1<Long, Integer> scalaTyped = (scala.Function1<Long, Integer> & Serializable) x -> popcount.call(x);
        spark.udf().register("scala_typed_forwarder", new SparkUserDefinedFunction(scalaTyped, DataTypes.IntegerType,
                seq(scala.Option.apply(Encoders.LONG())), scala.Option.apply(Encoders.INT()), scala.Option.apply("scala_typed_forwarder"), true, true));
        spark.udf().register("resolved_factor", new ResolvedFactorUdf(9), DataTypes.IntegerType);
        spark.udf().register("masked", new MaskedUdf(0xFFFF_0000L, true, 16), DataTypes.LongType);
        spark.udf().register("written_factor", new WrittenFactorUdf(9), DataTypes.IntegerType);
        spark.udf().register("cond_floor_mod", new CondFloorModUdf(), DataTypes.IntegerType);
        spark.udf().register("cond_floor_div", new CondFloorDivUdf(), DataTypes.IntegerType);
        spark.udf().register("cond_floor_mod_long", new CondFloorModLongUdf(), DataTypes.LongType);
        spark.udf().register("by_minus_one", new ByMinusOneUdf(), DataTypes.IntegerType);
        spark.udf().register("or_one", new OrOneUdf(), DataTypes.IntegerType);
        spark.udf().register("list_cond_div", new ListCondDivUdf(), DataTypes.LongType);

        spark.conf().set("spark.rapids.sql.udfCompiler.enabled", "false");
        try {
            System.out.println("\n=== Second review A: dispatch and forwarding ===");
            Dataset<Row> pairs = base.selectExpr("id", "CAST(seed - 50000 AS INT) AS x", "CAST(id % 11 AS INT) AS y");
            expectGpu(spark, "2A inherited call(Integer,Integer) with a call(int,int) overload in the child", "inherited_overload", pairs, "inherited_overload(x, y)", false);
            check("2A the CPU reference is the inherited method (x + 1)", cpuValue(spark, pairs, "inherited_overload(x, y)", "x + 1"), "CPU differs from x + 1");
            expectGpu(spark, "2A inherited implementation, child adds nothing", "inheriting_child", pairs, "inheriting_child(x, y)", false);
            expectGpu(spark, "2A bridge in a generic child, extra overload", "bridged_child", base, "bridged_child(seed)", false);
            expectGpu(spark, "2A typed Java UDF1 forwarder (Codex's restored pattern)", "typed_forwarder", base, "typed_forwarder(seed * 2654435761)", false);
            expectGpu(spark, "2A plain Java UDF1 lambda", "java_lambda", base, "java_lambda(seed)", false);
            expectGpu(spark, "2A typed Scala Function1 forwarder", "scala_typed_forwarder", base, "scala_typed_forwarder(seed)", false);

            System.out.println("\n=== Second review B: executor-invariant constants ===");
            Dataset<Row> ints = base.selectExpr("id", "CAST(seed - 50000 AS INT) AS x");
            expectFallback(spark, "2B readResolve from a system property", "resolved_factor", ints, "resolved_factor(x)");
            expectGpu(spark, "2B final long, boolean and int constants", "masked", base, "masked(seed * 2654435761)", false);
            expectGpu(spark, "2B deterministic writeObject (executor sees 900)", "written_factor", ints, "written_factor(x)", false);
            check("2B the CPU reference uses the streamed 900", cpuValue(spark, ints, "written_factor(x)", "x + 900"), "CPU differs from x + 900");

            System.out.println("\n=== Second review C: conditional divisors ===");
            Dataset<Row> divisors = base.selectExpr("id", "CAST(CASE WHEN (id & 15) = 0 THEN 7 ELSE seed % 40 - 20 END AS INT) AS x",
                    "CAST(CASE WHEN (id & 15) = 0 THEN (id % 5) + 1 ELSE (id % 5) - 2 END AS INT) AS y").where("NOT (x = 7 AND y = 0)");
            expectGpu(spark, "2C floorMod(x, x == 7 ? y : 3), no zero reaches it", "cond_floor_mod", divisors, "cond_floor_mod(x, y)", false);
            expectGpu(spark, "2C floorDiv(x, x == 7 ? y : 3), no zero reaches it", "cond_floor_div", divisors, "cond_floor_div(x, y)", false);
            Dataset<Row> zeroAt7 = base.selectExpr("id", "CAST(CASE WHEN (id & 15) = 0 THEN 7 ELSE seed % 40 - 20 END AS INT) AS x", "CAST(0 AS INT) AS y");
            expectBothFail(spark, "2C floorMod(7, 0) fails the query", "cond_floor_mod", zeroAt7, "cond_floor_mod(x, y)", "java.lang.ArithmeticException", "DIVISION_BY_ZERO");
            expectBothFail(spark, "2C floorDiv(7, 0) fails the query", "cond_floor_div", zeroAt7, "cond_floor_div(x, y)", "java.lang.ArithmeticException", "DIVISION_BY_ZERO");
            Dataset<Row> zeroElsewhere = base.selectExpr("id", "CAST(seed % 40 - 20 AS INT) AS x", "CAST(0 AS INT) AS y").where("x <> 7");
            expectGpu(spark, "2C y = 0 everywhere but x never 7: Java does not divide by y", "cond_floor_mod", zeroElsewhere, "cond_floor_mod(x, y)", false);
            Dataset<Row> longs = base.selectExpr("id", "CASE WHEN (id & 3) = 0 THEN seed * 1000003 ELSE seed % 2000 - 1000 END AS x",
                    "CASE WHEN (id & 7) = 1 THEN -3 ELSE (id % 9) + 1 END AS y");
            expectGpu(spark, "2C long floorMod, divisor assigned on three paths", "cond_floor_mod_long", longs, "cond_floor_mod_long(x, y)", false);
            expectBothFail(spark, "2C long floorMod reaching a zero divisor fails", "cond_floor_mod_long", longs.selectExpr("id", "abs(x) % 900 + 1 AS x", "0L AS y"),
                    "cond_floor_mod_long(x, y)", "java.lang.ArithmeticException", "DIVISION_BY_ZERO");
            Dataset<Row> minValues = base.selectExpr("id", "CAST(CASE WHEN (id & 7) = 0 THEN -2147483648 ELSE seed - 50000 END AS INT) AS x");
            expectGpu(spark, "2C x / -1, x % -1, floorDiv(x, -1) including MIN_VALUE", "by_minus_one", minValues, "by_minus_one(x)", false);
            expectGpu(spark, "2C x / (y | 1) needs no zero test", "or_one", divisors, "or_one(x, y)", false);
            String mixed = "slice(array(seed % 13 - 6, seed % 7, seed * 3), 1, CAST(id & 3 AS INT))";
            expectGpu(spark, "2C LIST fold dividing by a conditional element divisor", "list_cond_div", base, "list_cond_div(" + mixed + ")", false);
        } finally {
            spark.conf().set("spark.rapids.sql.udfCompiler.enabled", "true");
        }
    }

    /** Whether CPU Spark's result equals {@code expected} on every row. */
    private static boolean cpuValue(SparkSession spark, Dataset<Row> source, String call, String expected) {
        spark.conf().set("spark.rapids.sql.enabled", "false");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "false");
        return source.selectExpr(call + " AS a", expected + " AS b").where("NOT (a <=> b)").count() == 0;
    }

    @SafeVarargs
    private static <T> scala.collection.immutable.Seq<T> seq(T... items) {
        return scala.jdk.javaapi.CollectionConverters.asScala(java.util.List.of(items)).toList();
    }

    // ------------------------------------------------------------------------------ checks

    private static void expectGpu(SparkSession spark, String label, String udf, Dataset<Row> source, String call, boolean unused) {
        System.out.println("--- " + label + " ---");
        List<Row> cpu = run(spark, source, call, false);
        List<Row> gpu = run(spark, source, call, true);
        compare(label, cpu, gpu);
        String plan = plan(spark, source, call);
        check(label + ": the UDF runs on the GPU as gpu_" + udf, plan.contains("gpu_" + udf), firstLines(plan));
        check(label + ": no CPU bridge or row-based UDF", !plan.contains("GpuCpuBridge") && !(plan.contains("GpuRowBasedScalaUDF") || plan.contains("gpu_row_based_")), firstLines(plan));
    }

    private static void expectFallback(SparkSession spark, String label, String udf, Dataset<Row> source, String call) {
        System.out.println("--- " + label + " ---");
        List<Row> cpu = run(spark, source, call, false);
        List<Row> enabled = run(spark, source, call, true);
        compare(label, cpu, enabled);
        requireNotGpu(spark, label, udf, source, call);
    }

    private static void requireNotGpu(SparkSession spark, String label, String udf, Dataset<Row> source, String call) {
        String plan = planWithoutRunning(spark, source, call);
        check(label + ": NOT compiled for the GPU (gpu_" + udf + " absent)", !plan.contains("gpu_" + udf), firstLines(plan));
        check(label + ": the UDF is still in the plan, on the CPU path", plan.contains(udf), firstLines(plan));
        String decision = TornadoUdfProvider.decisions().getOrDefault(udf, "");
        check(label + ": the provider recorded a refusal", decision.startsWith("TORNADOVM_INELIGIBLE"), decision);
        System.out.println("  refusal: " + decision);
    }

    /** Both runs must fail, each with its expected cause. */
    private static void expectBothFail(SparkSession spark, String label, String udf, Dataset<Row> source, String call, String cpuCause, String gpuCause) {
        System.out.println("--- " + label + " ---");
        String cpu = failureOf(spark, source, call, false);
        String gpu = failureOf(spark, source, call, true);
        System.out.println("  cpu: " + cpu);
        System.out.println("  gpu: " + gpu);
        check(label + ": CPU Spark fails with " + cpuCause, cpu != null && cpu.contains(cpuCause), String.valueOf(cpu));
        check(label + ": the GPU-enabled run fails with " + gpuCause, gpu != null && gpu.contains(gpuCause), String.valueOf(gpu));
        for (String unrelated : new String[] { "UnsatisfiedLinkError", "NoClassDefFoundError", "NVRTC", "nvrtc", "NoSuchMethodError", "CUDA_ERROR" }) {
            check(label + ": the GPU failure is not " + unrelated, gpu == null || !gpu.contains(unrelated), gpu);
        }
    }

    /** The whole cause chain of the failure as class: message lines, or null if it succeeded. */
    private static String failureOf(SparkSession spark, Dataset<Row> source, String call, boolean enabled) {
        try {
            run(spark, source, call, enabled);
            return null;
        } catch (Exception e) {
            StringBuilder chain = new StringBuilder();
            for (Throwable t = e; t != null; t = t.getCause() == t ? null : t.getCause()) {
                chain.append(t.getClass().getName()).append(": ").append(String.valueOf(t.getMessage()).lines().findFirst().orElse("")).append(" | ");
            }
            return chain.toString();
        }
    }

    private static List<Row> run(SparkSession spark, Dataset<Row> source, String call, boolean enabled) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(enabled));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(enabled));
        return source.selectExpr("id", call + " AS result").orderBy("id").collectAsList();
    }

    private static String plan(SparkSession spark, Dataset<Row> source, String call) {
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
        Dataset<Row> df = source.selectExpr("id", call + " AS result");
        df.collect();
        return df.queryExecution().executedPlan().toString();
    }

    private static String planWithoutRunning(SparkSession spark, Dataset<Row> source, String call) {
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
        return source.selectExpr("id", call + " AS result").queryExecution().executedPlan().toString();
    }

    /** Values and nulls, row for row, with the result compared exactly. */
    private static void compare(String label, List<Row> cpu, List<Row> gpu) {
        check(label + ": row count", cpu.size() == gpu.size(), cpu.size() + " vs " + gpu.size());
        int mismatches = 0;
        int nulls = 0;
        List<String> examples = new ArrayList<>();
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            Object c = cpu.get(i).get(1);
            Object g = gpu.get(i).get(1);
            if (c == null) {
                nulls++;
            }
            if (!Objects.equals(c, g)) {
                if (mismatches++ < 3) {
                    examples.add("row " + cpu.get(i).get(0) + ": cpu=" + c + " gpu=" + g);
                }
            }
        }
        examples.forEach(e -> System.out.println("  " + e));
        System.out.println("  rows=" + cpu.size() + " cpu-nulls=" + nulls);
        check(label + ": every value and null identical to CPU Spark", mismatches == 0, mismatches + " rows differ");
    }

    private static String firstLines(String plan) {
        return plan.lines().limit(6).reduce("", (a, b) -> a + "\n    " + b);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }

    @SuppressWarnings("unused")
    private static final DataType UNUSED = DataTypes.NullType;
}
