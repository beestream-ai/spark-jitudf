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
package ai.beestream.jitudf;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

/**
 * The representative UDF patterns whose GPU eligibility the correctness work must not silently
 * reduce, and the counterexamples it must stop accepting.
 *
 * <p>
 * Each case says what it is <em>meant</em> to be: {@link Intent#SAFE} for a pattern whose GPU
 * result equals the Java/Spark result and which must therefore stay eligible, and
 * {@link Intent#UNSAFE} for one that was only ever accepted because a transformation changed its
 * meaning. {@link CoverageMatrixTest} prints the matrix and enforces both directions.
 * </p>
 */
final class CoverageCorpus {

    /**
     * SAFE: the GPU result can equal Java's, and the pattern must be GPU-eligible. UNSAFE: its
     * semantics cannot currently be preserved, and it must stay on the CPU. GAP: safe in principle
     * but not yet supported -- a coverage gap, kept separate so that it is never counted as a
     * correct refusal.
     */
    enum Intent {
        SAFE, UNSAFE, GAP
    }

    /**
     * One pattern. {@code check}, when present, is a property the emitted CUDA must have for an
     * acceptance to be correct (the right method compiled, a zero test present or absent): an
     * accepted row that fails it is wrong code, not coverage.
     */
    record Case(String id, String pattern, Intent intent, Object function, DataType[] inputs, DataType output, boolean nullable, boolean incompatibleOps,
            String checkName, java.util.function.Predicate<String> check) {
        Case(String id, String pattern, Intent intent, Object function, DataType[] inputs, DataType output, boolean nullable, boolean incompatibleOps) {
            this(id, pattern, intent, function, inputs, output, nullable, incompatibleOps, null, null);
        }
    }

    /** The error channel's zero test, for an int or a long divisor. */
    static boolean zeroTest(String cuda) {
        return java.util.regex.Pattern.compile("== 0(LL)?\\) return 2;").matcher(cuda).find();
    }

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;
    static final DataType F = DataTypes.FloatType;
    static final DataType D = DataTypes.DoubleType;
    static final DataType B = DataTypes.BooleanType;
    static final DataType LIST_L = DataTypes.createArrayType(L, false);
    static final DataType LIST_I = DataTypes.createArrayType(I, false);

    private CoverageCorpus() {
    }

    // ------------------------------------------------------------------------------- safe shapes

    public static final class SingleLong implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            return x * 31 + 7;
        }
    }

    public static final class MixedTypes implements UDF2<Integer, Float, Double> {
        @Override
        public Double call(Integer a, Float b) {
            return a.doubleValue() * b;
        }
    }

    public static final class ThreeInputs implements UDF3<Long, Integer, Long, Long> {
        @Override
        public Long call(Long a, Integer b, Long c) {
            return (a ^ c) + b;
        }
    }

    public static final class IntOut implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return x * x - 3;
        }
    }

    public static final class FloatOut implements UDF1<Float, Float> {
        @Override
        public Float call(Float x) {
            return x * 0.5f + 1f;
        }
    }

    public static final class DoubleOut implements UDF1<Double, Double> {
        @Override
        public Double call(Double x) {
            return x * x + 0.25;
        }
    }

    public static final class BoolOut implements UDF1<Long, Boolean> {
        @Override
        public Boolean call(Long x) {
            return (x & 1L) == 0L;
        }
    }

    public static final class Branches implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            if (x < 0) {
                return -1;
            }
            if (x > 100) {
                return 1;
            }
            return 0;
        }
    }

    public static final class CanonicalLoop implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            long acc = 0;
            for (int i = 0; i < 16; i++) {
                acc = acc * 31 + (x >>> i);
            }
            return acc;
        }
    }

    public static final class Collatz implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long n) {
            long x = n;
            int steps = 0;
            while (x != 1 && x > 0) {
                x = (x & 1) == 0 ? x / 2 : 3 * x + 1;
                steps++;
            }
            return steps;
        }
    }

    public static final class NestedHelpers implements UDF1<Long, Long> {
        static long mix(long v) {
            return rotate(v) ^ (v >>> 7);
        }

        static long rotate(long v) {
            return Long.rotateLeft(v, 13) * 0x9E3779B97F4A7C15L;
        }

        @Override
        public Long call(Long x) {
            return mix(x);
        }
    }

    public static final class VariableDiv implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return a / b + a % b;
        }
    }

    public static final class ConstantDiv implements UDF1<Long, Long> {
        @Override
        public Long call(Long a) {
            return a / 7 + a % 10;
        }
    }

    public static final class MathCalls implements UDF1<Double, Double> {
        @Override
        public Double call(Double x) {
            return Math.sqrt(Math.abs(x)) + Math.max(x, 0.0);
        }
    }

    public static final class IntMath implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return Math.max(Math.abs(a), b) + Math.min(a, b);
        }
    }

    public static final class FloorModConstant implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long h) {
            return Math.floorMod(Long.hashCode(h) * 31, 1000);
        }
    }

    public static final class FloorModConstantInt implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer h) {
            return Math.floorMod(h * 31, 1000);
        }
    }

    public static final class FloorDivVariable implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return Math.floorDiv(a, b);
        }
    }

    public static final class Configured implements UDF1<Integer, Integer> {
        private final int weight;
        private final int offset;

        public Configured(int weight, int offset) {
            this.weight = weight;
            this.offset = offset;
        }

        @Override
        public Integer call(Integer x) {
            return x * weight + offset;
        }
    }

    public static final class ConfiguredDouble implements UDF1<Double, Double> {
        private final double scale;

        public ConfiguredDouble(double scale) {
            this.scale = scale;
        }

        @Override
        public Double call(Double x) {
            return x * scale;
        }
    }

    public static final class NullTolerant implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            if (x == null) {
                return -1L;
            }
            return x + 1;
        }
    }

    public static final class NullReturning implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            if (x == null || x < 0) {
                return null;
            }
            return x * 2;
        }
    }

    public static final class NullableUnboxing implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return a + b;
        }
    }

    public static final class ActiveDays implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += Long.bitCount(arr[i]);
            }
            return acc;
        }
    }

    public static final class ListSum implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class ListSquares implements UDF1<int[], Integer> {
        @Override
        public Integer call(int[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] * arr[i] + 1;
            }
            return acc;
        }
    }

    public static final class ListSeqSum implements UDF1<scala.collection.Seq<Long>, Long> {
        @Override
        public Long call(scala.collection.Seq<Long> arr) {
            long acc = 0;
            for (int i = 0; i < arr.length(); i++) {
                acc += arr.apply(i);
            }
            return acc;
        }
    }

    public static final class ValidCasts implements UDF2<Integer, Long, Long> {
        @Override
        public Long call(Integer a, Long b) {
            return (Long) (Object) b + ((Integer) (Object) a).longValue();
        }
    }

    public static final class IntToLongConversion implements UDF1<Integer, Long> {
        @Override
        public Long call(Integer a) {
            return a.longValue() << 3;
        }
    }

    // ----------------------------------------------------------------------- unsafe (Codex probes)

    public static final class BitCount implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    public static final class StrideTwo implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i += 2) {
                acc += Long.bitCount(arr[i]);
            }
            return acc;
        }
    }

    public static final class BadCast implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return ((Boolean) (Object) x).booleanValue() ? 21 : 42;
        }
    }

    public static final class TransientField implements UDF1<Integer, Integer> {
        private final transient int factor;

        public TransientField() {
            this.factor = 9;
        }

        @Override
        public Integer call(Integer x) {
            return factor + x;
        }
    }

    public static final class SinInInt implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return (int) (Math.sin((double) x) * 1_000_000_000.0);
        }
    }

    /** A static scalar implementation with a primitive signature, which is how Scala compiles {@code (x: Long) => ...}. */
    public static final class ScalaShapes {
        public static long plusOne(long x) {
            return x + 1;
        }

        public static long plusOne(int x) {
            return x + 2;
        }
    }

    @SuppressWarnings("unchecked")
    static List<Case> all() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("C01", "single-input fixed-width (LONG->LONG)", Intent.SAFE, new SingleLong(), new DataType[] { L }, L, false, false));
        cases.add(new Case("C02", "multi-input mixed types (INT,FLOAT->DOUBLE)", Intent.SAFE, new MixedTypes(), new DataType[] { I, F }, D, false, true));
        cases.add(new Case("C03", "three inputs (LONG,INT,LONG->LONG)", Intent.SAFE, new ThreeInputs(), new DataType[] { L, I, L }, L, false, false));
        cases.add(new Case("C04", "INT32 output", Intent.SAFE, new IntOut(), new DataType[] { I }, I, false, false));
        cases.add(new Case("C05", "FLOAT32 output (incompatibleOps on)", Intent.SAFE, new FloatOut(), new DataType[] { F }, F, false, true));
        cases.add(new Case("C06", "FLOAT64 output (incompatibleOps on)", Intent.SAFE, new DoubleOut(), new DataType[] { D }, D, false, true));
        cases.add(new Case("C07", "BOOL8 output", Intent.SAFE, new BoolOut(), new DataType[] { L }, B, false, false));
        cases.add(new Case("C08", "branches / conditional returns", Intent.SAFE, new Branches(), new DataType[] { I }, I, false, false));
        cases.add(new Case("C09", "canonical counted loop", Intent.SAFE, new CanonicalLoop(), new DataType[] { L }, L, false, false));
        cases.add(new Case("C10", "data-dependent loop (Collatz)", Intent.SAFE, new Collatz(), new DataType[] { L }, I, false, false));
        cases.add(new Case("C11", "nested static helpers", Intent.SAFE, new NestedHelpers(), new DataType[] { L }, L, false, false));
        cases.add(new Case("C12", "integer division/remainder by a variable", Intent.SAFE, new VariableDiv(), new DataType[] { I, I }, I, false, false));
        cases.add(new Case("C13", "integer division/remainder by constants", Intent.SAFE, new ConstantDiv(), new DataType[] { L }, L, false, false));
        cases.add(new Case("C14", "Math.sqrt/abs/max on doubles (incompatibleOps on)", Intent.SAFE, new MathCalls(), new DataType[] { D }, D, false, true));
        cases.add(new Case("C15", "integer Math.abs/min/max", Intent.SAFE, new IntMath(), new DataType[] { I, I }, I, false, false));
        cases.add(new Case("C17", "Math.floorDiv by a variable", Intent.SAFE, new FloorDivVariable(), new DataType[] { I, I }, I, false, false));
        cases.add(new Case("C18", "constructor-configured final int fields", Intent.SAFE, new Configured(3, 11), new DataType[] { I }, I, false, false));
        cases.add(new Case("C19", "constructor-configured final double field (incompatibleOps on)", Intent.SAFE, new ConfiguredDouble(2.5), new DataType[] { D }, D, false, true));
        cases.add(new Case("C20", "nullable boxed input, unboxing body", Intent.SAFE, new NullableUnboxing(), new DataType[] { I, I }, I, true, false));
        cases.add(new Case("C21", "null-returning UDF", Intent.SAFE, new NullReturning(), new DataType[] { L }, L, true, false));
        cases.add(new Case("C22", "non-null result for a null input", Intent.SAFE, new NullTolerant(), new DataType[] { L }, L, true, false));
        cases.add(new Case("C23", "LIST fold: popcount sum (activity bitmap)", Intent.SAFE, new ActiveDays(), new DataType[] { LIST_L }, I, false, false));
        cases.add(new Case("C24", "LIST fold: plain long sum", Intent.SAFE, new ListSum(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C25", "LIST fold: element transform x*x+1", Intent.SAFE, new ListSquares(), new DataType[] { LIST_I }, I, false, false));
        cases.add(new Case("C26", "LIST fold through scala Seq", Intent.SAFE, new ListSeqSum(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C27", "Spark ToScalaUDF adapter around a Java UDF", Intent.SAFE,
                org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new BitCount()), new DataType[] { L }, I, false, false));
        cases.add(new Case("C28", "Spark ToScalaUDF adapter, two inputs", Intent.SAFE,
                org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new VariableDiv()), new DataType[] { I, I }, I, false, false));
        cases.add(new Case("C29", "capture-free static lambda (Scala shape), overloaded name", Intent.SAFE,
                (scala.Function1<Long, Long> & Serializable) ScalaShapes::plusOne, new DataType[] { L }, L, false, false));
        cases.add(new Case("C30", "valid casts through Object", Intent.SAFE, new ValidCasts(), new DataType[] { I, L }, L, false, false));
        cases.add(new Case("C31", "Integer.longValue widening", Intent.SAFE, new IntToLongConversion(), new DataType[] { I }, L, false, false));
        cases.add(new Case("C32", "LIST fold over a nullable LIST column", Intent.SAFE, new ActiveDays(), new DataType[] { LIST_L }, I, true, false));
        cases.add(new Case("C33", "Math.floorMod by a constant, int", Intent.SAFE, new FloorModConstantInt(), new DataType[] { I }, I, false, false));

        BitCount inner = new BitCount();
        cases.add(new Case("U01", "composed wrapper x -> 1 + inner.call(x)", Intent.UNSAFE,
                (scala.Function1<Object, Object> & Serializable) x -> 1 + (Integer) inner.call((Long) x), new DataType[] { L }, I, false, false));
        cases.add(new Case("U02", "LIST loop with stride two", Intent.UNSAFE, new StrideTwo(), new DataType[] { LIST_L }, I, false, false));
        cases.add(new Case("U04", "invalid Integer->Boolean cast", Intent.UNSAFE, new BadCast(), new DataType[] { I, I }, I, false, false));
        // Now compiled with the value an executor sees after serialization (0), not the driver's 9.
        cases.add(new Case("C34", "final transient field, folded to its executor-visible value", Intent.SAFE, new TransientField(), new DataType[] { I }, I, false, false));
        cases.add(new Case("C35", "Math.sin inside an INT signature, incompatibleOps on", Intent.SAFE, new SinInInt(), new DataType[] { I }, I, false, true));
        cases.add(new Case("U07", "Math.sin inside an INT signature, incompatibleOps off", Intent.UNSAFE, new SinInInt(), new DataType[] { I }, I, false, false));

        // Extended rows, drawn from the per-finding regression tests, so that "previously accepted"
        // is measured against the baseline commit rather than inferred.
        DataType listD = DataTypes.createArrayType(D, false);
        cases.add(new Case("C36", "LIST fold with temporaries in the body", Intent.SAFE, new ListFoldSemanticsTest.WithTemporary(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C37", "LIST fold with a branch in the element expression", Intent.SAFE, new ListFoldSemanticsTest.WithBranch(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C38", "LIST fold with the bound held in a local", Intent.SAFE, new ListFoldSemanticsTest.BoundInALocal(), new DataType[] { LIST_L }, I, false, false));
        cases.add(new Case("C39", "LIST fold written acc = E + acc", Intent.SAFE, new ListFoldSemanticsTest.TotalFirst(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C40", "LIST fold, int overflow wraps", Intent.SAFE, new ListFoldSemanticsTest.IntOverflow(), new DataType[] { LIST_I }, I, false, false));
        cases.add(new Case("C41", "LIST fold dividing by the element (error channel)", Intent.SAFE, new ListFoldSemanticsTest.DividesByElement(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("C42", "LIST fold over doubles (both RAPIDS FP switches on)", Intent.SAFE, new ListFoldSemanticsTest.DoubleSum(), new DataType[] { listD }, D, false, true));
        cases.add(new Case("C43", "final int, long and boolean fields", Intent.SAFE, new CastAndStateSemanticsTest.TwoConstants(3, 11, true), new DataType[] { I }, I, false, false));
        cases.add(new Case("C44", "final field set by readResolve", Intent.SAFE, new CastAndStateSemanticsTest.Resolved(3), new DataType[] { I }, I, false, false));
        cases.add(new Case("C45", "cast of a returned box", Intent.SAFE, new CastAndStateSemanticsTest.CastOfReturnedBox(), new DataType[] { I }, I, false, false));
        cases.add(new Case("C46", "Boolean cast", Intent.SAFE, new CastAndStateSemanticsTest.BooleanCast(), new DataType[] { B }, I, false, false));
        cases.add(new Case("C47", "Math.floorMod by a variable, long", Intent.SAFE, new ArithmeticEffectsTest.FloorModLong(), new DataType[] { L, L }, L, false, false));
        cases.add(new Case("C48", "floorDiv inside a static helper", Intent.SAFE, new ArithmeticEffectsTest.FloorDivInHelper(), new DataType[] { I, I }, I, false, false));
        cases.add(new Case("C49", "double->int/long narrowing (incompatibleOps on)", Intent.SAFE, new ArithmeticEffectsTest.Narrowing(), new DataType[] { D }, I, false, true));
        cases.add(new Case("C50", "Boolean predicate over sqrt (incompatibleOps on)", Intent.SAFE, new ArithmeticEffectsTest.FpPredicate(), new DataType[] { I }, B, false, true));

        cases.add(new Case("U10", "LIST loop with continue", Intent.UNSAFE, new ListFoldSemanticsTest.Skips(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U11", "LIST loop with a conditional update", Intent.UNSAFE, new ListFoldSemanticsTest.ConditionalUpdate(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U12", "LIST loop with an early return", Intent.UNSAFE, new ListFoldSemanticsTest.EarlyReturn(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U13", "LIST loop with break", Intent.UNSAFE, new ListFoldSemanticsTest.Breaks(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U14", "LIST loop starting at index 1", Intent.UNSAFE, new ListFoldSemanticsTest.StartsAtOne(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U15", "LIST loop with two accumulator writes", Intent.UNSAFE, new ListFoldSemanticsTest.TwoWrites(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U16", "LIST loop acc = acc*31 + x", Intent.UNSAFE, new ListFoldSemanticsTest.OrderDependent(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U17", "LIST loop with a non-zero initial accumulator", Intent.UNSAFE, new ListFoldSemanticsTest.NonZeroStart(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U18", "LIST loop with work after the loop", Intent.UNSAFE, new ListFoldSemanticsTest.AfterLoop(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U19", "LIST reversed traversal", Intent.UNSAFE, new ListFoldSemanticsTest.Reversed(), new DataType[] { LIST_L }, L, false, false));
        cases.add(new Case("U20", "Integer->Long cast", Intent.UNSAFE, new CastAndStateSemanticsTest.IntegerToLong(), new DataType[] { I }, L, false, false));
        cases.add(new Case("U21", "row-dependent cast (Integer or Long)", Intent.UNSAFE, new CastAndStateSemanticsTest.CastOfBranchMixedType(), new DataType[] { I, L }, I, false, false));
        cases.add(new Case("U22", "UDF replaced by writeReplace", Intent.UNSAFE, new CastAndStateSemanticsTest.Replaced(), new DataType[] { I }, I, false, false));
        cases.add(new Case("U23", "Boolean predicate over sqrt, incompatibleOps off", Intent.UNSAFE, new ArithmeticEffectsTest.FpPredicate(), new DataType[] { I }, B, false, false));
        cases.add(new Case("U24", "double conversion inside INT->LONG, incompatibleOps off", Intent.UNSAFE, new ArithmeticEffectsTest.FpViaBoxConversion(), new DataType[] { I }, L, false, false));
        cases.add(new Case("U25", "LIST fold over doubles, incompatibleOps off", Intent.UNSAFE, new ListFoldSemanticsTest.DoubleSum(), new DataType[] { listD }, D, false, false));
        addSecondReview(cases);
        return cases;
    }

    /**
     * Codex's second review: inherited dispatch, forwarders, serialization hooks, conditional
     * divisors. Rows S, U and G continue the numbering; checks make "accepted" mean "accepted
     * with the right code".
     */
    static void addSecondReview(List<Case> cases) {
        DataType listL = DataTypes.createArrayType(L, false);
        org.apache.spark.sql.internal.ToScalaUDF$ adapt = org.apache.spark.sql.internal.ToScalaUDF$.MODULE$;
        java.util.function.Predicate<String> plusOne = cuda -> cuda.contains("(unsigned int) 1)") && !cuda.contains("(unsigned int) 2)");
        java.util.function.Predicate<String> guarded = CoverageCorpus::zeroTest;
        java.util.function.Predicate<String> unguarded = cuda -> cuda.contains("__device__ void");

        // ---- dispatch and forwarding
        cases.add(new Case("S01", "inherited interface implementation (child adds nothing)", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.InheritingChild()), new DataType[] { I, I }, I, false, false, "computes x+1", plusOne));
        cases.add(new Case("S02", "Codex A: child adds call(int,int) overload; Spark runs inherited call(Integer,Integer)", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.PrimitiveOverloadChild()), new DataType[] { I, I }, I, false, false, "computes x+1, not x+2", plusOne));
        cases.add(new Case("S03", "child overrides the boxed call", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.OverridingChild()), new DataType[] { I, I }, I, false, false, "computes x*y", c -> c.contains("*")));
        cases.add(new Case("S04", "child overrides and adds overloads on both sides", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.OverridingWithOverloads()), new DataType[] { I, I }, I, false, false, "computes x-y",
                c -> c.contains("- (unsigned int) arg1") || c.contains("-")));
        cases.add(new Case("S05", "generic base, bridge in the child, extra overload", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.BridgedChild()), new DataType[] { L }, I, false, false, "computes bitCount*3", c -> c.contains("* (unsigned int) 3")));
        cases.add(new Case("S06", "implementation in a parent that fixes T, bridge followed by dispatch", Intent.SAFE,
                adapt.apply(new DispatchResolutionTest.LongImplChild()), new DataType[] { L }, I, false, false, null, null));
        DispatchResolutionTest.Popcount popcount = new DispatchResolutionTest.Popcount();
        UDF1<Long, Integer> popcountInterface = popcount;
        cases.add(new Case("S07", "Codex regression: typed forwarder Function1<Long,Integer> x -> inner.call(x)", Intent.SAFE,
                (scala.Function1<Long, Integer> & Serializable) x -> popcount.call(x), new DataType[] { L }, I, false, false, null, null));
        cases.add(new Case("S08", "erased forwarder Function1<Object,Object> x -> inner.call((Long) x)", Intent.SAFE,
                (scala.Function1<Object, Object> & Serializable) x -> popcount.call((Long) x), new DataType[] { L }, I, false, false, null, null));
        UDF1<Long, Integer> javaForwarder = x -> popcountInterface.call(x);
        cases.add(new Case("S09", "Java UDF1 lambda forwarding through the interface, behind Spark's adapter", Intent.SAFE,
                adapt.apply(javaForwarder), new DataType[] { L }, I, false, false, null, null));
        UDF1<Long, Integer> javaLambda = x -> Long.bitCount(x) + 1;
        cases.add(new Case("S10", "plain Java UDF1 lambda behind Spark's adapter", Intent.SAFE,
                adapt.apply(javaLambda), new DataType[] { L }, I, false, false, null, null));
        scala.runtime.java8.JFunction1$mcIJ$sp primitive = (scala.runtime.java8.JFunction1$mcIJ$sp & Serializable) x -> popcount.call(x);
        cases.add(new Case("S11", "Scala-shaped primitive forwarder, non-nullable input", Intent.SAFE, primitive, new DataType[] { L }, I, false, false, null, null));
        UDF1<Long, Integer> boundReference = popcount::call;
        cases.add(new Case("S34", "bound method reference popcount::call behind Spark's adapter", Intent.SAFE, adapt.apply(boundReference), new DataType[] { L }, I,
                false, false, null, null));
        cases.add(new Case("U26", "typed wrapper that adds one to the forwarded result", Intent.UNSAFE,
                (scala.Function1<Long, Integer> & Serializable) x -> popcount.call(x) + 1, new DataType[] { L }, I, false, false));
        cases.add(new Case("U27", "forwarder with a side effect", Intent.UNSAFE, (scala.Function1<Long, Integer> & Serializable) x -> {
            DispatchResolutionTest.sideEffect();
            return popcount.call(x);
        }, new DataType[] { L }, I, false, false));
        cases.add(new Case("U28", "Scala-shaped primitive forwarder over a nullable input (unboxing null differs)", Intent.UNSAFE, primitive,
                new DataType[] { L }, I, true, false));
        cases.add(new Case("U29", "child overriding an instance helper the inherited call uses (dynamic dispatch in the body)", Intent.UNSAFE,
                adapt.apply(new DispatchResolutionTest.TemplateChild()), new DataType[] { L }, L, false, false));
        cases.add(new Case("G01", "implementation is an interface default method", Intent.GAP,
                adapt.apply(new DispatchResolutionTest.UsesDefault()), new DataType[] { L }, I, false, false));
        cases.add(new Case("G02", "typed wrapper computing on the forwarded result (x -> inner(x) * 2): compilable as a composition", Intent.GAP,
                (scala.Function1<Long, Integer> & Serializable) x -> popcount.call(x) * 2, new DataType[] { L }, I, false, false));
        int captured = 5;
        UDF1<Integer, Integer> capturesPrimitive = x -> x + captured;
        cases.add(new Case("G03", "Java UDF lambda capturing an effectively-final int", Intent.GAP, adapt.apply(capturesPrimitive), new DataType[] { I }, I, false, false));

        // ---- constructor fields and serialization
        cases.add(new Case("S12", "constructor-configured final int behind Spark's adapter", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.Offset(41)), new DataType[] { I }, I, false, false, "folds 41", c -> c.contains("41")));
        cases.add(new Case("S13", "final long, boolean and int constants together", Intent.SAFE,
                new ExecutorInvariantFieldsTest.Masked(0xFFFF_0000L, true, 16), new DataType[] { L }, L, false, false, null, null));
        cases.add(new Case("S14", "final fields declared in a serializable parent", Intent.SAFE,
                new ExecutorInvariantFieldsTest.DerivedOffset(4, -3), new DataType[] { I }, I, false, false, null, null));
        cases.add(new Case("S15", "inherited call reading the parent's final fields, child adds an overload", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.ChildOfFields()), new DataType[] { I, I }, I, false, false, "folds 3 and 5", c -> c.contains("3") && c.contains("5")));
        cases.add(new Case("S16", "readResolve returning this, call reads no field", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.PureResolveNoFields()), new DataType[] { I }, I, false, false, null, null));
        cases.add(new Case("S17", "final transient field: executor value is 0", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.TransientFactor()), new DataType[] { I }, I, false, false, "does not fold 9", c -> !c.contains("9")));
        cases.add(new Case("U30", "Codex B: readResolve rebuilds the field from a system property", Intent.UNSAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.ResolvedFactor(9)), new DataType[] { I }, I, false, false));
        cases.add(new Case("U31", "readObject overwrites the final field from a system property", Intent.UNSAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.ReadObjectFactor(9)), new DataType[] { I }, I, false, false));
        cases.add(new Case("S32", "deterministic writeObject streams factor*100 (executor sees 900)", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.WrittenFactor(9)), new DataType[] { I }, I, false, false, "folds 900", c -> c.contains("900")));
        cases.add(new Case("S33", "readObject that only calls defaultReadObject", Intent.SAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.DefaultReadObject(12)), new DataType[] { I }, I, false, false, "folds 12", c -> c.contains("12")));
        cases.add(new Case("U33", "field of a non-serializable parent, set by its constructor on the executor", Intent.UNSAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.EnvironmentDerived()), new DataType[] { I }, I, false, false));
        cases.add(new Case("U34", "readResolve picks the class from a system property", Intent.UNSAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.ClassSwitch()), new DataType[] { I }, I, false, false));
        cases.add(new Case("U35", "final reference to a mutable array", Intent.UNSAFE,
                adapt.apply(new ExecutorInvariantFieldsTest.MutableReference()), new DataType[] { I }, I, false, false));

        // ---- conditional divisors
        cases.add(new Case("S18", "Codex C: floorMod(x, x == 7 ? y : 3)", Intent.SAFE, new ConditionalDivisorTest.CondFloorMod(), new DataType[] { I, I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S19", "Codex C: floorDiv(x, x == 7 ? y : 3)", Intent.SAFE, new ConditionalDivisorTest.CondFloorDiv(), new DataType[] { I, I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S20", "long floorMod, constant on one branch, parameter on the other", Intent.SAFE, new ConditionalDivisorTest.CondFloorModLong(),
                new DataType[] { L, L }, L, false, false, "zero test", guarded));
        cases.add(new Case("S21", "zero on one branch of the divisor", Intent.SAFE, new ConditionalDivisorTest.ZeroOnOneBranch(), new DataType[] { I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S22", "divisor local assigned on three paths", Intent.SAFE, new ConditionalDivisorTest.LocalAssignedOnPaths(), new DataType[] { I, I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S23", "nested conditional divisor", Intent.SAFE, new ConditionalDivisorTest.NestedConditional(), new DataType[] { I, I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S24", "conditional divisor passed to a floorMod helper", Intent.SAFE, new ConditionalDivisorTest.ThroughHelper(), new DataType[] { I, I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S25", "divisor decremented around a loop back edge", Intent.SAFE, new ConditionalDivisorTest.LoopChangesDivisor(), new DataType[] { I }, I, false, false,
                "zero test", guarded));
        cases.add(new Case("S26", "non-zero constants on both branches", Intent.SAFE, new ConditionalDivisorTest.NonZeroBothBranches(), new DataType[] { I }, I, false, false,
                "precision: no zero test needed", unguarded));
        cases.add(new Case("S27", "divisor y | 1 (never zero)", Intent.SAFE, new ConditionalDivisorTest.OrOne(), new DataType[] { I, I }, I, false, false,
                "precision: no zero test needed", unguarded));
        cases.add(new Case("S28", "division and remainder by constant -1 (MIN_VALUE wraps)", Intent.SAFE, new ConditionalDivisorTest.ByMinusOne(), new DataType[] { I }, I, false, false,
                "java_idiv", c -> c.contains("java_idiv")));
        cases.add(new Case("S29", "long floorDiv by -1", Intent.SAFE, new ConditionalDivisorTest.ByMinusOneLong(), new DataType[] { L }, L, false, false, null, null));
        cases.add(new Case("S30", "LIST fold dividing by a conditional element divisor", Intent.SAFE, new ConditionalDivisorTest.ListCondDiv(), new DataType[] { listL }, L, false, false,
                "zero test", guarded));
        cases.add(new Case("G04", "LIST fold holding the element in a temporary read inside a conditional (pre-existing recognizer limit)", Intent.GAP,
                new ConditionalDivisorTest.ListCondDivTemporary(), new DataType[] { listL }, L, false, false));
        cases.add(new Case("S31", "LIST fold dividing by conditional non-zero constants", Intent.SAFE, new ConditionalDivisorTest.ListSafeDiv(), new DataType[] { listL }, L, false,
                false, "precision: no zero test needed", c -> !zeroTest(c)));
    }
}
