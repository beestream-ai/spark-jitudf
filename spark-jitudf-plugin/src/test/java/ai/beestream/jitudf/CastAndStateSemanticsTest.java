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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Codex findings 4 and 5: a cast is erased only when it cannot fail, and a field is folded to the
 * value the executor's copy of the UDF holds after Spark ships it.
 */
class CastAndStateSemanticsTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;
    static final DataType D = DataTypes.DoubleType;
    static final DataType B = DataTypes.BooleanType;

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ===================================================================================== casts

    public static final class IntegerCast implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return ((Integer) (Object) x) * 2;
        }
    }

    public static final class LongCast implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            return ((Long) (Object) x) + 1;
        }
    }

    public static final class BooleanCast implements UDF1<Boolean, Integer> {
        @Override
        public Integer call(Boolean x) {
            return ((Boolean) (Object) x) ? 21 : 42;
        }
    }

    public static final class CastOfReturnedBox implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return (Integer) (Object) Integer.valueOf(x + 1);
        }
    }

    public static final class CastOfBranchSameType implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return (Integer) (a > 0 ? (Object) a : (Object) b);
        }
    }

    public static final class IntegerToLong implements UDF1<Integer, Long> {
        @Override
        public Long call(Integer x) {
            return (Long) (Object) x;
        }
    }

    public static final class LongToDouble implements UDF1<Long, Double> {
        @Override
        public Double call(Long x) {
            return (Double) (Object) x;
        }
    }

    public static final class ReturnedBoxWrongType implements UDF1<Integer, Long> {
        @Override
        public Long call(Integer x) {
            return (Long) (Object) Integer.valueOf(x);
        }
    }

    /** Valid or not depending on which branch runs: Java throws only when a <= 0. */
    public static final class CastOfBranchMixedType implements UDF2<Integer, Long, Integer> {
        @Override
        public Integer call(Integer a, Long b) {
            return (Integer) (a > 0 ? (Object) a : (Object) b);
        }
    }

    @Test
    void provablyValidCastsStayEligibleAndCompute() throws Exception {
        accepted(new IntegerCast(), List.of(I), I, new Object[][] { { 0 }, { -7 }, { Integer.MAX_VALUE } });
        accepted(new LongCast(), List.of(L), L, new Object[][] { { 0L }, { Long.MAX_VALUE } });
        accepted(new BooleanCast(), List.of(B), I, new Object[][] { { true }, { false } });
        accepted(new CastOfReturnedBox(), List.of(I), I, new Object[][] { { 1 }, { -1 } });
        accepted(new CastOfBranchSameType(), List.of(I, I), I, new Object[][] { { 1, 9 }, { -1, 9 } });
        accepted(new CoverageCorpus.ValidCasts(), List.of(I, L), L, new Object[][] { { 3, 4L } });
    }

    public static final class InstanceofDouble implements UDF2<Double, Integer, Integer> {
        @Override
        public Integer call(Double d, Integer i) {
            return (d instanceof Double ? 10 : 0) + i;
        }
    }

    @Test
    void aProvenInstanceofOnAWideValueLowersToValidCode() throws Exception {
        accepted(new InstanceofDouble(), List.of(D, I), I, new Object[][] { { 1.5, 2 }, { -0.0, -7 } });
    }

    @Test
    void castsThatCanFailAreDeclinedAndJavaReallyThrows() {
        refused(new CoverageCorpus.BadCast(), new DataType[] { I, I }, I, "throws ClassCastException", new Object[] { 1, 2 });
        refused(new IntegerToLong(), new DataType[] { I }, L, "throws ClassCastException", new Object[] { 5 });
        refused(new LongToDouble(), new DataType[] { L }, D, "throws ClassCastException", new Object[] { 5L });
        refused(new ReturnedBoxWrongType(), new DataType[] { I }, L, "throws ClassCastException", new Object[] { 5 });
        // Depends on the row: Java returns for a > 0 and throws otherwise. One erased cast cannot
        // reproduce that, so it is declined.
        refused(new CastOfBranchMixedType(), new DataType[] { I, L }, I, "throws ClassCastException", new Object[] { -1, 5L });
        assertEquals(3, LoweredOracle.reference(new CastOfBranchMixedType(), 3, 5L));
    }

    public static final class NullableCast implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            Integer y = (Integer) (Object) x;
            return y == null ? -1 : y + 1;
        }
    }

    public static final class NullableBadCast implements UDF1<Integer, Long> {
        @Override
        public Long call(Integer x) {
            return x == null ? -1L : (Long) (Object) x;
        }
    }

    @Test
    void castsOverANullableColumnFollowTheSameRule() {
        ProviderHarness.Outcome bad = ProviderHarness.cudf(new NullableBadCast(), new DataType[] { I }, L, true, "nullableBad");
        assertFalse(bad.accepted(), bad::decision);
        assertTrue(bad.decision().contains("ClassCastException"), bad::decision);
    }

    // ============================================================================ serialized state

    public static final class TwoConstants implements UDF1<Integer, Integer> {
        private final int weight;
        private final long offset;
        private final boolean negate;

        public TwoConstants(int weight, long offset, boolean negate) {
            this.weight = weight;
            this.offset = offset;
            this.negate = negate;
        }

        @Override
        public Integer call(Integer x) {
            int v = (int) (x * weight + offset);
            return negate ? -v : v;
        }
    }

    @Test
    void ordinaryFinalFieldsAreFoldedAndCompute() throws Exception {
        accepted(new TwoConstants(3, 11, false), List.of(I), I, new Object[][] { { 0 }, { 5 }, { -5 } });
        accepted(new TwoConstants(7, -2, true), List.of(I), I, new Object[][] { { 0 }, { 5 } });
        accepted(new CoverageCorpus.ConfiguredDouble(2.5), List.of(D), D, new Object[][] { { 1.0 }, { -0.0 } });
    }

    @Test
    void aTransientFieldIsFoldedToTheValueTheExecutorSees() throws Exception {
        CoverageCorpus.TransientField udf = new CoverageCorpus.TransientField();
        assertEquals(10, udf.call(1), "on the driver the field is 9");
        assertEquals(1, LoweredOracle.reference(udf, 1), "after Java serialization it is 0");
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, new DataType[] { I }, I, false, "transient");
        assertTrue(o.accepted(), o::decision);
        assertFalse(o.cuda().contains("9"), () -> "the driver's value must not be compiled in:\n" + o.cuda());
        accepted(udf, List.of(I), I, new Object[][] { { 1 }, { -4 } });
    }

    /** readResolve hands the executor a different instance, with a different value. */
    public static final class Resolved implements UDF1<Integer, Integer> {
        private final int k;

        public Resolved(int k) {
            this.k = k;
        }

        private Object readResolve() {
            return new Resolved(k * 10);
        }

        @Override
        public Integer call(Integer x) {
            return x + k;
        }
    }

    @Test
    void readResolveDecidesTheFoldedValue() throws Exception {
        Resolved udf = new Resolved(3);
        assertEquals(31, LoweredOracle.reference(udf, 1));
        accepted(udf, List.of(I), I, new Object[][] { { 1 }, { 100 } });
    }

    /** writeReplace ships a different class altogether. */
    public static final class Replaced implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return x + 1;
        }

        private Object writeReplace() {
            return new Other();
        }
    }

    public static final class Other implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return x - 1;
        }
    }

    @Test
    void aUdfReplacedDuringSerializationIsDeclined() {
        assertEquals(0, LoweredOracle.reference(new Replaced(), 1), "executors run Other.call");
        ProviderHarness.Outcome o = ProviderHarness.cudf(new Replaced(), new DataType[] { I }, I, false, "replaced");
        assertFalse(o.accepted(), o::decision);
        assertTrue(o.decision().contains("replaced during serialization"), o::decision);
    }

    /** Every deserialization gives a different value: nothing about it is a constant. */
    public static final class Unstable implements UDF1<Integer, Integer> {
        static final java.util.concurrent.atomic.AtomicInteger COUNTER = new java.util.concurrent.atomic.AtomicInteger();
        private final int k;

        public Unstable(int k) {
            this.k = k;
        }

        private Object readResolve() {
            return new Unstable(COUNTER.incrementAndGet());
        }

        @Override
        public Integer call(Integer x) {
            return x + k;
        }
    }

    @Test
    void aFieldWhoseDeserializedValueVariesIsNotFolded() {
        ProviderHarness.Outcome o = ProviderHarness.cudf(new Unstable(1), new DataType[] { I }, I, false, "unstable");
        assertFalse(o.accepted(), o::decision);
        assertTrue(o.decision().contains("instance field"), o::decision);
    }

    public static final class MutableReference implements UDF1<Integer, Integer> {
        private final int[] table = { 1, 2, 3 };

        @Override
        public Integer call(Integer x) {
            return x + table.length;
        }
    }

    public static final class NotShippable implements UDF1<Integer, Integer> {
        private final Object lock = new Object();
        private final int k = 4;

        @Override
        public Integer call(Integer x) {
            return x + k;
        }
    }

    @Test
    void stateThatIsNotAShippedConstantIsDeclined() {
        ProviderHarness.Outcome mutable = ProviderHarness.cudf(new MutableReference(), new DataType[] { I }, I, false, "mutableRef");
        assertFalse(mutable.accepted(), mutable::decision);
        ProviderHarness.Outcome unshippable = ProviderHarness.cudf(new NotShippable(), new DataType[] { I }, I, false, "notShippable");
        assertFalse(unshippable.accepted(), unshippable::decision);
        assertTrue(unshippable.decision().contains("serialization"), unshippable::decision);
    }

    @Test
    void twoInstancesOfOneClassNeverShareACompilation() {
        ProviderHarness.Outcome three = ProviderHarness.cudf(new CoverageCorpus.Configured(3, 0), new DataType[] { I }, I, false, "configured");
        ProviderHarness.Outcome five = ProviderHarness.cudf(new CoverageCorpus.Configured(5, 0), new DataType[] { I }, I, false, "configured");
        ProviderHarness.Outcome threeAgain = ProviderHarness.cudf(new CoverageCorpus.Configured(3, 0), new DataType[] { I }, I, false, "configured");
        assertTrue(three.accepted() && five.accepted() && threeAgain.accepted());
        assertNotEquals(three.cuda(), five.cuda(), "each instance's constant must be compiled in");
        assertFalse(five.decision().contains("cached"), five::decision);
        assertEquals(three.cuda(), threeAgain.cuda());
        assertTrue(threeAgain.decision().contains("cached"), "an equal instance may reuse: " + threeAgain.decision());
    }

    // ===================================================================================== helpers

    static void accepted(Object udf, List<DataType> inputs, DataType output, Object[][] samples) throws Exception {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, inputs.toArray(new DataType[0]), output, false, udf.getClass().getSimpleName(),
                output.equals(D) || inputs.contains(D));
        assertTrue(o.accepted(), () -> udf.getClass().getSimpleName() + ": " + o.decision());
        Method body = LoweredOracle.scalarBody(udf, inputs, output);
        for (Object[] sample : samples) {
            Object[] primitive = sample.clone();
            for (int i = 0; i < primitive.length; i++) {
                if (primitive[i] instanceof Boolean b) {
                    primitive[i] = b;
                }
            }
            assertEquals(LoweredOracle.reference(udf, sample), LoweredOracle.run(body, primitive),
                    () -> udf.getClass().getSimpleName() + " on " + java.util.Arrays.toString(sample));
        }
    }

    static void refused(Object udf, DataType[] inputs, DataType output, String reason, Object[] javaThrowsOn) {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, inputs, output, false, udf.getClass().getSimpleName());
        assertFalse(o.accepted(), () -> udf.getClass().getSimpleName() + " must be declined: " + o.decision());
        assertTrue(o.decision().contains(reason), () -> udf.getClass().getSimpleName() + ": " + o.decision());
        assertEquals("threw:ClassCastException", LoweredOracle.reference(udf, javaThrowsOn), "Java itself must throw here");
    }

    @SuppressWarnings("unused")
    private static final Serializable UNUSED = 0;
}
