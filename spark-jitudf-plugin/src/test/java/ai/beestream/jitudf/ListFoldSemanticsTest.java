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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.util.Arrays;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Codex findings 2 and 3: a LIST fold becomes a child transform plus a segmented sum only when the
 * Java loop provably computes that sum, and the executor keeps the Java result for empty and null
 * rows.
 *
 * <p>
 * The positive cases execute the element function the split actually extracted, apply it to every
 * element of each row and sum -- which is what {@link CudfListReductionUdf} does on the device --
 * and compare with the user's UDF called on the same row. Nothing here re-implements the UDF body
 * by hand.
 * </p>
 */
class ListFoldSemanticsTest {

    static final DataType LIST_L = DataTypes.createArrayType(DataTypes.LongType, false);
    static final DataType LIST_I = DataTypes.createArrayType(DataTypes.IntegerType, false);
    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    static final long[][] ROWS = {
            {}, { 1 }, { 1, 3, 7 }, { -1, -2, -3 }, { Long.MAX_VALUE, 1 }, { Long.MIN_VALUE, -1 },
            { Integer.MAX_VALUE, Integer.MAX_VALUE, 5 }, { 0, 0, 0, 0 }, { 0xFFFF_FFFFL, 0x5555_5555_5555_5555L, 255 } };

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ------------------------------------------------------------------------------ positive

    public static final class ElementSquares implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] * arr[i] - 3 * arr[i];
            }
            return acc;
        }
    }

    public static final class TotalFirst implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc = (arr[i] ^ 0x5A) + acc;
            }
            return acc;
        }
    }

    public static final class BoundInALocal implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int n = arr.length;
            int acc = 0;
            for (int i = 0; i < n; i++) {
                acc += Long.bitCount(arr[i]);
            }
            return acc;
        }
    }

    public static final class WithTemporary implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                long v = arr[i];
                long w = v >>> 3;
                acc += w + (v & 7);
            }
            return acc;
        }
    }

    public static final class WithBranch implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] > 0 ? arr[i] : -arr[i];
            }
            return acc;
        }
    }

    public static final class IntOverflow implements UDF1<int[], Integer> {
        @Override
        public Integer call(int[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] * 3;
            }
            return acc;
        }
    }

    @Test
    void canonicalFoldsMatchTheJavaLoopOnEveryRow() throws Exception {
        check(new CoverageCorpus.ActiveDays(), LIST_L, I);
        check(new CoverageCorpus.ListSum(), LIST_L, L);
        check(new ElementSquares(), LIST_L, L);
        check(new TotalFirst(), LIST_L, L);
        check(new BoundInALocal(), LIST_L, I);
        check(new WithTemporary(), LIST_L, L);
        check(new WithBranch(), LIST_L, L);
        check(new CoverageCorpus.ListSeqSum(), LIST_L, L);
    }

    @Test
    void integerFoldsWrapLikeJava() throws Exception {
        Method element = LoweredOracle.elementFunction(new IntOverflow(), LIST_I, I);
        int[][] rows = { {}, { Integer.MAX_VALUE, Integer.MAX_VALUE }, { Integer.MIN_VALUE, -1, 7 }, { 1 } };
        for (int[] row : rows) {
            long[] widened = Arrays.stream(row).asLongStream().toArray();
            assertEquals(new IntOverflow().call(row), LoweredOracle.reduceRow(element, PrimitiveKind.INT, widened), Arrays.toString(row));
        }
    }

    private static void check(Object udf, DataType listType, DataType output) throws Exception {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, new DataType[] { listType }, output, false, udf.getClass().getSimpleName());
        assertTrue(o.accepted() && o.udf() instanceof CudfListReductionUdf, () -> udf.getClass().getSimpleName() + ": " + o.decision());
        Method element = LoweredOracle.elementFunction(udf, listType, output);
        PrimitiveKind result = PrimitiveKind.ofSparkType(output);
        for (long[] row : ROWS) {
            Object expected = udf instanceof CoverageCorpus.ListSeqSum seq ? seq.call(scala.jdk.javaapi.CollectionConverters.asScala(Arrays.stream(row).boxed().toList()).toSeq())
                    : LoweredOracle.reference(udf, (Object) row);
            assertEquals(expected, LoweredOracle.reduceRow(element, result, row), () -> udf.getClass().getSimpleName() + " on " + Arrays.toString(row));
        }
    }

    // ------------------------------------------------------------------------------ negative

    public static final class Skips implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                if (arr[i] < 0) {
                    continue;
                }
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class ConditionalUpdate implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                if (arr[i] > 0) {
                    acc += arr[i];
                }
            }
            return acc;
        }
    }

    public static final class EarlyReturn implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                if (arr[i] == 0) {
                    return acc;
                }
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class Breaks implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                if (acc > 100) {
                    break;
                }
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class Reversed implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = arr.length - 1; i >= 0; i--) {
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class StartsAtOne implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 1; i < arr.length; i++) {
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class ShortBound implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length - 1; i++) {
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class TwoWrites implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i];
                acc += 1;
            }
            return acc;
        }
    }

    public static final class OrderDependent implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc = acc * 31 + arr[i];
            }
            return acc;
        }
    }

    public static final class NonZeroStart implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 7;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i];
            }
            return acc;
        }
    }

    public static final class AfterLoop implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i];
            }
            return acc * 2;
        }
    }

    public static final class Positional implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] * i;
            }
            return acc;
        }
    }

    public static final class NeighbourRead implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i + 1 < arr.length; i++) {
                acc += arr[i + 1] - arr[i];
            }
            return acc;
        }
    }

    @Test
    void loopsThatAreNotAFullSumAreDeclinedForTheRightReason() {
        declined(new CoverageCorpus.StrideTwo(), I, "advances by 2");
        declined(new Skips(), L, "");
        declined(new ConditionalUpdate(), L, "skip the accumulation");
        declined(new EarlyReturn(), L, "");
        declined(new Breaks(), L, "");
        declined(new Reversed(), L, "");
        declined(new StartsAtOne(), L, "before the loop");
        declined(new ShortBound(), L, "");
        declined(new TwoWrites(), L, "");
        declined(new OrderDependent(), L, "");
        declined(new NonZeroStart(), L, "does not start at zero");
        declined(new AfterLoop(), L, "after the loop");
        declined(new Positional(), L, "loop index itself");
        declined(new NeighbourRead(), L, "");
        declined(new IncrementBeforeAccess(), L, "not provably inside its row");
    }

    /** The index moves between the bounds check and the read: the read can be one past the row. */
    public static final class IncrementBeforeAccess implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                if (acc > 3) {
                    i++;
                }
                acc += arr[i];
            }
            return acc;
        }
    }

    private static void declined(Object udf, DataType output, String reason) {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, new DataType[] { LIST_L }, output, false, udf.getClass().getSimpleName());
        assertFalse(o.accepted(), () -> udf.getClass().getSimpleName() + " must not become a segmented sum: " + o.decision());
        assertTrue(o.decision().contains(reason), () -> udf.getClass().getSimpleName() + " declined for an unexpected reason: " + o.decision());
    }

    // ---------------------------------------------------------------------- floating point and errors

    public static final class DoubleSum implements UDF1<double[], Double> {
        @Override
        public Double call(double[] arr) {
            double acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i];
            }
            return acc;
        }
    }

    @Test
    void aFloatingPointFoldNeedsBothRapidsSwitches() {
        DataType listD = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ProviderHarness.Outcome off = ProviderHarness.cudf(new DoubleSum(), new DataType[] { listD }, DataTypes.DoubleType, false, "dsumOff", false);
        assertFalse(off.accepted(), off::decision);
        ProviderHarness.Outcome on = ProviderHarness.cudf(new DoubleSum(), new DataType[] { listD }, DataTypes.DoubleType, false, "dsumOn", true);
        assertTrue(on.accepted(), on::decision);
        String reordered = ProviderHarness.withSettings(true, () -> {
            org.apache.spark.sql.internal.SQLConf.get().setConfString(SparkSemantics.VARIABLE_FLOAT_AGG, "false");
            try {
                return SparkSemantics.refusalForReorderedFloatSum();
            } finally {
                org.apache.spark.sql.internal.SQLConf.get().unsetConf(SparkSemantics.VARIABLE_FLOAT_AGG);
            }
        });
        assertTrue(reordered != null && reordered.contains(SparkSemantics.VARIABLE_FLOAT_AGG), String.valueOf(reordered));
    }

    public static final class DividesByElement implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += 1000 / arr[i];
            }
            return acc;
        }
    }

    @Test
    void anElementFunctionThatCanDivideByZeroCarriesTheErrorChannel() {
        ProviderHarness.Outcome o = ProviderHarness.cudf(new DividesByElement(), new DataType[] { LIST_L }, L, false, "divElem");
        assertTrue(o.accepted(), o::decision);
        CudfListReductionUdf udf = (CudfListReductionUdf) o.udf();
        assertTrue(udf.reportsArithmeticErrors(), "a zero element must fail the query, as Java's ArithmeticException does");
        assertTrue(o.cuda().contains("__device__ int GENERIC_TRANSFORM_OP") && o.cuda().contains("return 2"), o::cuda);
    }

    @Test
    void aNullableListColumnStaysEligibleAndANullRowFailsLikeJava() throws Exception {
        ProviderHarness.Outcome o = ProviderHarness.cudf(new CoverageCorpus.ActiveDays(), new DataType[] { LIST_L }, I, true, "nullableList");
        assertTrue(o.accepted(), o::decision);
        // The Java UDF dereferences the list first: a null row is NullPointerException, a failed task.
        assertEquals("threw:NullPointerException", LoweredOracle.reference(new CoverageCorpus.ActiveDays(), (Object) null));
        Method element = LoweredOracle.elementFunction(new CoverageCorpus.ActiveDays(), LIST_L, I);
        assertEquals("threw:NullListException", LoweredOracle.reduceRow(element, PrimitiveKind.INT, null));
    }
}
