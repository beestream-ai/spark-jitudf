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
import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Codex's second review, finding C: whether a division needs its zero test is decided by what the
 * divisor can be on <em>every</em> path to it, joined at each merge point -- not by the instruction
 * in front of it. Uncertainty keeps the UDF on the GPU with the test compiled in; only provable
 * safety removes it.
 *
 * <p>
 * For each UDF the test checks three things: the provider accepts it; the CUDA has the zero test
 * ({@code __device__ int ... return 2}) exactly when some path can divide by zero; and the lowered
 * body equals Java on safe inputs and raises where Java raises.
 * </p>
 */
class ConditionalDivisorTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;
    static final DataType LIST_L = DataTypes.createArrayType(DataTypes.LongType, false);

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ============================================================================ may divide by zero

    public static final class CondFloorMod implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorMod(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondFloorDiv implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorDiv(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondFloorModLong implements UDF2<Long, Long, Long> {
        @Override
        public Long call(Long x, Long y) {
            return Math.floorMod(x, x > 0 ? 1000L : y.longValue());
        }
    }

    public static final class ZeroOnOneBranch implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return 100 / (x > 0 ? 0 : 3);
        }
    }

    public static final class LocalAssignedOnPaths implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            int d;
            if (x > 10) {
                d = 5;
            } else if (x > 0) {
                d = y;
            } else {
                d = 7;
            }
            return Math.floorDiv(x, d);
        }
    }

    public static final class NestedConditional implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return x % (x > 0 ? (x > 100 ? 9 : y) : 11);
        }
    }

    public static final class ThroughHelper implements UDF2<Integer, Integer, Integer> {
        static int bucket(int a, int b) {
            return Math.floorMod(a, b);
        }

        @Override
        public Integer call(Integer x, Integer y) {
            return bucket(x, x == 7 ? y : 3);
        }
    }

    public static final class LoopChangesDivisor implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            int d = 3;
            int acc = 0;
            for (int i = 0; i < (x & 7); i++) {
                acc += 100 / d;
                d = d - 1;
            }
            return acc;
        }
    }

    public static final class VariableDivisorLong implements UDF2<Long, Long, Long> {
        @Override
        public Long call(Long x, Long y) {
            return x / y;
        }
    }

    // ============================================================================ provably non-zero

    public static final class NonZeroBothBranches implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return Math.floorMod(x, x > 0 ? 7 : -3);
        }
    }

    public static final class SameConstantBothPaths implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            long d;
            if (x > 0) {
                d = 1000L;
            } else {
                d = 1000L;
            }
            return Math.floorDiv(x, d);
        }
    }

    public static final class OrOne implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return x / (y | 1);
        }
    }

    public static final class ByMinusOne implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return x / -1 + x % -1;
        }
    }

    public static final class ByMinusOneLong implements UDF1<Long, Long> {
        @Override
        public Long call(Long x) {
            return Math.floorDiv(x, -1L);
        }
    }

    public static final class KnownConstant implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return Math.floorMod(x, 1000) + x / 7;
        }
    }

    /** A LIST fold whose element arithmetic divides conditionally. */
    public static final class ListCondDiv implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += 1000L / (arr[i] > 5 ? arr[i] : 3L);
            }
            return acc;
        }
    }

    public static final class ListSafeDiv implements UDF1<long[], Long> {
        @Override
        public Long call(long[] arr) {
            long acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += arr[i] / (arr[i] > 5 ? 7L : 3L);
            }
            return acc;
        }
    }

    // ============================================================================ tests

    @Test
    void codexCounterexampleKeepsTheGpuAndRaisesForZero() throws Exception {
        for (Object udf : new Object[] { new CondFloorMod(), new CondFloorDiv() }) {
            String cuda = guarded(udf, List.of(I, I), I);
            assertTrue(hasZeroTest(cuda), cuda);
            Method body = LoweredOracle.scalarBody(udf, List.of(I, I), I);
            assertEquals("threw:ArithmeticException", LoweredOracle.reference(udf, 7, 0));
            assertEquals("threw:ArithmeticException", LoweredOracle.run(body, 7, 0));
            for (int[] xy : new int[][] { { 7, 4 }, { 7, -4 }, { -8, 0 }, { 8, 0 }, { Integer.MIN_VALUE, 5 }, { 7, -1 } }) {
                assertEquals(LoweredOracle.reference(udf, xy[0], xy[1]), LoweredOracle.run(body, xy[0], xy[1]));
            }
        }
    }

    @Test
    void anyPathThatCanReachZeroKeepsTheTest() throws Exception {
        Object[][] cases = {
                { new CondFloorModLong(), List.of(L, L), L, new Object[][] { { 5L, 0L }, { -5L, 3L }, { -5L, 0L } } },
                { new ZeroOnOneBranch(), List.of(I), I, new Object[][] { { 1 }, { -1 }, { 0 } } },
                { new LocalAssignedOnPaths(), List.of(I, I), I, new Object[][] { { 20, 0 }, { 5, 0 }, { 5, 2 }, { -9, 0 } } },
                { new NestedConditional(), List.of(I, I), I, new Object[][] { { 200, 0 }, { 50, 0 }, { 50, 7 }, { -5, 0 } } },
                { new ThroughHelper(), List.of(I, I), I, new Object[][] { { 7, 0 }, { 7, 2 }, { 8, 0 } } },
                { new LoopChangesDivisor(), List.of(I), I, new Object[][] { { 0 }, { 2 }, { 3 }, { 4 }, { 7 } } },
                { new VariableDivisorLong(), List.of(L, L), L, new Object[][] { { 7L, 0L }, { Long.MIN_VALUE, -1L }, { -9L, 2L } } } };
        for (Object[] c : cases) {
            Object udf = c[0];
            @SuppressWarnings("unchecked")
            List<DataType> in = (List<DataType>) c[1];
            String cuda = guarded(udf, in, (DataType) c[2]);
            assertTrue(hasZeroTest(cuda), () -> udf.getClass().getSimpleName() + "\n" + cuda);
            Method body = LoweredOracle.scalarBody(udf, in, (DataType) c[2]);
            for (Object[] args : (Object[][]) c[3]) {
                assertEquals(LoweredOracle.reference(udf, args), LoweredOracle.run(body, args), () -> udf.getClass().getSimpleName() + java.util.Arrays.toString(args));
            }
        }
        // The zero paths really do raise in Java: these are the rows the GPU must fail on.
        assertEquals("threw:ArithmeticException", LoweredOracle.reference(new ZeroOnOneBranch(), 1));
        assertEquals("threw:ArithmeticException", LoweredOracle.reference(new LoopChangesDivisor(), 4));
    }

    @Test
    void provablyNonZeroDivisorsNeedNoTest() throws Exception {
        Object[][] cases = {
                { new NonZeroBothBranches(), List.of(I), I, new Object[][] { { 5 }, { -5 }, { Integer.MIN_VALUE } } },
                { new SameConstantBothPaths(), List.of(L), L, new Object[][] { { 12345L }, { -12345L } } },
                { new OrOne(), List.of(I, I), I, new Object[][] { { 9, 0 }, { 9, -2 }, { Integer.MIN_VALUE, -1 } } },
                { new ByMinusOne(), List.of(I), I, new Object[][] { { 5 }, { Integer.MIN_VALUE }, { Integer.MAX_VALUE } } },
                { new ByMinusOneLong(), List.of(L), L, new Object[][] { { 5L }, { Long.MIN_VALUE } } },
                { new KnownConstant(), List.of(I), I, new Object[][] { { -1 }, { 123456 } } } };
        for (Object[] c : cases) {
            Object udf = c[0];
            @SuppressWarnings("unchecked")
            List<DataType> in = (List<DataType>) c[1];
            ProviderHarness.Outcome o = ProviderHarness.cudf(udf, in.toArray(new DataType[0]), (DataType) c[2], false, udf.getClass().getSimpleName());
            assertTrue(o.accepted(), o::decision);
            assertTrue(o.cuda().contains("__device__ void"), () -> udf.getClass().getSimpleName() + " needs no error channel\n" + o.cuda());
            Method body = LoweredOracle.scalarBody(udf, in, (DataType) c[2]);
            for (Object[] args : (Object[][]) c[3]) {
                assertEquals(LoweredOracle.reference(udf, args), LoweredOracle.run(body, args), () -> udf.getClass().getSimpleName() + java.util.Arrays.toString(args));
            }
        }
        // Java wraps MIN_VALUE / -1; TC's java_idiv does the same, which is why -1 is no longer refused.
        assertEquals(Integer.MIN_VALUE, LoweredOracle.reference(new ByMinusOne(), Integer.MIN_VALUE));
        assertTrue(ProviderHarness.cudf(new ByMinusOne(), new DataType[] { I }, I, false, "m1").cuda().contains("java_idiv"));
    }

    @Test
    void listFoldsCarryTheSameRule() throws Exception {
        ProviderHarness.Outcome cond = ProviderHarness.cudf(new ListCondDiv(), new DataType[] { LIST_L }, L, false, "list_cond");
        assertTrue(cond.accepted(), cond::decision);
        assertTrue(hasZeroTest(cond.cuda()), cond::cuda);
        ProviderHarness.Outcome safe = ProviderHarness.cudf(new ListSafeDiv(), new DataType[] { LIST_L }, L, false, "list_safe");
        assertTrue(safe.accepted(), safe::decision);
        assertFalse(safe.cuda().contains("return 2"), safe::cuda);
        Method element = LoweredOracle.elementFunction(new ListCondDiv(), LIST_L, L);
        for (long[] row : new long[][] { {}, { 6, 9, 100 }, { 1, 2, 3 }, { -4, 0, 7 } }) {
            assertEquals(LoweredOracle.reference(new ListCondDiv(), scalaSeqOrArray(row)), LoweredOracle.reduceRow(element, PrimitiveKind.LONG, row));
        }
    }

    @Test
    void theRuntimeBackendStillRequiresAnExactSafeConstant() {
        for (Object udf : new Object[] { new CondFloorMod(), new OrOne(), new ByMinusOne(), new NonZeroBothBranches() }) {
            ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow();
            List<DataType> in = udf instanceof UDF2 ? List.of(I, I) : List.of(I);
            EligibilityResult e = UdfEligibilityAnalyzer.analyse(resolved, in, I, false, true, false);
            assertFalse(e.isEligible(), () -> udf.getClass().getSimpleName() + " has no error channel on the runtime backend");
        }
        EligibilityResult k = UdfEligibilityAnalyzer.analyse(UdfResolver.resolve(new KnownConstant()).orElseThrow(), List.of(I), I, false, true, false);
        assertTrue(k.isEligible(), k::reason);
    }

    // ============================================================================ helpers

    /** The error channel's zero test, for an int or a long divisor. */
    private static boolean hasZeroTest(String cuda) {
        return java.util.regex.Pattern.compile("== 0(LL)?\\) return 2;").matcher(cuda).find();
    }

    private static Object scalaSeqOrArray(long[] row) {
        return row;
    }

    /** Accepted, compiled with the error channel; the CUDA. */
    private static String guarded(Object udf, List<DataType> in, DataType out) {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, in.toArray(new DataType[0]), out, false, udf.getClass().getSimpleName());
        assertTrue(o.accepted(), () -> udf.getClass().getSimpleName() + ": " + o.decision());
        assertTrue(o.cuda().contains("__device__ int"), () -> udf.getClass().getSimpleName() + " must report errors\n" + o.cuda());
        return o.cuda();
    }
}
