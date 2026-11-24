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
import static org.junit.jupiter.api.Assertions.assertNull;
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
 * Codex findings 6 and 7: what a computation can do is read from all of its reachable code,
 * including the JDK bodies TC inlines, and not from its signature.
 */
class ArithmeticEffectsTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;
    static final DataType B = DataTypes.BooleanType;

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ================================================================== floorDiv / floorMod

    public static final class FloorDivLong implements UDF2<Long, Long, Long> {
        @Override
        public Long call(Long a, Long b) {
            return Math.floorDiv(a, b);
        }
    }

    public static final class FloorModLong implements UDF2<Long, Long, Long> {
        @Override
        public Long call(Long a, Long b) {
            return Math.floorMod(a, b);
        }
    }

    public static final class FloorModMixed implements UDF2<Long, Integer, Integer> {
        @Override
        public Integer call(Long a, Integer b) {
            return Math.floorMod(a, b);
        }
    }

    public static final class FloorDivInHelper implements UDF2<Integer, Integer, Integer> {
        static int bucket(int a, int b) {
            return Math.floorDiv(a, b) + 1;
        }

        @Override
        public Integer call(Integer a, Integer b) {
            return bucket(a, b);
        }
    }

    public static final class FloorDivByZeroConstant implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer a) {
            return Math.floorDiv(a, 0);
        }
    }

    public static final class FloorDivByMinusOne implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer a) {
            return Math.floorDiv(a, -1);
        }
    }

    @Test
    void aVariableDivisorInsideFloorDivOrFloorModGetsTheErrorChannel() {
        for (Object udf : new Object[] { new CodexProbeTest.FloorDivUdf(), new CodexProbeTest.FloorModUdf(), new FloorDivInHelper() }) {
            errorChannel(udf, new DataType[] { I, I }, I);
        }
        errorChannel(new FloorDivLong(), new DataType[] { L, L }, L);
        errorChannel(new FloorModLong(), new DataType[] { L, L }, L);
        errorChannel(new FloorModMixed(), new DataType[] { L, I }, I);
        errorChannel(new FloorDivByZeroConstant(), new DataType[] { I }, I);
    }

    @Test
    void aNonZeroConstantDivisorNeedsNoErrorChannel() {
        ProviderHarness.Outcome mod = ProviderHarness.cudf(new CoverageCorpus.FloorModConstantInt(), new DataType[] { I }, I, false, "modConst");
        assertTrue(mod.accepted(), mod::decision);
        assertTrue(mod.cuda().contains("__device__ void GENERIC_TRANSFORM_OP"), mod::cuda);
        ProviderHarness.Outcome minusOne = ProviderHarness.cudf(new FloorDivByMinusOne(), new DataType[] { I }, I, false, "divMinusOne");
        assertTrue(minusOne.accepted(), minusOne::decision);
    }

    @Test
    void floorDivAndFloorModComputeJavasAnswerWhereJavaDoesNotThrow() throws Exception {
        Method div = LoweredOracle.scalarBody(new CodexProbeTest.FloorDivUdf(), List.of(I, I), I);
        Method mod = LoweredOracle.scalarBody(new CodexProbeTest.FloorModUdf(), List.of(I, I), I);
        int[] values = { 7, -7, 0, 1, -1, Integer.MIN_VALUE, Integer.MAX_VALUE };
        for (int a : values) {
            for (int b : values) {
                assertEquals(LoweredOracle.reference(new CodexProbeTest.FloorDivUdf(), a, b), LoweredOracle.run(div, a, b), a + " floorDiv " + b);
                assertEquals(LoweredOracle.reference(new CodexProbeTest.FloorModUdf(), a, b), LoweredOracle.run(mod, a, b), a + " floorMod " + b);
            }
        }
        assertEquals(Integer.MIN_VALUE, LoweredOracle.run(div, Integer.MIN_VALUE, -1), "MIN_VALUE floorDiv -1 wraps in Java");
        assertEquals("threw:ArithmeticException", LoweredOracle.reference(new CodexProbeTest.FloorDivUdf(), 1, 0));
    }

    @Test
    void theRuntimeBackendWhichCannotReportErrorsDeclinesAVariableFloorDivisor() throws Exception {
        ResolvedUdf resolved = UdfResolver.resolve(new CodexProbeTest.FloorDivUdf()).orElseThrow();
        EligibilityResult e = UdfEligibilityAnalyzer.analyse(resolved, List.of(I, I), I, false, true, false);
        assertFalse(e.isEligible());
        assertTrue(e.reason().contains("floorDiv"), e.reason());
        EligibilityResult constant = UdfEligibilityAnalyzer.analyse(UdfResolver.resolve(new CoverageCorpus.FloorModConstantInt()).orElseThrow(), List.of(I), I, false, true, false);
        assertTrue(constant.isEligible(), constant.reason());
    }

    private static void errorChannel(Object udf, DataType[] inputs, DataType output) {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, inputs, output, false, udf.getClass().getSimpleName());
        assertTrue(o.accepted(), () -> udf.getClass().getSimpleName() + ": " + o.decision());
        String cuda = o.cuda();
        assertTrue(cuda.contains("__device__ int GENERIC_TRANSFORM_OP"), () -> udf.getClass().getSimpleName() + " must return a cudf error code:\n" + cuda);
        assertTrue(cuda.contains("return 2;"), () -> udf.getClass().getSimpleName() + " must report DIVISION_BY_ZERO:\n" + cuda);
    }

    // ===================================================================== floating point

    public static final class FpPredicate implements UDF1<Integer, Boolean> {
        @Override
        public Boolean call(Integer x) {
            return Math.sqrt(x) > 3.0;
        }
    }

    public static final class FpInHelper implements UDF1<Integer, Integer> {
        static double scale(int x) {
            return x * 0.5;
        }

        @Override
        public Integer call(Integer x) {
            return (int) scale(x);
        }
    }

    public static final class FpViaBoxConversion implements UDF1<Integer, Long> {
        @Override
        public Long call(Integer x) {
            return (long) (x.doubleValue() / 3);
        }
    }

    public static final class FpBucket implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            return x * 0.1 > 10 ? 1 : 0;
        }
    }

    public static final class Narrowing implements UDF1<Double, Integer> {
        @Override
        public Integer call(Double x) {
            return (int) (double) x + (int) (long) (double) x;
        }
    }

    @Test
    void integerAndBooleanSignaturesWithFloatingPointInsideNeedTheSwitch() {
        Object[][] cases = { { new CoverageCorpus.SinInInt(), I, I }, { new FpPredicate(), I, B }, { new FpInHelper(), I, I },
                { new FpViaBoxConversion(), I, L }, { new FpBucket(), L, I } };
        for (Object[] c : cases) {
            String name = c[0].getClass().getSimpleName();
            ProviderHarness.Outcome off = ProviderHarness.cudf(c[0], new DataType[] { (DataType) c[1] }, (DataType) c[2], false, name + "Off", false);
            assertFalse(off.accepted(), () -> name + " computes in floating point and must not pass with the switch off: " + off.decision());
            assertTrue(off.decision().contains(SparkSemantics.INCOMPATIBLE_OPS), off::decision);
            ProviderHarness.Outcome on = ProviderHarness.cudf(c[0], new DataType[] { (DataType) c[1] }, (DataType) c[2], false, name + "On", true);
            assertTrue(on.accepted(), () -> name + " is allowed once the user opts in: " + on.decision());
        }
    }

    @Test
    void integerUdfsWithNoFloatingPointAnywhereNeedNoSwitch() {
        Object[][] cases = { { new CoverageCorpus.IntOut(), I, I }, { new CoverageCorpus.Collatz(), L, I }, { new CoverageCorpus.NestedHelpers(), L, L },
                { new CoverageCorpus.BitCount(), L, I }, { new CoverageCorpus.IntMath(), null, I }, { new CoverageCorpus.FloorModConstantInt(), I, I } };
        for (Object[] c : cases) {
            DataType[] inputs = c[1] == null ? new DataType[] { I, I } : new DataType[] { (DataType) c[1] };
            String name = c[0].getClass().getSimpleName();
            ProviderHarness.Outcome off = ProviderHarness.cudf(c[0], inputs, (DataType) c[2], false, name, false);
            assertTrue(off.accepted(), () -> name + " has no floating point and must stay eligible: " + off.decision());
        }
    }

    @Test
    void doubleToIntNarrowingKeepsJavasSaturationInTheLoweredCode() throws Exception {
        Method body = LoweredOracle.scalarBody(new Narrowing(), List.of(DataTypes.DoubleType), I);
        for (double x : new double[] { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e300, -1e300, 2147483647.9, -2147483648.9, -0.0, 3.7 }) {
            assertEquals(LoweredOracle.reference(new Narrowing(), x), LoweredOracle.run(body, x), "x=" + x);
        }
        ProviderHarness.Outcome o = ProviderHarness.cudf(new Narrowing(), new DataType[] { DataTypes.DoubleType }, I, false, "narrowing", true);
        assertTrue(o.accepted(), o::decision);
        // Java's d2i/d2l map NaN to 0 and saturate; the device conversion must be the saturating,
        // round-toward-zero one with an explicit NaN test.
        assertTrue(o.cuda().contains("__double2int_rz") && o.cuda().contains("__double2ll_rz"), o::cuda);
        assertTrue(o.cuda().contains("!="), () -> "a NaN guard is expected:\n" + o.cuda());
    }

    @Test
    void theEffectSummaryOfAnInlinedJdkBodyIsReadFromThatBody() {
        var floorDiv = new org.objectweb.asm.tree.MethodInsnNode(org.objectweb.asm.Opcodes.INVOKESTATIC, "java/lang/Math", "floorDiv", "(II)I", false);
        var bitCount = new org.objectweb.asm.tree.MethodInsnNode(org.objectweb.asm.Opcodes.INVOKESTATIC, "java/lang/Long", "bitCount", "(J)I", false);
        var sqrt = new org.objectweb.asm.tree.MethodInsnNode(org.objectweb.asm.Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
        assertTrue(ComputationEffects.ofJdkCall(floorDiv).mayDivideByZero());
        assertFalse(ComputationEffects.ofJdkCall(floorDiv).usesFloatingPoint());
        assertEquals(ComputationEffects.Effects.NONE, ComputationEffects.ofJdkCall(bitCount));
        assertFalse(ComputationEffects.ofJdkCall(sqrt).mayDivideByZero(), "an intrinsic never runs the JDK body");
        assertTrue(ComputationEffects.ofJdkCall(sqrt).usesFloatingPoint());
        for (String jdk : UdfEligibilityAnalyzer.supportedJdkStatics()) {
            int dot = jdk.indexOf('.');
            int paren = jdk.indexOf('(');
            var call = new org.objectweb.asm.tree.MethodInsnNode(org.objectweb.asm.Opcodes.INVOKESTATIC, jdk.substring(0, dot), jdk.substring(dot + 1, paren), jdk.substring(paren), false);
            assertNull(ComputationEffects.ofJdkCall(call).mayDivideByZero() ? jdk : null, "no listed bit-manipulation static divides");
        }
    }
}
