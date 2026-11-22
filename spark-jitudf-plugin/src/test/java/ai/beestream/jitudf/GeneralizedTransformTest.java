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

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

/**
 * The three capabilities the widened cuDF binding unlocks, checked on the driver and without a GPU:
 * several input columns, an output type other than INT32, and a nullable column.
 *
 * <p>
 * None of these were ever the compiler's limits. TC has compiled {@code void f(R *out, A0 in0, A1
 * in1, ...)} for any arity and any fixed-width result since device functions existed, and its
 * null-aware mode has compiled {@code cuda::std::optional}s since PR #1335. What they were is the
 * Java binding's: {@code ColumnView.transform} passed one column and always asked for INT32, with
 * no way to ask for null-awareness at all. So what these tests establish is that the planner now
 * accepts the shapes the compiler was already producing -- and, for nulls, that the <em>meaning</em>
 * is the Java method's own, not a mask copied from the input.
 * </p>
 */
class GeneralizedTransformTest {

    // ---------------------------------------------------------------------------------------
    // Several inputs
    // ---------------------------------------------------------------------------------------

    /** Two inputs of the same type. */
    public static final class WeightedSumUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer a, Integer b) {
            return a + b * 2;
        }
    }

    /** Two inputs of different types, and a result that is neither. */
    public static final class MixedRatioUdf implements UDF2<Integer, Float, Double> {
        private static final long serialVersionUID = 1L;

        @Override
        public Double call(Integer a, Float b) {
            return a.doubleValue() + b.doubleValue();
        }
    }

    /** Three inputs, a loop, and a 64-bit result. */
    public static final class ThreeInputUdf implements UDF3<Long, Integer, Double, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Integer b, Double c) {
            long acc = a;
            for (int i = 0; i < b; i++) {
                acc = acc * 31 + c.longValue();
            }
            return acc;
        }
    }

    @Test
    void twoInputsCompileToATwoArgumentDeviceFunction() {
        TcCudaCompiler.Result result = compile(new WeightedSumUdf(),
                List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> "signature was:\n" + cuda),
                // cudf passes the output pointer and then one value per column, so three parameters.
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 2), () -> "cudf would not bind this:\n" + cuda),
                // ... and two is not one: the single-input check must now fail.
                () -> assertTrue(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 1) != null, cuda));
    }

    @Test
    void mixedInputTypesKeepTheirOwnWidths() {
        TcCudaCompiler.Result result = compile(new MixedRatioUdf(),
                List.of(DataTypes.IntegerType, DataTypes.FloatType), DataTypes.DoubleType);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(double *"), () -> "signature was:\n" + cuda),
                () -> assertTrue(cuda.contains("int ") && cuda.contains("float "), () -> "both input widths should survive:\n" + cuda),
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 2), cuda));
    }

    @Test
    void threeInputsCompile() {
        TcCudaCompiler.Result result = compile(new ThreeInputUdf(),
                List.of(DataTypes.LongType, DataTypes.IntegerType, DataTypes.DoubleType), DataTypes.LongType);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(long long *"), () -> "signature was:\n" + cuda),
                () -> assertTrue(cuda.contains("for") || cuda.contains("while"), () -> "the loop should survive:\n" + cuda),
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 3), cuda));
    }

    // ---------------------------------------------------------------------------------------
    // Output types other than INT32
    // ---------------------------------------------------------------------------------------

    public static final class LongOutUdf implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long v) {
            long x = v;
            long acc = 0;
            for (int i = 0; i < 16; i++) {
                acc += x * i;
                x >>>= 1;
            }
            return acc;
        }
    }

    public static final class DoubleOutUdf implements UDF1<Double, Double> {
        private static final long serialVersionUID = 1L;

        @Override
        public Double call(Double v) {
            return Math.sqrt(v * v + 1.0);
        }
    }

    public static final class FloatOutUdf implements UDF1<Integer, Float> {
        private static final long serialVersionUID = 1L;

        @Override
        public Float call(Integer v) {
            return v * 0.5f;
        }
    }

    @Test
    void theOutputPointerHasTheResultsOwnType() {
        assertAll(
                () -> assertTrue(compile(new LongOutUdf(), List.of(DataTypes.LongType), DataTypes.LongType)
                        .source().contains("__device__ void GENERIC_TRANSFORM_OP(long long *")),
                () -> assertTrue(compile(new DoubleOutUdf(), List.of(DataTypes.DoubleType), DataTypes.DoubleType)
                        .source().contains("__device__ void GENERIC_TRANSFORM_OP(double *")),
                () -> assertTrue(compile(new FloatOutUdf(), List.of(DataTypes.IntegerType), DataTypes.FloatType)
                        .source().contains("__device__ void GENERIC_TRANSFORM_OP(float *")));
    }

    @Test
    void theCudfTypeAskedForIsTheResultsType() {
        // The type that reaches the binding is PrimitiveKind's, so this is the mapping the
        // executor will use when it calls transform.
        assertAll(
                () -> assertEquals(ai.rapids.cudf.DType.INT32, PrimitiveKind.INT.cudfType()),
                () -> assertEquals(ai.rapids.cudf.DType.INT64, PrimitiveKind.LONG.cudfType()),
                () -> assertEquals(ai.rapids.cudf.DType.FLOAT32, PrimitiveKind.FLOAT.cudfType()),
                () -> assertEquals(ai.rapids.cudf.DType.FLOAT64, PrimitiveKind.DOUBLE.cudfType()));
    }

    // ---------------------------------------------------------------------------------------
    // Nullable columns
    // ---------------------------------------------------------------------------------------

    /** Answers for a null rather than propagating it: the case a validity-mask copy gets wrong. */
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

    /** Returns null for a null, and only for a null. */
    public static final class PropagatingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long value) {
            if (value == null) {
                return null;
            }
            return value.intValue() * 2;
        }
    }

    /** Returns null for inputs it does not like, which are not null. */
    public static final class RejectingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long value) {
            if (value == null) {
                return 0;
            }
            if (value < 0) {
                return null;
            }
            return value.intValue();
        }
    }

    /** Unboxes without checking: a null is a NullPointerException on the CPU. */
    public static final class UnboxingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long value) {
            return (int) (value * 3L);
        }
    }

    /** Two nullable inputs, either of which may be absent. */
    public static final class CoalescingUdf implements UDF2<Long, Integer, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Integer b) {
            if (a == null && b == null) {
                return null;
            }
            long av = a == null ? 0L : a;
            long bv = b == null ? 0L : b.longValue();
            return av + bv;
        }
    }

    @Test
    void aNullableColumnIsStillRefusedByTheRuntimeBackend() {
        // No null-aware mode and no binding that could pass optionals: the old refusal stands, and
        // the reason says which backend it is about.
        ResolvedUdf resolved = resolve(new DefaultingUdf());
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.LongType), DataTypes.IntegerType, true, true, false, false);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("this backend cannot"), result::reason);
    }

    @Test
    void aNullableColumnIsAcceptedNullAware() {
        ResolvedUdf resolved = resolve(new DefaultingUdf());
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.LongType), DataTypes.IntegerType, true, true, true, true);
        assertTrue(result.isEligible(), result::reason);
        assertTrue(result.isNullAware(), "a nullable column must be compiled null-aware");
    }

    @Test
    void aUdfWithoutNullTestsOnANonNullableColumnIsNotCompiledNullAware() {
        // Null-awareness costs an optional per value, so it is not used where nothing needs it.
        ResolvedUdf resolved = resolve(new WeightedSumUdf());
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved,
                List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType, false, true, true, true);
        assertTrue(result.isEligible(), result::reason);
        assertFalse(result.isNullAware());
    }

    @Test
    void aDefensiveNullTestIsCompiledNullAwareEvenOnANonNullableColumn() {
        // The test is dead code over a column that cannot hold a null, but it is still in the
        // bytecode, and the plain lowering erases the box it is about. Refusing the UDF would be
        // the alternative, and it would be a refusal of something that works.
        ResolvedUdf resolved = resolve(new DefaultingUdf());
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved,
                List.of(DataTypes.LongType), DataTypes.IntegerType, false, true, true, true);
        assertTrue(result.isEligible(), result::reason);
        assertTrue(result.isNullAware(), "a UDF that tests for null has to be compiled where null means something");
    }

    @Test
    void aUdfThatAnswersForANullCompilesToOptionals() {
        TcCudaCompiler.Result result = compileNullAware(new DefaultingUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.startsWith("#include <cuda/std/optional>"), () -> "the optional header must come first:\n" + cuda),
                // Null-aware functions always return an error code, so the result is int, not void.
                () -> assertTrue(cuda.contains("__device__ int GENERIC_TRANSFORM_OP(cuda::std::optional<int> *"), () -> "signature was:\n" + cuda),
                () -> assertTrue(cuda.contains("cuda::std::optional<long long>"), () -> "the input must arrive as an optional:\n" + cuda),
                // The #include has no '(' in it, so cudf still renames the right identifier.
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 1), () -> "cudf would not bind this:\n" + cuda));
    }

    @Test
    void aUdfThatReturnsNullCompilesToANullopt() {
        TcCudaCompiler.Result result = compileNullAware(new PropagatingUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType);
        assertNull(result.refusal(), result::refusal);
        assertTrue(result.source().contains("nullopt"), () -> "a returned null is a nullopt:\n" + result.source());
    }

    @Test
    void aUdfThatReturnsNullForANonNullInputAlsoCompiles() {
        // This is the one a validity-mask copy can never express: the output is null where the
        // input was not.
        TcCudaCompiler.Result result = compileNullAware(new RejectingUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType);
        assertNull(result.refusal(), result::refusal);
        assertTrue(result.source().contains("nullopt"), () -> "a returned null is a nullopt:\n" + result.source());
    }

    @Test
    void unboxingANullIsAnErrorCodeRatherThanAWrongNumber() {
        TcCudaCompiler.Result result = compileNullAware(new UnboxingUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType);
        assertNull(result.refusal(), result::refusal);
        // 3 is TC's code for a NullPointerException. cudf names 0, 1 and 2 only, so it reports this
        // as UNKNOWN_ERROR: the query fails, as it does on the CPU, but the exception is not the
        // same exception and this integration does not claim it is.
        assertTrue(result.source().contains("return 3;"), () -> "no NPE code emitted:\n" + result.source());
    }

    @Test
    void twoNullableInputsCompile() {
        TcCudaCompiler.Result result = compileNullAware(new CoalescingUdf(),
                List.of(DataTypes.LongType, DataTypes.IntegerType), DataTypes.LongType);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains("__device__ int GENERIC_TRANSFORM_OP(cuda::std::optional<long long> *"), () -> "signature was:\n" + cuda),
                () -> assertTrue(cuda.contains("cuda::std::optional<int>"), () -> "the second input must be an optional too:\n" + cuda),
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 2), cuda));
    }

    /** Compares two boxes by identity, which Java answers from its Integer cache, not the numbers. */
    public static final class BoxIdentityUdf implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer a, Integer b) {
            return a == b ? 1 : 0;
        }
    }

    @Test
    void comparingTwoBoxesIsStillRefused() {
        ResolvedUdf resolved = resolve(new BoxIdentityUdf());
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved,
                List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType, true, true, true, true);
        assertFalse(result.isEligible(), "reference identity is not arithmetic, null-aware or not");
        assertTrue(result.reason().contains("identity"), result::reason);
    }

    @Test
    void theCacheKeyDistinguishesANullAwareCompilation() {
        // The two compilations of one UDF produce different CUDA; sharing a key would ship one of
        // them for the other's column.
        // A UDF that does not mention null, so the nullability of the column is what decides which
        // of the two compilations it gets -- and the two produce different CUDA.
        ResolvedUdf resolved = resolve(new WeightedSumUdf());
        List<DataType> types = List.of(DataTypes.IntegerType, DataTypes.IntegerType);
        EligibilityResult plain = UdfEligibilityAnalyzer.analyse(resolved, types, DataTypes.IntegerType, false, true, true, true);
        EligibilityResult nullable = UdfEligibilityAnalyzer.analyse(resolved, types, DataTypes.IntegerType, true, true, true, true);
        assertFalse(plain.isNullAware());
        assertTrue(nullable.isNullAware());
        String plainKey = KernelIdentity.of(resolved, plain.inputs(), plain.outputKind())
                + (plain.reportsArithmeticErrors() ? "+errc" : "") + (plain.isNullAware() ? "+null" : "");
        String nullableKey = KernelIdentity.of(resolved, nullable.inputs(), nullable.outputKind())
                + (nullable.reportsArithmeticErrors() ? "+errc" : "") + (nullable.isNullAware() ? "+null" : "");
        assertFalse(plainKey.equals(nullableKey), "the null-aware compilation needs its own cache entry");
    }

    // ---------------------------------------------------------------------------------------

    private static ResolvedUdf resolve(Object udf) {
        return ConstantFieldFolding.fold(UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure())));
    }

    private static TcCudaCompiler.Result compile(Object udf, List<DataType> inputTypes, DataType outputType) {
        ResolvedUdf resolved = resolve(udf);
        // canRunNullAware false: this helper is for the plain shape, and none of the UDFs it is
        // used with mention null, so the two would agree anyway -- but saying so here keeps the
        // assertion below meaningful rather than incidental.
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, inputTypes, outputType, false, true, true, false);
        assertTrue(eligibility.isEligible(), () -> "not eligible: " + eligibility.reason());
        assertFalse(eligibility.isNullAware());
        String identity = KernelIdentity.of(resolved, eligibility.inputs(), eligibility.outputKind());
        CudfDeviceFunctionGenerator.Generated generated =
                CudfDeviceFunctionGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind(), identity);
        return TcCudaCompiler.compile(generated, eligibility.inputs(), udf.getClass().getClassLoader(), eligibility.reportsArithmeticErrors());
    }

    private static TcCudaCompiler.Result compileNullAware(Object udf, List<DataType> inputTypes, DataType outputType) {
        ResolvedUdf resolved = resolve(udf);
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, inputTypes, outputType, true, true, true, true);
        assertTrue(eligibility.isEligible(), () -> "not eligible: " + eligibility.reason());
        assertTrue(eligibility.isNullAware());
        String identity = KernelIdentity.of(resolved, eligibility.inputs(), eligibility.outputKind()) + "+null";
        CudfDeviceFunctionGenerator.Generated generated =
                CudfDeviceFunctionGenerator.generateNullAware(resolved, eligibility.inputs(), identity);
        return TcCudaCompiler.compileNullAware(generated, eligibility.inputs(), udf.getClass().getClassLoader(), eligibility.reportsArithmeticErrors());
    }
}
