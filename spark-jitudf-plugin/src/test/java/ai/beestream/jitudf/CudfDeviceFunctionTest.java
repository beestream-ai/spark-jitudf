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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

/**
 * The driver half of the NVIDIA-aligned path, end to end and without a GPU.
 *
 * <p>
 * Every test here runs the real sequence -- resolve the UDF, analyse it, lower it, generate the
 * cuDF entry class, compile it with TornadoVM's TC compiler, and check the CUDA that comes out --
 * on an ordinary JVM. Nothing is stubbed. That is possible at all because TC reads class files
 * rather than loading them and depends on no native library, which is the same property that lets
 * this run on a Spark driver.
 * </p>
 */
class CudfDeviceFunctionTest {

    /** A loop: what the RAPIDS UDF compiler cannot express, since an expression tree has no back edge. */
    public static final class CollatzUdf implements UDF1<Long, Integer> {
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

    /** Bit intrinsics: an activity bitmap packed in a long. */
    public static final class ActiveDaysUdf implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long bitmap) {
            return Long.bitCount(bitmap);
        }
    }

    /** Murmur3_32 with a non-default seed, verified elsewhere to be bit-equal to Guava's. */
    public static final class BucketUdf implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long userXorSalt) {
            int h1 = 0;
            int k1 = (int) (long) userXorSalt;
            k1 *= 0xcc9e2d51;
            k1 = Integer.rotateLeft(k1, 15);
            k1 *= 0x1b873593;
            h1 ^= k1;
            h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
            k1 = (int) (userXorSalt >>> 32);
            k1 *= 0xcc9e2d51;
            k1 = Integer.rotateLeft(k1, 15);
            k1 *= 0x1b873593;
            h1 ^= k1;
            h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
            h1 ^= 8;
            h1 ^= h1 >>> 16;
            h1 *= 0x85ebca6b;
            h1 ^= h1 >>> 13;
            h1 *= 0xc2b2ae35;
            h1 ^= h1 >>> 16;
            return Math.floorMod(h1, 1000);
        }
    }

    /** Two inputs and a double result: beyond today's cuDF Java binding, but not beyond the compiler. */
    public static final class TwoInputUdf implements UDF2<Double, Double, Double> {
        @Override
        public Double call(Double a, Double b) {
            double acc = 0;
            for (int i = 0; i < 8; i++) {
                acc += Math.sqrt(a * i + b);
            }
            return acc;
        }
    }

    /** Integer division: throws on the CPU for a zero divisor, does not on the GPU. */
    public static final class DividingUdf implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            int acc = 0;
            for (int i = 1; i < 8; i++) {
                acc += (int) (x / i);
            }
            return acc;
        }
    }

    /** The ordinary way a UDF is configured: a constructor argument kept in a final field. */
    public static final class ConfiguredUdf implements UDF1<Long, Integer> {
        private final int rounds;

        public ConfiguredUdf(int rounds) {
            this.rounds = rounds;
        }

        @Override
        public Integer call(Long seed) {
            long x = seed;
            int steps = 0;
            for (int i = 0; i < rounds; i++) {
                x = (x & 1L) == 1L ? 3 * x + 1 : x >>> 1;
                steps++;
            }
            return steps;
        }
    }

    private record Compiled(CudfDeviceFunctionGenerator.Generated generated, TcCudaCompiler.Result result) {
    }

    private static Compiled compile(Object udf, List<org.apache.spark.sql.types.DataType> inputTypes, org.apache.spark.sql.types.DataType outputType) {
        // Folded first, exactly as TornadoUdfProvider does, so this helper exercises the real path.
        ResolvedUdf resolved = ConstantFieldFolding.fold(UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure())));
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, inputTypes, outputType, false, true);
        assertTrue(eligibility.isEligible(), () -> "not eligible: " + eligibility.reason());
        String identity = KernelIdentity.of(resolved, eligibility.inputs(), eligibility.outputKind());
        CudfDeviceFunctionGenerator.Generated generated = CudfDeviceFunctionGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind(), identity);
        TcCudaCompiler.Result result = TcCudaCompiler.compile(generated, eligibility.inputs(), udf.getClass().getClassLoader());
        return new Compiled(generated, result);
    }

    @Test
    void aLoopingUdfBecomesASelfContainedCudaDeviceFunction() {
        TcCudaCompiler.Result result = compile(new CollatzUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> "signature was:\n" + cuda),
                // cudf's kernel calls f(out_ptr, in...), so the output pointer comes first.
                () -> assertTrue(cuda.contains("long long"), () -> "the long input should survive:\n" + cuda),
                () -> assertTrue(cuda.contains("while"), () -> "the loop should survive:\n" + cuda),
                // None of the things only a launched kernel may do.
                () -> assertTrue(!cuda.contains("__global__"), () -> "a transform UDF is not a kernel:\n" + cuda),
                () -> assertTrue(!cuda.contains("threadIdx"), () -> "cudf owns the launch:\n" + cuda));
    }

    @Test
    void theGeneratedSourceIsWhatCudfWouldBind() {
        // cudf renames the first function in the string; this asserts the first one is ours.
        TcCudaCompiler.Result result = compile(new CollatzUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal());
        assertNull(CudaSourceCheck.check(result.source(), TcCudaCompiler.ENTRY_SYMBOL, 1));
    }

    /**
     * Integer division and remainder compile. They were refused outright, on the grounds that TC
     * lowers Java's division semantics through {@code java_idiv}/{@code java_lrem} helpers and cudf
     * binds the first function in the source. Two things were wrong with that. TC emits those
     * helpers as lambdas inside the entry now, so a helper is no longer a second function; and the
     * case never arose anyway, because {@link UdfEligibilityAnalyzer} only admits a constant
     * divisor, for which TC emits a plain {@code /} and no helper at all. The refusal was costing
     * every dividing UDF for a problem none of them had.
     */
    @Test
    void integerDivisionByAConstantCompilesAsOneFunction() {
        TcCudaCompiler.Result result = compile(new RatioUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                () -> assertTrue(cuda.contains(" / 7") || cuda.contains("/ 7LL"), () -> "a constant divisor is a plain divide:\n" + cuda),
                () -> assertTrue(cuda.contains(" % 13") || cuda.contains("% 13"), () -> "and so is the remainder:\n" + cuda),
                // No helper, so nothing can precede the entry -- checked by cudf's own rule.
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 1), () -> "cudf would bind the wrong function:\n" + cuda));
    }

    /** Divides and takes a remainder by constants, which is what the analyser admits. */
    public static final class RatioUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            return (int) ((v / 7L) % 13L);
        }
    }

    /**
     * A divisor the analysis cannot prove non-zero is no longer refused outright: the compilation
     * carries error reporting, and the entry returns a cudf error code. This needs no change to the
     * transform binding, because cudf's JIT kernel casts a non-void result to {@code cudf::errc}
     * (release/26.08, cpp/src/transform/jit/kernel.cu) and raises for the call.
     */
    @Test
    void aVariableDivisorCompilesWithErrorReporting() {
        ResolvedUdf resolved = ConstantFieldFolding.fold(UdfResolver.resolve(new VariableDivisorUdf()).orElseThrow());
        EligibilityResult without = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.LongType), DataTypes.IntegerType, false, true, false);
        assertFalse(without.isEligible(), "without an error channel this must still be refused");
        assertTrue(without.reason().contains("cannot prove non-zero"), without::reason);

        EligibilityResult with = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.LongType), DataTypes.IntegerType, false, true, true);
        assertTrue(with.isEligible(), with::reason);
        assertTrue(with.reportsArithmeticErrors(), "the compilation has to report errors");

        String identity = KernelIdentity.of(resolved, with.inputs(), with.outputKind());
        CudfDeviceFunctionGenerator.Generated generated = CudfDeviceFunctionGenerator.generate(resolved, with.inputs(), with.outputKind(), identity);
        TcCudaCompiler.Result result = TcCudaCompiler.compile(generated, with.inputs(), new VariableDivisorUdf().getClass().getClassLoader(), true);
        assertNull(result.refusal(), result::refusal);
        String cuda = result.source();
        assertAll(
                // int, not void: that is the whole mechanism.
                () -> assertTrue(cuda.contains("__device__ int GENERIC_TRANSFORM_OP("), () -> "signature was:\n" + cuda),
                // 2 is cudf::errc::DIVISION_BY_ZERO.
                () -> assertTrue(cuda.contains("return 2;"), () -> "no division-by-zero code emitted:\n" + cuda),
                () -> assertTrue(cuda.contains("return 0;"), () -> "no success code emitted:\n" + cuda),
                () -> assertNull(CudaSourceCheck.check(cuda, TcCudaCompiler.ENTRY_SYMBOL, 1), () -> "cudf would bind the wrong function:\n" + cuda));
    }

    /** Divides by a value derived from the input, which no constant analysis can prove non-zero. */
    public static final class VariableDivisorUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            long d = v >>> 32;
            return (int) (v / d);
        }
    }

    @Test
    void jdkBitIntrinsicsCompileByInliningTheJdksOwnBytecode() {
        // Long.bitCount is in no TornadoVM intrinsic table. It compiles because TC follows the call
        // into java.base and translates the JDK's own implementation.
        TcCudaCompiler.Result result = compile(new ActiveDaysUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal(), result::refusal);
        assertTrue(result.source().contains("Long.java"), () -> "expected the JDK's own body to be inlined:\n" + result.source());
    }

    @Test
    void aCustomSeedMurmur3BucketingUdfCompiles() {
        // Integer.rotateLeft and Math.floorMod are likewise absent from the intrinsic table.
        TcCudaCompiler.Result result = compile(new BucketUdf(), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal(), result::refusal);
        assertTrue(result.source().contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> result.source());
    }

    @Test
    void severalInputsAndANonIntegerOutputCompile() {
        // The compiler is not the limit here: cudf's Java binding is. This proves the generated code
        // is ready for the widened binding before that binding exists.
        Compiled compiled = compile(new TwoInputUdf(), List.of(DataTypes.DoubleType, DataTypes.DoubleType), DataTypes.DoubleType);
        assertNull(compiled.result().refusal(), () -> compiled.result().refusal());
        assertTrue(compiled.result().source().contains("__device__ void GENERIC_TRANSFORM_OP(double *"), () -> compiled.result().source());
        assertEquals("(Luk/ac/manchester/tornado/api/types/arrays/DoubleArray;DD)V", compiled.generated().entryDescriptor());
    }

    @Test
    void theSourceCheckRefusesWhatCudfWouldMisbind() {
        // A helper ahead of the entry: cudf would rename the helper, not the entry.
        String withHelper = "__device__ static int helper(int x) { return x; }\n"
                + "__device__ void GENERIC_TRANSFORM_OP(int *out, int in0) { out[0] = helper(in0); }\n";
        String refusal = CudaSourceCheck.check(withHelper, TcCudaCompiler.ENTRY_SYMBOL, 1);
        assertNotNull(refusal);
        assertTrue(refusal.contains("helper"), refusal);
    }

    @Test
    void theSourceCheckRefusesAKernelAndAWrongArity() {
        assertAll(
                () -> assertNotNull(CudaSourceCheck.check("extern \"C\" __global__ void GENERIC_TRANSFORM_OP(int *o) { }", TcCudaCompiler.ENTRY_SYMBOL, 0)),
                () -> assertNotNull(CudaSourceCheck.check("__device__ void GENERIC_TRANSFORM_OP(int *o) { o[0] = threadIdx.x; }", TcCudaCompiler.ENTRY_SYMBOL, 0)),
                () -> assertNotNull(CudaSourceCheck.check("__device__ void GENERIC_TRANSFORM_OP(int *o, int a) { o[0] = a; }", TcCudaCompiler.ENTRY_SYMBOL, 2)),
                () -> assertNotNull(CudaSourceCheck.check("__device__ void GENERIC_TRANSFORM_OP(int o, int a) { }", TcCudaCompiler.ENTRY_SYMBOL, 1)));
    }

    @Test
    void aListArgumentIsRefusedBecauseTransformCannotTakeOne() {
        // cudf::transform throws on any input that is not fixed-width or a string, so a LIST has no
        // path here at all. It is refused while generating rather than discovered on the executor.
        ResolvedUdf resolved = UdfResolver.resolve(new CollatzUdf()).orElseThrow();
        List<InputSpec> listInput = List.of(InputSpec.list(PrimitiveKind.DOUBLE));
        String identity = KernelIdentity.of(resolved, listInput, PrimitiveKind.INT);
        assertTrue(assertThrowsIllegalArgument(() -> CudfDeviceFunctionGenerator.generate(resolved, listInput, PrimitiveKind.INT, identity)).contains("LIST"));
    }

    @Test
    void theProviderReturnsACudfTransformUdfUnderTheCudfBackend() {
        // The whole driver path through the entry point RAPIDS actually calls.
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        try {
            Object udf = TornadoUdfProvider.tryCompile(new CollatzUdf(), new org.apache.spark.sql.types.DataType[] { DataTypes.LongType },
                    DataTypes.IntegerType, false, true, "collatz", new String[] { "LOOP" }, "a loop");
            assertNotNull(udf, "the provider refused a UDF it should accept");
            assertTrue(udf instanceof CudfTransformUdf, () -> "got " + udf.getClass());
            String cuda = ((CudfTransformUdf) udf).cudaSource();
            assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> cuda);
        } finally {
            restore(previous);
        }
    }

    @Test
    void theProviderRefusesWhatCudfsBindingCannotRunYet() {
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        try {
            // Two inputs and a double result: the compiler handles it (see the test above), but
            // ColumnView.transform takes one column and returns INT32, so planning must decline.
            Object udf = TornadoUdfProvider.tryCompile(new TwoInputUdf(),
                    new org.apache.spark.sql.types.DataType[] { DataTypes.DoubleType, DataTypes.DoubleType },
                    DataTypes.DoubleType, false, true, "twoInput", new String[] { "LOOP" }, "a loop");
            assertNull(udf, "a UDF outside cudf's binding must be refused at planning time, not failed on an executor");
        } finally {
            restore(previous);
        }
    }

    @Test
    void theDefaultBackendIsUnchanged() {
        // This work must not alter an existing deployment until it is asked for.
        assertEquals(TornadoUdfProvider.BACKEND_TORNADO_RUNTIME, TornadoUdfProvider.backend());
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty("tornado.spark.backend");
        } else {
            System.setProperty("tornado.spark.backend", previous);
        }
    }

    /** Division by a provably non-zero constant: what the analyser allows, and what now compiles. */
    public static final class ConstantDivisorUdf implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            long acc = 0;
            for (int i = 0; i < 8; i++) {
                acc += x / 3;
            }
            return (int) acc;
        }
    }

    /**
     * Which backend is in use decides this one, and that is the point: cudf can be told that a row
     * failed, and the TornadoVM runtime path cannot. So the same UDF compiles for one and is refused
     * by the other, rather than both guessing.
     */
    @Test
    void aVariableDivisorIsAdmittedOnlyByTheBackendThatCanReportTheError() {
        String previous = System.getProperty("tornado.spark.backend");
        try {
            System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
            Object onCudf = TornadoUdfProvider.tryCompile(new DividingUdf(), new org.apache.spark.sql.types.DataType[] { DataTypes.LongType },
                    DataTypes.IntegerType, false, true, "dividing", new String[] { "LOOP" }, "a loop");
            assertNotNull(onCudf, () -> "reason was: " + TornadoUdfProvider.decisions().get("dividing"));

            System.clearProperty("tornado.spark.backend");
            Object onRuntime = TornadoUdfProvider.tryCompile(new DividingUdf(), new org.apache.spark.sql.types.DataType[] { DataTypes.LongType },
                    DataTypes.IntegerType, false, true, "dividing-runtime", new String[] { "LOOP" }, "a loop");
            assertNull(onRuntime, "the runtime backend has no way to report a zero divisor");
            String decision = TornadoUdfProvider.decisions().get("dividing-runtime");
            assertTrue(decision != null && decision.contains("cannot prove non-zero"), () -> "reason was: " + decision);
        } finally {
            restore(previous);
        }
    }

    /** The provider end of the same change: this UDF used to be refused and now becomes a kernel. */
    @Test
    void aDividingUdfReachesTheGpuPath() {
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        try {
            Object udf = TornadoUdfProvider.tryCompile(new ConstantDivisorUdf(), new org.apache.spark.sql.types.DataType[] { DataTypes.LongType },
                    DataTypes.IntegerType, false, true, "constantDivisor", new String[] { "LOOP" }, "a loop");
            assertNotNull(udf, () -> "reason was: " + TornadoUdfProvider.decisions().get("constantDivisor"));
            assertTrue(udf instanceof CudfTransformUdf, () -> "expected a cudf transform UDF, got " + udf.getClass());
        } finally {
            restore(previous);
        }
    }

    @Test
    void floatingPointIsGatedBehindRapidsIncompatibleOps() {
        // fma contraction makes a GPU result differ from Spark's in the last ulp. More accurate is
        // still different, so it needs the same opt-in RAPIDS uses for every other such operation.
        assertTrue(SparkSemantics.usesFloatingPoint(PrimitiveKind.DOUBLE, List.of(InputSpec.scalar(PrimitiveKind.LONG))));
        assertTrue(SparkSemantics.usesFloatingPoint(PrimitiveKind.INT, List.of(InputSpec.scalar(PrimitiveKind.FLOAT))));
        assertNull(SparkSemantics.refusalFor(PrimitiveKind.INT, List.of(InputSpec.scalar(PrimitiveKind.LONG))),
                "an all-integer UDF needs no opt-in");
        String refusal = SparkSemantics.refusalFor(PrimitiveKind.DOUBLE, List.of(InputSpec.scalar(PrimitiveKind.DOUBLE)));
        assertNotNull(refusal);
        assertTrue(refusal.contains(SparkSemantics.INCOMPATIBLE_OPS), refusal);
    }

    @Test
    void compiledCudaIsCachedByTheCodesIdentity() {
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        TornadoUdfProvider.resetDecisions();
        try {
            org.apache.spark.sql.types.DataType[] in = { DataTypes.LongType };
            Object first = TornadoUdfProvider.tryCompile(new CollatzUdf(), in, DataTypes.IntegerType, false, true, "c1", new String[] { "LOOP" }, "");
            long hitsAfterFirst = TornadoUdfProvider.cudaCacheHits();
            // A second, equal instance of the same class: same code, so the same cache entry.
            Object second = TornadoUdfProvider.tryCompile(new CollatzUdf(), in, DataTypes.IntegerType, false, true, "c2", new String[] { "LOOP" }, "");
            assertNotNull(first);
            assertNotNull(second);
            assertEquals(0L, hitsAfterFirst, "the first compilation cannot be a cache hit");
            assertEquals(1L, TornadoUdfProvider.cudaCacheHits(), "the second should have been served from the cache");
            assertEquals(((CudfTransformUdf) first).cudaSource(), ((CudfTransformUdf) second).cudaSource());
        } finally {
            TornadoUdfProvider.resetDecisions();
            restore(previous);
        }
    }

    @Test
    void aConstructorConfiguredUdfCompilesByFoldingItsFinalField() {
        // Without folding this is refused for using `this` to reach state. The instance exists on
        // the driver and the field is final, so its value is a constant to the kernel.
        TcCudaCompiler.Result result = compile(new ConfiguredUdf(16), List.of(DataTypes.LongType), DataTypes.IntegerType).result();
        assertNull(result.refusal(), result::refusal);
        assertTrue(result.source().contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> result.source());
    }

    @Test
    void differentFieldValuesProduceDifferentKernels() {
        // The danger folding introduces: two instances of one class that compute different things.
        // The identity is taken from the folded bytes, so they cannot share a cache entry.
        ResolvedUdf eight = ConstantFieldFolding.fold(UdfResolver.resolve(new ConfiguredUdf(8)).orElseThrow());
        ResolvedUdf sixteen = ConstantFieldFolding.fold(UdfResolver.resolve(new ConfiguredUdf(16)).orElseThrow());
        List<InputSpec> inputs = List.of(InputSpec.scalar(PrimitiveKind.LONG));
        assertNotEquals(KernelIdentity.of(eight, inputs, PrimitiveKind.INT), KernelIdentity.of(sixteen, inputs, PrimitiveKind.INT),
                "two field values that compute different results must not share a kernel");
    }

    @Test
    void aNonFinalFieldIsNotFolded() {
        // A mutable field can change after planning, so reading it now would be a guess.
        ResolvedUdf resolved = UdfResolver.resolve(new MutableUdf()).orElseThrow();
        assertSame(resolved, ConstantFieldFolding.fold(resolved), "a non-final field must be left for the analysis to refuse");
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.LongType), DataTypes.IntegerType, false, true);
        assertTrue(!eligibility.isEligible(), "a UDF reading mutable state must still be refused");
    }

    /** Mutable state: must not be folded, and must still be refused. */
    public static final class MutableUdf implements UDF1<Long, Integer> {
        private int calls;

        @Override
        public Integer call(Long seed) {
            calls++;
            return calls;
        }
    }

    @Test
    void everyAdmittedJdkStaticActuallyCompiles() {
        // The allow-list is a claim about specific java.base bodies, so each one is compiled here.
        // A method that stopped compiling would otherwise be found by NVRTC on an executor, where
        // there is no way back to the CPU.
        List<String> failures = new java.util.ArrayList<>();
        for (String signature : UdfEligibilityAnalyzer.supportedJdkStatics()) {
            String owner = signature.substring(0, signature.indexOf('.'));
            String name = signature.substring(signature.indexOf('.') + 1, signature.indexOf('('));
            String descriptor = signature.substring(signature.indexOf('('));
            String result = JdkStaticProbe.compile(owner, name, descriptor);
            if (result != null) {
                failures.add(signature + ": " + result);
            }
        }
        assertTrue(failures.isEmpty(), () -> "these admitted methods did not compile:\n" + String.join("\n", failures));
    }

    private static String assertThrowsIllegalArgument(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException e) {
            return e.getMessage();
        }
        throw new AssertionError("expected an IllegalArgumentException");
    }
}
