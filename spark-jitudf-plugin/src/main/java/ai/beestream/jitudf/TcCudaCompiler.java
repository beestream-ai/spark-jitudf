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

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import uk.ac.manchester.tornado.tc.ir.target.TargetCaps;
import uk.ac.manchester.tornado.tc.runtime.TcCompiler;
import uk.ac.manchester.tornado.tc.runtime.TcOptions;
import uk.ac.manchester.tornado.tc.tornado.ValueSpecialisation;

/**
 * Compiles one UDF into CUDA, on the driver, with no GPU and no TornadoVM runtime.
 *
 * <p>
 * This is the piece that makes the architecture NVIDIA asked for possible. TC reads class files
 * rather than loading classes, and depends on nothing but its own modules, so it runs in a Spark
 * driver as ordinary library code. What crosses to the executors afterwards is a string.
 * </p>
 *
 * <p>
 * What is compiled is {@link CudfDeviceFunctionGenerator#BODY_METHOD}: a static method over
 * primitives that returns the UDF's result.
 * {@link TcCompiler#compileDeviceFunction(uk.ac.manchester.tornado.tc.frontend.Frontend.BytecodeSource, String, String, String, TargetCaps, TcOptions, List)}
 * turns that into {@code __device__ void f(R *out, A0 in0, ...)}, the shape cuDF calls, deriving the
 * output pointer from the return type rather than taking it as a parameter. There are no array
 * headers and no launch parameters, and nothing of the batch size reaches the code, so one
 * compilation serves every batch.
 * </p>
 *
 * <p>
 * The emitted function is renamed to {@link #ENTRY_SYMBOL} afterwards. cuDF renames the first
 * function in the source to that name regardless; doing it here makes the rename a no-op and keeps
 * the shipped source readable. TC names the function after its class and method, which carries the
 * cache key and would otherwise be what a reader of the CUDA sees.
 * </p>
 */
public final class TcCudaCompiler {

    /**
     * The name the emitted function is given. cuDF renames the first function in the source to
     * {@code GENERIC_TRANSFORM_OP} regardless, so emitting it under that name already makes the
     * rename a no-op and keeps the shipped source readable.
     */
    public static final String ENTRY_SYMBOL = "GENERIC_TRANSFORM_OP";

    private TcCudaCompiler() {
    }

    /** CUDA source and the symbol it defines, or the reason there is none. */
    public record Result(String source, String entryName, String refusal) {
        public boolean ok() {
            return refusal == null;
        }

        static Result refused(String reason) {
            return new Result(null, null, reason);
        }
    }

    /**
     * Compiles {@code generated}'s entry method into a cuDF-compatible device function.
     *
     * @param generated
     *         the entry class {@link CudfDeviceFunctionGenerator} built.
     * @param inputs
     *         the UDF's arguments, used to size the fact list and to check the emitted signature.
     * @param userLoader
     *         the loader that defined the user's UDF, for its helpers' bytes.
     */
    public static Result compile(CudfDeviceFunctionGenerator.Generated generated, List<InputSpec> inputs, ClassLoader userLoader) {
        return compile(generated, inputs, userLoader, false);
    }

    /**
     * As above; {@code reportArithmeticErrors} compiles with {@code TcOptions.withDeviceFunctionErrors},
     * which makes the device function return a {@code cudf::errc} rather than nothing -- 1 for an
     * overflow in {@code Math.*Exact}, 2 for an integer division by zero.
     *
     * <p>
     * This needs no change to the transform binding. cudf's JIT kernel decides at compile time what
     * to do with the result, in {@code cpp/src/transform/jit/kernel.cu} of release/26.08:
     * </p>
     *
     * <pre>
     * if constexpr (!cuda::std::is_void_v&lt;decltype(GENERIC_TRANSFORM_OP(a...))&gt;) {
     *   return static_cast&lt;cudf::errc&gt;(GENERIC_TRANSFORM_OP(a...));
     * } else {
     *   (void)GENERIC_TRANSFORM_OP(a...);
     *   return errc::SUCCESS;
     * }
     * </pre>
     *
     * <p>
     * The per-row codes are reduced with {@code max} into a per-thread value and then atomically into
     * one per launch, which the host turns into {@code cudf::evaluation_error} for the call. So a UDF
     * that divides by zero on any row fails the query, as Java's {@code ArithmeticException} would,
     * instead of writing a wrong number.
     * </p>
     */
    public static Result compile(CudfDeviceFunctionGenerator.Generated generated, List<InputSpec> inputs, ClassLoader userLoader, boolean reportArithmeticErrors) {
        CompositeBytecodeSource source = new CompositeBytecodeSource(userLoader).with(generated.internalName(), generated.classFile());

        // One fact per parameter of the compiled method, which is one per UDF input: nothing is assumed
        // about any of them, since they are plain values that differ per row. The output is not among
        // them -- TC derives the pointer cudf passes from the return type.
        List<ValueSpecialisation.Fact> facts = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            facts.add(new ValueSpecialisation.Fact.Unknown());
        }

        TcCompiler.Compilation compilation;
        try {
            compilation = TcCompiler.compileDeviceFunction(source, generated.className(), generated.bodyMethod(), generated.bodyDescriptor(),
                    target(), TcOptions.DEFAULTS.withDeviceFunctionErrors(reportArithmeticErrors), facts);
        } catch (RuntimeException | LinkageError e) {
            // TC reports failures as diagnostics rather than exceptions, so anything thrown here is
            // unexpected. It is still a reason to leave the query alone, not to fail it.
            return Result.refused("the CUDA compiler failed unexpectedly: " + e);
        }
        if (!compilation.succeeded()) {
            return Result.refused("TornadoVM's compiler declined the UDF: " + describe(compilation));
        }

        String cuda = renameEntry(compilation.source(), compilation.entryName());
        String refusal = CudaSourceCheck.check(cuda, ENTRY_SYMBOL, inputs.size());
        if (refusal != null) {
            return Result.refused(refusal);
        }
        return new Result(cuda, ENTRY_SYMBOL, null);
    }

    /**
     * Compiles a nullable UDF: TC's null-aware mode, for {@code cudf::transform} with
     * {@code null_aware::YES}.
     *
     * <p>
     * What changes is the whole signature. Each input arrives as a {@code cuda::std::optional}, the
     * output is written through a {@code cuda::std::optional<R> *}, and the function returns an
     * {@code int} error code rather than nothing:
     * </p>
     *
     * <pre>
     * #include &lt;cuda/std/optional&gt;
     * __device__ int GENERIC_TRANSFORM_OP(cuda::std::optional&lt;int&gt; *out, cuda::std::optional&lt;long long&gt; v_opt)
     * </pre>
     *
     * <p>
     * That is the shape libcudf's kernel calls in its null-aware branch: it reads each input with
     * {@code nullable_element}, hands the function an {@code optional} output to fill, and takes
     * the output column's validity from {@code has_value()} afterwards. So the UDF's own answer for
     * a null -- a number, or a null -- is what lands in the column, which is the only way to be
     * right about a Java UDF, whose parameters are boxed and which Spark therefore calls <em>with</em>
     * the null rather than short-circuiting it.
     * </p>
     *
     * <p>
     * A null the method unboxes without checking is Java's {@code NullPointerException}; TC emits
     * error code 3 for it. cudf only names 0, 1 and 2 ({@code SUCCESS}, {@code OVERFLOW},
     * {@code DIVISION_BY_ZERO}), so 3 is reported as {@code UNKNOWN_ERROR}: the query fails, as it
     * would on the CPU, but the exception is not reproduced.
     * </p>
     *
     * @param reportArithmeticErrors
     *         also compile with {@code --errors}, for a UDF whose divisor cannot be proved non-zero
     *         or which calls {@code Math.*Exact}. Null-awareness alone does not admit those.
     */
    public static Result compileNullAware(CudfDeviceFunctionGenerator.Generated generated, List<InputSpec> inputs, ClassLoader userLoader, boolean reportArithmeticErrors) {
        CompositeBytecodeSource source = new CompositeBytecodeSource(userLoader).with(generated.internalName(), generated.classFile());

        TcOptions options = TcOptions.DEFAULTS.withNullAwareDeviceFunction(true).withDeviceFunctionErrors(reportArithmeticErrors);

        TcCompiler.Compilation compilation;
        try {
            compilation = TcCompiler.compileDeviceFunction(source, generated.className(), generated.bodyMethod(), generated.bodyDescriptor(), target(), options);
        } catch (RuntimeException | LinkageError e) {
            return Result.refused("the CUDA compiler failed unexpectedly: " + e);
        }
        if (!compilation.succeeded()) {
            return Result.refused("TornadoVM's compiler declined the nullable UDF: " + describe(compilation));
        }

        String cuda = renameEntry(compilation.source(), compilation.entryName());
        String refusal = CudaSourceCheck.check(cuda, ENTRY_SYMBOL, inputs.size());
        if (refusal != null) {
            return Result.refused(refusal);
        }
        return new Result(cuda, ENTRY_SYMBOL, null);
    }

    /**
     * What the generated code is compiled for.
     *
     * <p>
     * The compute capability only selects which features TC will allow itself; the source is CUDA
     * C++ that NVRTC compiles on the executor for whatever device is actually there, so this does
     * not pin the code to one GPU. {@code sm_89} is chosen as a capability floor that still permits
     * the double precision Spark's numeric types need.
     * </p>
     */
    private static TargetCaps target() {
        return TargetCaps.CUDA_SM89;
    }

    /**
     * {@code source} with the function TC named {@code emitted} called {@link #ENTRY_SYMBOL} instead.
     *
     * <p>
     * Whole-word only, so a longer identifier that merely starts with the emitted name is left alone.
     * {@link CudaSourceCheck} is what establishes that the result is a single function bound under
     * that name; this only changes the name.
     * </p>
     */
    private static String renameEntry(String source, String emitted) {
        if (emitted == null || emitted.equals(ENTRY_SYMBOL)) {
            return source;
        }
        return Pattern.compile("(?<![A-Za-z0-9_])" + Pattern.quote(emitted) + "(?![A-Za-z0-9_])").matcher(source).replaceAll(Matcher.quoteReplacement(ENTRY_SYMBOL));
    }

    private static String describe(TcCompiler.Compilation compilation) {
        if (compilation.diagnostics().isEmpty()) {
            return "no reason given";
        }
        return compilation.diagnostics().stream().map(d -> d.code() + " " + d.message()).collect(Collectors.joining("; "));
    }
}
