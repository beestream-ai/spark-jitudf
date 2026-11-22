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

import java.util.List;

/**
 * Checks, on the driver, that generated CUDA is something {@code cudf::transform} will actually
 * bind and run.
 *
 * <p>
 * This exists because of a constraint in libcudf that is easy to miss and expensive to discover
 * late. {@code cudf::jit::parse_single_function_cuda} does not look for a function by name: it
 * scans for the <em>first</em> {@code (} in the whole source string, walks back to the identifier
 * in front of it, and renames that identifier to {@code GENERIC_TRANSFORM_OP}. So a source file
 * carrying a helper definition, a forward declaration, or any parenthesised preamble ahead of the
 * entry does not fail -- it silently binds the wrong function.
 * </p>
 *
 * <p>
 * The other half of the job is the division of labour between driver and executor. Everything the
 * driver gets wrong here would otherwise be found by NVRTC inside a Spark task, where there is no
 * way back: a {@code RapidsUDF} has no channel to decline once planning has chosen it, so the
 * query fails. Refusing on the driver turns that into an ordinary planning-time refusal, which is
 * the only safe kind.
 * </p>
 *
 * <p>
 * What it cannot do is replace NVRTC. It checks shape, not semantics, so it is a filter in front of
 * the executor rather than a proof that the source compiles.
 * </p>
 */
public final class CudaSourceCheck {

    /** Constructs that only make sense inside a launched kernel, never inside a transform UDF. */
    private static final List<String> FORBIDDEN = List.of(
            "threadIdx", "blockIdx", "blockDim", "gridDim", "__shared__", "__syncthreads", "atomicAdd");

    private CudaSourceCheck() {
    }

    /**
     * Returns null when {@code source} is usable, or the reason it is not.
     *
     * @param source
     *         the CUDA that would be shipped to executors.
     * @param entryName
     *         the function TC was asked to emit.
     * @param inputs
     *         how many values cuDF will pass after the output pointer.
     */
    public static String check(String source, String entryName, int inputs) {
        if (source == null || source.isBlank()) {
            return "the compiler produced no source";
        }
        String stripped = stripComments(source);

        int firstParen = stripped.indexOf('(');
        if (firstParen < 0) {
            return "the generated source declares no function";
        }
        // Exactly what libcudf does, so that what is checked is what will be bound.
        int end = firstParen - 1;
        while (end >= 0 && Character.isWhitespace(stripped.charAt(end))) {
            end--;
        }
        int start = end;
        while (start >= 0 && !Character.isWhitespace(stripped.charAt(start))) {
            start--;
        }
        String bound = stripped.substring(start + 1, end + 1);
        if (!bound.equals(entryName)) {
            return "cudf binds the first function in the source, which here is '" + bound + "' and not the entry '" + entryName
                    + "'. The generated code needs a helper or a declaration ahead of the entry, which this path cannot ship.";
        }

        if (stripped.contains("__global__")) {
            return "the generated source declares a __global__ kernel; a transform UDF is a device function, because cudf owns the launch";
        }

        long definitions = countDefinitions(stripped);
        if (definitions != 1) {
            return "the generated source defines " + definitions + " functions, and cudf::transform binds only the first one";
        }

        for (String forbidden : FORBIDDEN) {
            if (stripped.contains(forbidden)) {
                return "the generated source uses '" + forbidden + "', which a transform UDF cannot: cudf owns the launch";
            }
        }

        // cudf calls f(out_ptr, in0..inN-1): one pointer parameter, then one value per input.
        int close = stripped.indexOf(')', firstParen);
        if (close < 0) {
            return "the entry's parameter list is not closed";
        }
        String parameters = stripped.substring(firstParen + 1, close).trim();
        int declared = parameters.isEmpty() ? 0 : parameters.split(",").length;
        if (declared != inputs + 1) {
            return "the entry takes " + declared + " parameters but cudf will pass " + (inputs + 1) + " (one output pointer and " + inputs + " inputs)";
        }
        if (!parameters.contains("*")) {
            return "the entry's first parameter is not a pointer, so cudf's output argument would not bind";
        }
        return null;
    }

    /**
     * Functions in the source, counted by their CUDA qualifier.
     *
     * <p>
     * Counting braces instead would be wrong in a way that is easy to miss: {@code while (x) &#123;}
     * and {@code if (x) &#123;} both look like a parenthesis followed by an opening brace, so a UDF
     * with a loop would be read as several functions. Every function TC emits carries
     * {@code __device__} or {@code __global__}, including the helpers in its preamble, which are
     * exactly the ones this is looking for.
     * </p>
     */
    private static long countDefinitions(String source) {
        return occurrences(source, "__device__") + occurrences(source, "__global__");
    }

    private static long occurrences(String source, String token) {
        long count = 0;
        for (int i = source.indexOf(token); i >= 0; i = source.indexOf(token, i + token.length())) {
            count++;
        }
        return count;
    }

    /** Line and block comments removed, as libcudf's parser removes them before looking. */
    private static String stripComments(String source) {
        StringBuilder out = new StringBuilder(source.length());
        for (int i = 0; i < source.length(); ) {
            if (source.startsWith("//", i)) {
                int end = source.indexOf('\n', i);
                i = end < 0 ? source.length() : end;
            } else if (source.startsWith("/*", i)) {
                int end = source.indexOf("*/", i + 2);
                i = end < 0 ? source.length() : end + 2;
            } else {
                out.append(source.charAt(i++));
            }
        }
        return out.toString();
    }
}
