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

import java.nio.charset.StandardCharsets;

import ai.rapids.cudf.ColumnVector;

import uk.ac.manchester.tornado.tc.frontend.Frontend;
import uk.ac.manchester.tornado.tc.ir.target.TargetCaps;
import uk.ac.manchester.tornado.tc.runtime.TcCompiler;
import uk.ac.manchester.tornado.tc.runtime.TcOptions;

/**
 * Whether a STRING column can reach {@code cudf::transform} through the binding we already have.
 *
 * <p>
 * This is a question about three layers at once, and only the middle one was ever in doubt. TC
 * compiles a {@code byte[]} parameter into a {@code cudf::string_view} and reads it through
 * {@code data()} and {@code size_bytes()}. libcudf's transform accepts a {@code string_view} column
 * ({@code cpp/src/transform/transform.cu} of release/26.08 constrains inputs to
 * {@code is_fixed_width<T>() || std::same_as<T, cudf::string_view>}). And the JNI entry point hands
 * the column through without inspecting its type, fixing only the output at INT32
 * ({@code ColumnViewJni.cpp:1549}). So STRING to INT32 should already work, with no binding change
 * -- and this runs it rather than arguing it.
 * </p>
 *
 * <p>
 * What this does NOT show is a transparent java {@code UDF1<String, Integer>}. That needs the
 * plugin to turn {@code String} operations into operations on UTF-8 bytes, and Java's String is
 * UTF-16: {@code length()} counts chars, not bytes, and {@code charAt} indexes code units. The
 * device function here is written against {@code byte[]} on purpose, so it says only what it can.
 * </p>
 */
public final class StringTransformProbe {

    private StringTransformProbe() {
    }

    /** FNV-1a over the row's bytes: no String semantics, just the bytes cudf hands over. */
    public static int fnv1a(byte[] s) {
        int h = 0x811c9dc5;
        for (int i = 0; i < s.length; i++) {
            h = (h ^ (s[i] & 0xFF)) * 0x01000193;
        }
        return h;
    }

    public static void main(String[] args) throws Exception {
        int checks = 0;
        int failures = 0;

        // 1. Compile the byte[] method into a cudf device function, with TC and nothing else.
        // Under spark-submit the application jar is not on java.class.path, so ask the class itself.
        String jar = new java.io.File(StringTransformProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getPath();
        var source = Frontend.BytecodeSource.classpath(jar);
        var target = TargetCaps.CUDA_SM89;
        var compilation = TcCompiler.compileDeviceFunction(source, StringTransformProbe.class.getName().replace('.', '/'), "fnv1a", "([B)I", target, TcOptions.DEFAULTS);
        if (!compilation.succeeded()) {
            System.out.println("FAIL  TC refused the byte[] method: " + compilation.diagnostics());
            System.exit(1);
        }
        String cuda = compilation.source().replaceAll("(?<![A-Za-z0-9_])" + java.util.regex.Pattern.quote(compilation.entryName()) + "(?![A-Za-z0-9_])", "GENERIC_TRANSFORM_OP");
        System.out.println("=== generated device function ===");
        System.out.println(cuda);

        checks++;
        boolean takesStringView = cuda.contains("cudf::string_view");
        if (!takesStringView) {
            failures++;
        }
        System.out.println((takesStringView ? "  PASS" : "  FAIL") + "  the parameter is a cudf::string_view");

        // 2. Run it over a STRING column through the binding that exists today.
        String[] rows = { "", "a", "spark", "tornadovm", "a longer string with spaces", "éèê", "0123456789" };
        try (ColumnVector in = ColumnVector.fromStrings(rows); ColumnVector out = in.transform(cuda, false)) {
            try (var host = out.copyToHost()) {
                for (int i = 0; i < rows.length; i++) {
                    int want = fnv1a(rows[i].getBytes(StandardCharsets.UTF_8));
                    int got = host.getInt(i);
                    checks++;
                    boolean ok = want == got;
                    if (!ok) {
                        failures++;
                    }
                    System.out.println((ok ? "  PASS" : "  FAIL") + "  row " + i + " \"" + rows[i] + "\": gpu=" + got + " java=" + want);
                }
            }
        } catch (Exception e) {
            checks++;
            failures++;
            System.out.println("  FAIL  transform on a STRING column threw: " + e);
        }

        System.out.println();
        System.out.println("=== STRING PROBE CHECKS: " + checks + ", FAILURES: " + failures + " ===");
        System.exit(failures == 0 ? 0 : 1);
    }
}
