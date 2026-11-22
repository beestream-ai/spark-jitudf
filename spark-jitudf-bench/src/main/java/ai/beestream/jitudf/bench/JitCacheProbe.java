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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.OutputNullability;

import uk.ac.manchester.tornado.tc.frontend.Frontend;
import uk.ac.manchester.tornado.tc.ir.target.TargetCaps;
import uk.ac.manchester.tornado.tc.runtime.TcCompiler;
import uk.ac.manchester.tornado.tc.runtime.TcOptions;

/**
 * What a UDF costs before it computes anything: TC on the driver, and NVRTC inside cuDF.
 *
 * <p>
 * NVIDIA's Appendix A reports roughly 1.8 s for the first {@code transform} in a process (cuDF's
 * JIT initialisation included), about 0.3 s for each further distinct UDF, and 13-52 ms once
 * {@code LIBCUDF_KERNEL_CACHE_PATH} is warm. Our own previously reported ~38 ms is a different
 * quantity entirely -- it is TC turning bytecode into CUDA on the driver, and says nothing about
 * what the executor then pays. This separates them.
 * </p>
 *
 * <p>
 * One fact worth knowing before reading any of these numbers: cuDF uses a disk cache <em>whether
 * or not</em> {@code LIBCUDF_KERNEL_CACHE_PATH} is set. {@code cpp/src/runtime/context.cpp} of the
 * pinned revision resolves, in order, {@code $LIBCUDF_KERNEL_CACHE_PATH}, then
 * {@code $XDG_CACHE_HOME/libcudf}, then {@code $HOME/.cache/libcudf}, then {@code $TMPDIR/libcudf},
 * then {@code /tmp/libcudf}. So a run that sets nothing is not a cold run; it is a run using
 * {@code ~/.cache/libcudf}, which is warm after the first time anyone used it. Every scenario here
 * therefore points the variable at a directory this experiment owns.
 * </p>
 *
 * <p>
 * Each distinct UDF is the same function with a different constant, so the CUDA differs and the
 * cache key with it, while the work per row does not. Timings are of the {@code transform} call,
 * which is compile plus launch plus synchronise; the execution component is estimated from the
 * repeat calls, where nothing is compiled.
 * </p>
 */
public final class JitCacheProbe {

    private JitCacheProbe() {
    }

    /** k is folded in as a literal, so each k compiles to different CUDA and caches separately. */
    private static String cudaFor(int k) {
        return "__device__ void GENERIC_TRANSFORM_OP(long long *out, long long a) {"
                + " long long acc = a; for (int i = 0; i < " + k + "; i++) { acc = acc * 31 + i; } *out = acc; }";
    }

    public static void main(String[] args) throws Exception {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 1_000_000;
        int distinct = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        int repeats = args.length > 2 ? Integer.parseInt(args[2]) : 3;

        System.out.println("JITPROBE cache_path=" + String.valueOf(System.getenv("LIBCUDF_KERNEL_CACHE_PATH")));
        System.out.println("JITPROBE rows=" + rows + " distinct=" + distinct + " repeats=" + repeats);

        long[] data = new long[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = i;
        }

        // Driver-side TC compilation, measured on its own so it is never confused with NVRTC's.
        String jar = new java.io.File(JitCacheProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getPath();
        var source = Frontend.BytecodeSource.classpath(jar);
        long tcStart = System.nanoTime();
        var compiled = TcCompiler.compileDeviceFunction(source, FpFidelityProbe.class.getName().replace('.', '/'),
                "onlyArithmetic", "(D)D", TargetCaps.CUDA_SM89, TcOptions.DEFAULTS);
        long tcMs = (System.nanoTime() - tcStart) / 1_000_000;
        System.out.println("JITPROBE tc_driver_compile_ms=" + tcMs + " succeeded=" + compiled.succeeded());

        try (ColumnVector in = ColumnVector.fromLongs(data)) {
            for (int u = 0; u < distinct; u++) {
                // k differs per UDF so the generated source, and therefore the cache entry, differs.
                String cuda = cudaFor(10 + u);
                for (int r = 0; r < repeats; r++) {
                    long t0 = System.nanoTime();
                    try (ColumnVector out = ColumnView.transform(new ColumnView[] { in }, cuda, false,
                            DType.INT64, false, OutputNullability.PRESERVE)) {
                        // Reading one value forces the launch to have completed.
                        try (var host = out.copyToHost()) {
                            host.getLong(0);
                        }
                    }
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    System.out.println("JITPROBE udf=" + u + " call=" + r + " ms=" + ms);
                }
            }
        }
        System.out.println("JITPROBE done");
    }
}
