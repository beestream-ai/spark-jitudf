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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

/**
 * A fold over a list column, compiled for cuDF without a segmented transform entry point.
 *
 * <p>
 * The claim under test is that NVIDIA's one near-fit customer pattern -- per-day activity flags
 * packed into an {@code array<bigint>}, counted per row -- is reachable with the shipped cuDF API,
 * by transforming the list's child and reducing over the offsets. Both their review and this
 * project's own analysis had it blocked on new libcudf code.
 * </p>
 */
class ListReductionTest {

    /** The customer pattern: count set days across a row's packed bitmaps. */
    public static final class ActiveDaysUdf implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += Long.bitCount(arr[i]);
            }
            return acc;
        }
    }

    /** Not a fold: the result depends on position, so it must be refused rather than guessed. */
    public static final class PositionalUdf implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i++) {
                acc += Long.bitCount(arr[i]) * i;
            }
            return acc;
        }
    }

    private static TornadoUdfProvider.Decision ignored;

    private static Object compile(Object udf, String name) {
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        try {
            return TornadoUdfProvider.tryCompile(udf,
                    new org.apache.spark.sql.types.DataType[] { DataTypes.createArrayType(DataTypes.LongType, false) },
                    DataTypes.IntegerType, false, true, name, new String[] { "LOOP" }, "a loop");
        } finally {
            if (previous == null) {
                System.clearProperty("tornado.spark.backend");
            } else {
                System.setProperty("tornado.spark.backend", previous);
            }
        }
    }

    @Test
    void aFoldOverAListCompilesWithoutASegmentedTransform() {
        Object udf = compile(new ActiveDaysUdf(), "activeDays");
        assertNotNull(udf, () -> "refused: " + TornadoUdfProvider.decisions().get("activeDays"));
        assertTrue(udf instanceof CudfListReductionUdf, () -> "got " + udf.getClass());
        String cuda = ((CudfListReductionUdf) udf).cudaSource();
        // The element function, not a per-row kernel: it takes one value, not a pointer and a range.
        assertTrue(cuda.contains("__device__ void GENERIC_TRANSFORM_OP(int *"), () -> cuda);
        assertTrue(cuda.contains("long long"), () -> "the element is a long: " + cuda);
        assertTrue(!cuda.contains("while") && !cuda.contains("for"), () -> "the loop belongs to the reduction, not the kernel: " + cuda);
    }

    @Test
    void theSplitIsRefusedWhenTheResultDependsOnPosition() {
        // acc += arr[i] * i is not a fold over independent elements, so no element function exists.
        assertNull(compile(new PositionalUdf(), "positional"));
        String decision = TornadoUdfProvider.decisions().get("positional");
        assertTrue(decision != null && decision.contains("not a sum over its elements"), () -> "reason was: " + decision);
    }

    @Test
    void theExtractedElementFunctionComputesWhatTheLoopBodyDid() throws Exception {
        // Run the extracted element function on the CPU and compare with the original UDF, so that
        // "the split is correct" is checked rather than assumed.
        ResolvedUdf resolved = ConstantFieldFolding.fold(UdfResolver.resolve(new ActiveDaysUdf()).orElseThrow());
        List<InputSpec> inputs = List.of(InputSpec.list(PrimitiveKind.LONG));
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved,
                List.of(DataTypes.createArrayType(DataTypes.LongType, false)), DataTypes.IntegerType, false, true);
        assertTrue(eligibility.isEligible(), eligibility::reason);

        org.objectweb.asm.tree.ClassNode node = new org.objectweb.asm.tree.ClassNode();
        new org.objectweb.asm.ClassReader(resolved.classFile()).accept(node, org.objectweb.asm.ClassReader.SKIP_FRAMES);
        org.objectweb.asm.tree.MethodNode user = node.methods.stream()
                .filter(m -> m.name.equals("call") && m.desc.equals(org.objectweb.asm.Type.getMethodDescriptor(resolved.method())))
                .findFirst().orElseThrow();
        org.objectweb.asm.tree.MethodNode body = AsmUdfLowering.lower(user, "udfBody", inputs, PrimitiveKind.INT, false);
        var split = ListReductionSplit.split(body, PrimitiveKind.LONG, PrimitiveKind.INT);
        assertTrue(split.isPresent(), "the fold should split");

        // The element function here is `(int) x`; summing it must equal the original UDF.
        long[] row = { 0b1011L, 0xffL, 1L, 0L };
        int expected = new ActiveDaysUdf().call(row);
        int sum = 0;
        for (long v : row) {
            sum += Long.bitCount(v);
        }
        assertEquals(expected, sum, "the reduction of the element function must equal the UDF");
    }
}
