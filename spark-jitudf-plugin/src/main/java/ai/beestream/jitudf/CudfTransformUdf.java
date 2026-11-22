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

import java.io.Serializable;

import ai.rapids.cudf.ColumnVector;

import com.nvidia.spark.RapidsUDF;

/**
 * The executor side of the NVIDIA-aligned architecture: a {@link RapidsUDF} that hands cuDF a
 * string.
 *
 * <p>
 * Compare {@link TornadoRapidsUdf}, which this replaces. That one borrows device pointers out of
 * cuDF's columns, allocates through RMM, synchronises the whole device on both sides of the call
 * and drives a TornadoVM execution plan. This one does none of that, because it does not need to:
 * cuDF compiles the source with NVRTC, allocates the output through its own memory resource,
 * launches on its own stream and owns every pointer involved. The three objections NVIDIA raised to
 * running TornadoVM inside executors -- buffers it cannot see, allocations outside RMM, and streams
 * ordered against nothing -- do not have anywhere to arise.
 * </p>
 *
 * <p>
 * What crosses from the driver is the CUDA source and the output type. No device pointer, no
 * bytecode, and nothing that needs TornadoVM on this JVM.
 * </p>
 *
 * <p>
 * <strong>Arity.</strong> cuDF's Java binding currently takes exactly one input column and always
 * produces INT32 ({@code ColumnView.transform}); libcudf itself has taken several inputs and a
 * chosen output type for some time, and widening the binding is the small JNI change NVIDIA
 * describes as its own opportunity. Until that lands, a UDF outside this shape is refused while
 * planning rather than failed here.
 * </p>
 */
public final class CudfTransformUdf implements RapidsUDF, Serializable {

    private static final long serialVersionUID = 1L;

    private final String cudaSource;
    private final String udfName;
    private final PrimitiveKind outputKind;
    private final int inputCount;

    public CudfTransformUdf(String cudaSource, String udfName, PrimitiveKind outputKind, int inputCount) {
        this.cudaSource = cudaSource;
        this.udfName = udfName;
        this.outputKind = outputKind;
        this.inputCount = inputCount;
    }

    public String cudaSource() {
        return cudaSource;
    }

    @Override
    public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) {
        if (args.length != inputCount) {
            throw new IllegalStateException("the kernel for " + udfName + " takes " + inputCount + " columns but was given " + args.length);
        }
        if (args.length != 1) {
            // Guarded at planning time; repeated here so that a future change to the planner cannot
            // quietly produce a call this binding would mis-execute.
            throw new IllegalStateException("cudf's transform binding takes one input column, but " + udfName + " has " + args.length);
        }
        if (outputKind != PrimitiveKind.INT) {
            throw new IllegalStateException("cudf's transform binding produces INT32, but " + udfName + " returns " + outputKind.cudfType());
        }
        // The input is not closed here: it belongs to the caller. The result does not need closing
        // either, because it is what this method returns.
        return args[0].transform(cudaSource, false);
    }

    @Override
    public String toString() {
        return "CudfTransformUdf[" + udfName + ", " + inputCount + " inputs -> " + outputKind.cudfType() + "]";
    }
}
