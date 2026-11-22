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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.OutputNullability;

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
 * <strong>Arity, output type and nulls.</strong> This calls the generalized binding --
 * {@code ColumnView.transform(ColumnView[], String, boolean, DType, boolean, OutputNullability)} --
 * which takes any number of input columns, produces the type asked for, and can run the function
 * over nullable inputs. That is NVIDIA's opportunity #2, and libcudf had all three of those
 * capabilities already: what was missing was only the Java and JNI surface to reach them.
 * </p>
 *
 * <p>
 * The output nullability is always {@code PRESERVE}, never {@code ALL_VALID}. {@code ALL_VALID}
 * asserts that the function produces a value for every row, and for a null-aware function this
 * integration has no way to establish that -- it would have to prove the user's method never
 * returns null. {@code PRESERVE} costs a validity mask that may turn out to be all ones; being
 * wrong the other way is undefined behaviour.
 * </p>
 */
public final class CudfTransformUdf implements RapidsUDF, Serializable {

    private static final long serialVersionUID = 1L;

    private final String cudaSource;
    private final String udfName;
    private final PrimitiveKind outputKind;
    private final int inputCount;
    private final boolean nullAware;

    public CudfTransformUdf(String cudaSource, String udfName, PrimitiveKind outputKind, int inputCount) {
        this(cudaSource, udfName, outputKind, inputCount, false);
    }

    public CudfTransformUdf(String cudaSource, String udfName, PrimitiveKind outputKind, int inputCount, boolean nullAware) {
        this.cudaSource = cudaSource;
        this.udfName = udfName;
        this.outputKind = outputKind;
        this.inputCount = inputCount;
        this.nullAware = nullAware;
    }

    public String cudaSource() {
        return cudaSource;
    }

    /** Whether the device function takes and returns {@code cuda::std::optional}s. */
    public boolean nullAware() {
        return nullAware;
    }

    @Override
    public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) {
        if (args.length != inputCount) {
            // The source was compiled for exactly this many parameters, and cudf passes one value
            // per column it is given, so a mismatch would bind arguments to the wrong parameters
            // rather than fail. Checked here because the compilation and the call are far apart.
            throw new IllegalStateException("the kernel for " + udfName + " takes " + inputCount + " columns but was given " + args.length);
        }
        // The inputs are not closed here: they belong to the caller. The result does not need
        // closing either, because it is what this method returns.
        return ColumnView.transform(args, cudaSource, false, outputKind.cudfType(), nullAware, OutputNullability.PRESERVE);
    }

    @Override
    public String toString() {
        return "CudfTransformUdf[" + udfName + ", " + inputCount + " inputs -> " + outputKind.cudfType() + (nullAware ? ", null-aware" : "") + "]";
    }
}
