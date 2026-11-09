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
import java.util.ArrayList;
import java.util.List;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;

import com.nvidia.spark.RapidsUDF;

/**
 * The executor side: a {@link RapidsUDF} that runs the generated TornadoVM kernel over the cuDF
 * columns RAPIDS hands it and gives back a cuDF column.
 *
 * <p>
 * RAPIDS's {@code GpuProjectExec} calls this once per batch through
 * {@code GpuUserDefinedFunction.columnarEval}. Everything it needs travelled here as the
 * {@link KernelDescriptor} the driver produced, so no GPU was required to plan the query and no
 * device pointer was serialised; the addresses only exist inside this method.
 * </p>
 *
 * <p>
 * What the method does not do is as important as what it does. It does not copy the inputs: they
 * are bound where they are. It does not close them: they belong to the caller. It does not retry on
 * the CPU: a failure here fails the query, which is why the decision to come here at all was made
 * conservatively at planning time. And it does not free the output buffer on the success path,
 * because the column returned owns it from then on.
 * </p>
 */
public final class TornadoRapidsUdf implements RapidsUDF, Serializable {

    private static final long serialVersionUID = 1L;

    private final KernelDescriptor descriptor;

    /** Per-phase nanoseconds of the most recent batch, for the benchmark breakdown. */
    private transient volatile Timings lastTimings;

    public TornadoRapidsUdf(KernelDescriptor descriptor) {
        this.descriptor = descriptor;
    }

    /**
     * Nanoseconds spent in each phase of one batch.
     *
     * <p>
     * {@code planLookupNanos} is kept apart from {@code bindNanos} because the two are very
     * different costs that happen next to each other: looking the plan up is a map read on every
     * batch but builds a task graph on the first, while binding is the per-batch work this design
     * exists to make cheap. Reporting them together would hide which one a number belongs to.
     * </p>
     */
    public record Timings(long columnAccessNanos, long outputAllocNanos, long planLookupNanos, long bindNanos, long inputSyncNanos, long executeNanos, long outputSyncNanos, long wrapNanos, long tornadoCompilerNanos, long driverInstallNanos, long dispatchNanos,
            int numRows, boolean firstBatchForPlan, long copyInBytes, long copyOutBytes, long kernelNanos, long inputBytesBound, long outputBytesBound) {
        public long totalNanos() {
            return columnAccessNanos + outputAllocNanos + planLookupNanos + bindNanos + inputSyncNanos + executeNanos + outputSyncNanos + wrapNanos;
        }
    }

    public KernelDescriptor descriptor() {
        return descriptor;
    }

    public Timings lastTimings() {
        return lastTimings;
    }

    /**
     * Times the whole call, including the guard before the phases and the cleanup after them.
     *
     * <p>
     * The eight phase timers run from t0 to t8 and are contiguous, so they sum to the time between
     * those two points exactly. They do not cover the argument checks before t0 or the unbind and
     * close in the {@code finally} after t8. For an end-to-end attribution that has to add up,
     * what matters is the time the caller actually loses to this method, so that is measured here
     * and the difference from the phase sum is reported rather than left as a gap.
     * </p>
     */
    /**
     * Whether this JVM collects host phase timings at all.
     *
     * <p>
     * Measuring is not free, and until this flag existed there was no way to run the integration
     * without it: eight {@code nanoTime} calls per batch, a ninth around the whole call, thirteen
     * atomic accumulations and a synchronized sample list ran on every batch whatever the
     * configuration said. That made "what does the integration cost" unanswerable, because every
     * available measurement already included the cost of measuring.
     * </p>
     *
     * <p>
     * {@code static final} read from a system property, so when it is false the JIT folds the
     * timing away entirely rather than branching on it per batch. It gates only observation:
     * nothing about the kernel, the plan cache, the bindings or the synchronization changes with
     * it, which is what makes an instrumented run comparable to an uninstrumented one.
     * </p>
     */
    static final boolean TIMING = Boolean.getBoolean("tornado.spark.metrics");

    /** {@link System#nanoTime()} when timing is on, and a constant the JIT can fold when it is not. */
    private static long now() {
        return TIMING ? System.nanoTime() : 0L;
    }

    @Override
    public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) {
        if (!TIMING && !IntervalLog.ENABLED) {
            return evaluateColumnarInner(numRows, args);
        }
        long entry = System.nanoTime();
        try {
            return evaluateColumnarInner(numRows, args);
        } finally {
            long exit = System.nanoTime();
            if (TIMING) {
                TornadoUdfMetrics.recordCall(exit - entry);
            }
            // The call's own bounds, so that "calls never overlap each other, and no compiler
            // phase runs inside one" can be checked rather than asserted.
            IntervalLog.record("call", entry, exit);
        }
    }

    private ColumnVector evaluateColumnarInner(int numRows, ColumnVector... args) {
        if (args.length != descriptor.inputs().size()) {
            throw new IllegalStateException("the kernel for " + descriptor.udfName() + " takes " + descriptor.inputs().size() + " columns but was given " + args.length);
        }
        if (numRows == 0) {
            // An RMM allocation of zero bytes is not meaningful and the kernel has nothing to do;
            // an empty batch has an empty answer. The scalar is only a type carrier and is closed
            // here, since nothing else would.
            try (ai.rapids.cudf.Scalar zero = scalarZero()) {
                return ColumnVector.fromScalar(zero, 0);
            }
        }

        long t0 = now();
        // One binding per kernel input view: a fixed-width column gives one, a LIST column gives
        // its offsets and then its flattened child.
        List<CudfBufferAdapter.InputBinding> inputs = new ArrayList<>(args.length);
        long listElements = 0;
        for (int i = 0; i < args.length; i++) {
            InputSpec spec = descriptor.inputs().get(i);
            if (spec.isList()) {
                CudfBufferAdapter.ListBinding list = CudfBufferAdapter.describeList(args[i], spec.kind(), numRows);
                inputs.add(list.offsets());
                inputs.add(list.values());
                listElements = Math.max(listElements, list.values().rows());
            } else {
                inputs.add(CudfBufferAdapter.describe(args[i], spec.kind(), numRows));
            }
        }
        long t1 = now();

        DeviceMemoryBuffer output = CudfBufferAdapter.allocateOutput(numRows, descriptor.outputKind());
        long t2 = now();
        boolean handedOver = false;
        // A LIST whose rows are all empty has no child payload. The kernel still takes a pointer
        // for it, so it gets one this method owns and closes, rather than a borrowed address that
        // does not exist. No row can read through it: every row's start equals its end.
        List<DeviceMemoryBuffer> sentinels = new ArrayList<>(1);
        KernelExecution execution = null;
        try {
            for (int i = 0; i < inputs.size(); i++) {
                if (inputs.get(i).isEmpty()) {
                    DeviceMemoryBuffer sentinel = CudfBufferAdapter.allocateOutput(1, inputs.get(i).kind());
                    sentinels.add(sentinel);
                    inputs.set(i, new CudfBufferAdapter.InputBinding(sentinel.getAddress(), sentinel.getLength(), 1, inputs.get(i).kind()));
                }
            }
            long plansBefore = TornadoKernelCache.plansBuilt();
            // A capacity of zero would ask for a zero-length device view. One element is the
            // smallest view there is, and it matches the sentinel bound above.
            execution = TornadoKernelCache.execution(descriptor, numRows, Math.max(listElements, 1));
            boolean firstForThisPlan = TornadoKernelCache.plansBuilt() != plansBefore;
            long t3 = now();

            execution.bind(inputs, output.getAddress(), output.getLength());
            long t4 = now();

            // cuDF produced these columns on its own stream; wait for that before reading them.
            CudfBufferAdapter.synchronizeBeforeKernel();
            long t5 = now();

            execution.execute();
            long t6 = now();

            // And make the kernel's writes visible to whichever stream reads the result next.
            CudfBufferAdapter.synchronizeAfterKernel();
            long t7 = now();

            ColumnVector result = CudfBufferAdapter.wrapAsColumn(output, numRows, descriptor.outputKind());
            handedOver = true;
            long t8 = now();

            // Summing binding lengths, building the record and accumulating it are all
            // observation, so they belong behind the same flag as the timers themselves.
            if (TIMING) {
                long inputBytes = 0;
                for (CudfBufferAdapter.InputBinding input : inputs) {
                    inputBytes += input.byteLength();
                }
                lastTimings = new Timings(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, execution.lastTornadoCompilerNanos(),
                        execution.lastDriverInstallNanos(), execution.lastDispatchNanos(), numRows, firstForThisPlan, execution.lastCopyInBytes(),
                        execution.lastCopyOutBytes(), execution.lastKernelNanos(), inputBytes, output.getLength());
                TornadoUdfMetrics.record(lastTimings);
            }
            return result;
        } finally {
            // Borrowed addresses are valid for exactly one batch. Leaving them bound means a cached
            // view still points at memory the caller is about to close, which is a dangling pointer
            // waiting for a later batch to fail before rebinding. Releasing them here, on success
            // and on failure alike, keeps the borrow scoped to the work that borrowed it.
            if (execution != null) {
                try {
                    execution.unbind();
                } catch (RuntimeException e) {
                    // Nothing here should fail, but a failure to unbind must not replace the
                    // original exception on the way out.
                }
            }
            if (!handedOver) {
                output.close();
            }
            for (DeviceMemoryBuffer sentinel : sentinels) {
                sentinel.close();
            }
        }
    }

    private ai.rapids.cudf.Scalar scalarZero() {
        return switch (descriptor.outputKind()) {
            case INT -> ai.rapids.cudf.Scalar.fromInt(0);
            case LONG -> ai.rapids.cudf.Scalar.fromLong(0L);
            case FLOAT -> ai.rapids.cudf.Scalar.fromFloat(0f);
            case DOUBLE -> ai.rapids.cudf.Scalar.fromDouble(0d);
        };
    }

    @Override
    public String toString() {
        return "TornadoRapidsUdf[" + descriptor + "]";
    }
}
