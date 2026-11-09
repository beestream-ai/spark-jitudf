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

import java.util.Optional;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;

/**
 * The boundary between a cuDF column and a TornadoVM device view.
 *
 * <p>
 * An input column is not copied anywhere. Its data buffer already is a device allocation of the
 * right layout -- fixed-width elements, no header -- so what crosses is its address and its length,
 * and the kernel reads libcudf's bytes where libcudf put them. The buffer belongs to the caller of
 * {@code evaluateColumnar}, which is why nothing here closes one.
 * </p>
 *
 * <p>
 * An output column is allocated through RMM, which is the executor's own allocator: RAPIDS
 * configured it, it accounts for the pool and the spill framework, and allocating around it with
 * {@code cuMemAlloc} would take memory RAPIDS believes it has reserved. The kernel writes into that
 * allocation, and the {@link ColumnVector} built around it afterwards takes ownership, so the
 * downstream operator frees it on the normal path.
 * </p>
 */
public final class CudfBufferAdapter {

    private CudfBufferAdapter() {
    }

    /**
     * What a device view needs to be bound to one column's data buffer.
     *
     * <p>
     * An address of 0 with a length of 0 means "this column has no payload at all", which happens
     * for a LIST whose every row is empty. The caller substitutes an owned sentinel allocation, so
     * that the kernel receives a valid pointer it provably never reads through.
     * </p>
     */
    public record InputBinding(long address, long byteLength, long rows, PrimitiveKind kind) {
        boolean isEmpty() {
            return address == 0 && byteLength == 0;
        }
    }

    /**
     * What the two device views of a LIST column need: the offsets ({@code numRows + 1} int32
     * values) and the flattened child the offsets index into.
     */
    public record ListBinding(InputBinding offsets, InputBinding values) {
    }

    /**
     * Describes {@code column}'s data buffer, failing when it is not the fixed-width layout the
     * kernel was generated for.
     */
    public static InputBinding describe(ColumnVector column, PrimitiveKind expected, int numRows) {
        PrimitiveKind actual = PrimitiveKind.ofCudfType(column.getType());
        if (actual != expected) {
            throw new IllegalStateException("the column is " + column.getType() + " but the kernel was compiled for " + expected.cudfType());
        }
        if (column.getNullCount() != 0) {
            throw new IllegalStateException("the column has " + column.getNullCount() + " nulls, and this kernel was only admitted for non-null input");
        }
        BaseDeviceMemoryBuffer data = column.getData();
        if (data == null) {
            throw new IllegalStateException("the column has no data buffer");
        }
        long needed = (long) numRows * expected.byteWidth();
        if (data.getLength() < needed) {
            throw new IllegalStateException("the column's data buffer is " + data.getLength() + " bytes, less than the " + needed + " bytes " + numRows + " rows need");
        }
        return new InputBinding(data.getAddress(), needed, numRows, expected);
    }

    /**
     * Describes a LIST column's two buffers.
     *
     * <p>
     * Nothing is gathered or copied: the offsets buffer is libcudf's own, and so is the child. The
     * kernel reads row {@code r}'s elements between {@code offsets[r]} and {@code offsets[r+1]} of
     * the child, which is precisely the layout libcudf already has.
     * </p>
     */
    public static ListBinding describeList(ColumnVector column, PrimitiveKind expectedElement, int numRows) {
        if (!DType.LIST.equals(column.getType())) {
            throw new IllegalStateException("the column is " + column.getType() + " but the kernel was compiled for a LIST");
        }
        if (column.getNullCount() != 0) {
            throw new IllegalStateException("the LIST column has " + column.getNullCount() + " null rows, and this kernel was only admitted for non-null input");
        }
        BaseDeviceMemoryBuffer offsets = column.getOffsets();
        if (offsets == null) {
            throw new IllegalStateException("the LIST column has no offsets buffer");
        }
        long offsetsNeeded = (long) (numRows + 1) * Integer.BYTES;
        if (offsets.getLength() < offsetsNeeded) {
            throw new IllegalStateException("the offsets buffer is " + offsets.getLength() + " bytes, less than the " + offsetsNeeded + " bytes " + numRows + " rows need. "
                    + "A sliced LIST column is not supported: its offsets would not be indices into the child from zero.");
        }
        // The child view is a handle this method creates and therefore has to close. The payload it
        // points at belongs to the parent column, which the caller keeps alive, so closing the
        // handle does not invalidate the address read out of it.
        try (ColumnView child = column.getChildColumnView(0)) {
            PrimitiveKind actualElement = PrimitiveKind.ofCudfType(child.getType());
            if (actualElement != expectedElement) {
                throw new IllegalStateException("the list elements are " + child.getType() + " but the kernel was compiled for " + expectedElement.cudfType());
            }
            if (child.getNullCount() != 0) {
                throw new IllegalStateException("the list has " + child.getNullCount() + " null elements, and this kernel was only admitted for non-null input");
            }
            long childElements = child.getRowCount();
            BaseDeviceMemoryBuffer values = child.getData();
            if (childElements == 0 || values == null) {
                // Every row is empty. cuDF need not allocate a child payload at all for that, and
                // this is a perfectly ordinary batch: a length-controlled reduction over it returns
                // the identity for every row. The bounds proof guarantees each access satisfies
                // start <= start + i < end, and end equals start on every row here, so no element
                // is ever read. The caller binds a small owned sentinel so the kernel still has a
                // valid pointer argument.
                return new ListBinding(
                        new InputBinding(offsets.getAddress(), offsetsNeeded, numRows + 1L, PrimitiveKind.INT),
                        new InputBinding(0L, 0L, 0L, expectedElement));
            }
            long valuesNeeded = childElements * expectedElement.byteWidth();
            if (values.getLength() < valuesNeeded) {
                throw new IllegalStateException("the child buffer is " + values.getLength() + " bytes, less than the " + valuesNeeded + " bytes its " + childElements + " elements need");
            }
            return new ListBinding(
                    new InputBinding(offsets.getAddress(), offsetsNeeded, numRows + 1L, PrimitiveKind.INT),
                    new InputBinding(values.getAddress(), valuesNeeded, childElements, expectedElement));
        }
    }

    /**
     * An RMM allocation for the kernel's output, sized for {@code numRows} of {@code kind}.
     *
     * <p>
     * The caller owns it until {@link #wrapAsColumn} hands it to a {@link ColumnVector}, and must
     * close it if anything fails before that.
     * </p>
     */
    public static DeviceMemoryBuffer allocateOutput(int numRows, PrimitiveKind kind) {
        return DeviceMemoryBuffer.allocate((long) numRows * kind.byteWidth());
    }

    /**
     * Wraps a written output buffer as a cuDF column that owns it. The column has no validity mask
     * because this path only runs for UDFs whose inputs cannot be null and whose output therefore
     * cannot be either.
     */
    public static ColumnVector wrapAsColumn(DeviceMemoryBuffer data, int numRows, PrimitiveKind kind) {
        return new ColumnVector(kind.cudfType(), numRows, Optional.of(0L), data, null, null);
    }

    /**
     * Makes everything cuDF has queued visible to a kernel that is about to read it.
     *
     * <p>
     * The two sides do not share a stream: RAPIDS issues its work on cuDF's streams and TornadoVM
     * launches on its own, and nothing orders one against the other. This is the coarse way to
     * establish that order -- wait for the whole device -- which is correct and is measured
     * separately in the benchmark breakdown, rather than assumed to be free.
     * </p>
     */
    public static void synchronizeBeforeKernel() {
        Cuda.DEFAULT_STREAM.sync();
    }

    /**
     * Makes the kernel's writes visible to the cuDF operator that consumes the result.
     *
     * <p>
     * {@code TornadoExecutionPlan.execute()} has already waited for its own stream when this runs;
     * this additionally covers anything the driver has outstanding on the device, so the column
     * handed back is complete whichever stream the next operator uses.
     * </p>
     */
    public static void synchronizeAfterKernel() {
        Cuda.deviceSynchronize();
    }
}
