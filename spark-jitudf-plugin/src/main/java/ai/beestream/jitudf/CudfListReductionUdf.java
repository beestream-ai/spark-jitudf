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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.SegmentedReductionAggregation;

import com.nvidia.spark.RapidsUDF;

/**
 * A reduction over a {@code LIST} column, run as a transform of its child plus a segmented
 * reduction.
 *
 * <p>
 * This is the answer to a gap that both this project's own analysis and NVIDIA's review treated as
 * needing new libcudf code. {@code cudf::transform} refuses a {@code LIST} input, so a per-row
 * kernel over a list has no entry point -- their opportunity #5, ranked high-cost and deferred.
 * </p>
 *
 * <p>
 * A reduction does not need a per-row kernel. A list column is offsets plus a flattened child, and
 * the child is an ordinary fixed-width column, so:
 * </p>
 *
 * <ol>
 * <li>apply the element function to the whole child with one {@code transform} -- the child is
 * fixed-width, so this is the path that already works;</li>
 * <li>put the transformed child back behind the original offsets;</li>
 * <li>reduce each row's segment with {@code listReduce}.</li>
 * </ol>
 *
 * <p>
 * Both halves are in the shipped cuDF Java API. Nothing upstream changes, and one kernel launch
 * covers every element of every row rather than one thread per row -- which for a list column is
 * also the better shape, since rows differ in length and a per-row kernel would have threads idling
 * on the short ones.
 * </p>
 *
 * <p>
 * What it does not cover: a reduction that is not a sum, a UDF that reads two lists at once (the
 * MovieLens cosine similarity is the example), and anything whose per-row answer is not a fold over
 * independent elements. Those are refused at planning time by {@link ListReductionSplit}.
 * </p>
 */
public final class CudfListReductionUdf implements RapidsUDF, Serializable {

    private static final long serialVersionUID = 1L;

    private final String elementCuda;
    private final String udfName;
    private final PrimitiveKind resultKind;

    public CudfListReductionUdf(String elementCuda, String udfName, PrimitiveKind resultKind) {
        this.elementCuda = elementCuda;
        this.udfName = udfName;
        this.resultKind = resultKind;
    }

    public String cudaSource() {
        return elementCuda;
    }

    @Override
    public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) {
        if (args.length != 1) {
            throw new IllegalStateException(udfName + " takes one list column but was given " + args.length);
        }
        ColumnVector list = args[0];
        if (!DType.LIST.equals(list.getType())) {
            throw new IllegalStateException(udfName + " expected a LIST column but got " + list.getType());
        }
        // The child is borrowed from the caller's column; the transformed child and the rebuilt view
        // are ours and are closed here. The reduction's result is what this method returns, so it is
        // the one thing not closed.
        try (ColumnView child = list.getChildColumnView(0)) {
            try (ColumnVector mapped = child.transform(elementCuda, false)) {
                try (ColumnView rebuilt = list.replaceChildrenWithViews(new int[] { 0 }, new ColumnView[] { mapped })) {
                    return rebuilt.listReduce(SegmentedReductionAggregation.sum(), resultKind.cudfType());
                }
            }
        }
    }

    @Override
    public String toString() {
        return "CudfListReductionUdf[" + udfName + " -> " + resultKind.cudfType() + "]";
    }
}
