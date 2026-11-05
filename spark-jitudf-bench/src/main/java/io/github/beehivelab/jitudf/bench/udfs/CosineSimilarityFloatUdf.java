package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF2;

import scala.collection.Seq;

/**
 * Cosine similarity over {@code float} vectors, written to match NVIDIA's published CUDA UDF as
 * closely as an ordinary JVM UDF can.
 *
 * <p>
 * This exists for one purpose: to compare automatic compilation of JVM bytecode against a
 * hand-written CUDA kernel <b>at the same data precision</b>. The project's main scoring UDF is
 * {@code double} throughout; NVIDIA's is typed on {@code float}, so comparing the two also compares
 * eight-byte traffic against four-byte traffic. This one removes that difference.
 * </p>
 *
 * <p>
 * Matched to NVIDIA's implementation deliberately, element for element:
 * </p>
 * <ul>
 * <li>Inputs are {@code float} and the result is {@code float}.</li>
 * <li>The dot product and both magnitudes accumulate in {@code double} and the result is narrowed
 * once at the end &mdash; which is what their {@code call()} does, and what their kernel does.</li>
 * <li>Magnitudes are {@code sqrt} of the sum of squares, multiplied, then divided into the dot
 * product.</li>
 * </ul>
 *
 * <p>
 * One difference remains and is deliberate. NVIDIA's version throws
 * {@code IllegalArgumentException} when the two vectors differ in length; this one scores the
 * shorter prefix. A kernel has no exception semantics, so a UDF that can throw is refused outright
 * by the compiler, and refusing it would defeat the comparison. On the data used here every vector
 * is the full 1,128-element Tag Genome, so the branch is never reached &mdash; but the two
 * implementations are not equivalent for ragged input, and that is a real difference rather than a
 * hidden one.
 * </p>
 *
 * <p>
 * A zero magnitude divides by zero and yields a non-finite result here exactly as it does in
 * NVIDIA's implementation, rather than being special-cased to zero.
 * </p>
 */
public class CosineSimilarityFloatUdf implements UDF2<Seq<Float>, Seq<Float>, Float> {

    private static final long serialVersionUID = 1L;

    @Override
    public Float call(Seq<Float> a, Seq<Float> b) {
        int n = Math.min(a.length(), b.length());
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < n; i++) {
            float x = a.apply(i);
            float y = b.apply(i);
            dot += x * y;
            normA += x * x;
            normB += y * y;
        }
        return (float) (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}
