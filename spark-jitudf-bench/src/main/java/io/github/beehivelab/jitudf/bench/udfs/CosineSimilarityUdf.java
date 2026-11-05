package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF2;

import scala.collection.Seq;

/**
 * Cosine similarity between two per-row vectors, in the shape NVIDIA publishes as the example of a
 * UDF worth accelerating.
 *
 * <p>
 * The argument type is {@code scala.collection.Seq}, not {@code double[]}: that is what Spark hands
 * a Java UDF for an {@code ArrayType} column, and writing the signature any other way would fail at
 * run time on the CPU as well. Behind it in cuDF is a {@code LIST} column -- offsets plus one
 * flattened child -- which is what the kernel indexes.
 * </p>
 *
 * <p>
 * Unlike NVIDIA's version this does not throw when the two vectors differ in length; it uses the
 * shorter. A kernel has no exception semantics, and refusing the UDF for a throw it would never
 * reach is less useful than defining the case.
 * </p>
 */
public class CosineSimilarityUdf implements UDF2<Seq<Double>, Seq<Double>, Double> {

    private static final long serialVersionUID = 1L;

    @Override
    public Double call(Seq<Double> a, Seq<Double> b) {
        int n = Math.min(a.length(), b.length());
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < n; i++) {
            double x = a.apply(i);
            double y = b.apply(i);
            dot += x * y;
            normA += x * x;
            normB += y * y;
        }
        double denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator == 0.0 ? 0.0 : dot / denominator;
    }
}
