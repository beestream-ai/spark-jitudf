package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF3;

/**
 * Fixed-width numeric arithmetic over three columns, with no loop: this is the shape that isolates
 * the per-batch cost of the integration from the cost of the computation.
 */
public class MathUdf implements UDF3<Double, Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    @Override
    public Double call(Double a, Double b, Double c) {
        double scaled = a * 1.5 + b * 0.25 - c;
        double norm = Math.sqrt(a * a + b * b + c * c);
        return Math.cbrt(scaled) + norm * 0.5;
    }
}
