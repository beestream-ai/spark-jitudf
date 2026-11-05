package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF2;

/** A loop that calls a static helper in the same class, to exercise the reachable-method walk. */
public class HelperLoopUdf implements UDF2<Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    private static double step(double acc, double a, double b, int i) {
        return acc * 0.5 + a * b + i * 0.125;
    }

    @Override
    public Double call(Double a, Double b) {
        double acc = 0.0;
        for (int i = 0; i < 32; i++) {
            acc = step(acc, a, b, i);
        }
        return acc;
    }
}
