package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF2;

/**
 * A loop whose trip count depends on the data, which is the case an expression tree cannot express
 * even in principle. The bound is clamped so one row cannot run away with the whole warp.
 */
public class DataDependentLoopUdf implements UDF2<Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    @Override
    public Double call(Double a, Double b) {
        int steps = (int) (a % 97.0);
        if (steps < 1) {
            steps = 1;
        }
        if (steps > 96) {
            steps = 96;
        }
        double acc = b;
        for (int i = 0; i < steps; i++) {
            acc = acc * 0.97 + a * 0.013 + i * 0.001;
        }
        return acc;
    }
}
