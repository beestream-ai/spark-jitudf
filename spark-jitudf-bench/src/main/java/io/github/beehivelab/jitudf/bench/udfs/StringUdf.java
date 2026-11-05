package io.github.beehivelab.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF1;

/**
 * A UDF neither compiler can take: RAPIDS has no expression for it and a GPU kernel has no strings.
 * It is here to show the refusal, which has to leave the query on the CPU and still be correct.
 */
public class StringUdf implements UDF1<Double, String> {

    private static final long serialVersionUID = 1L;

    @Override
    public String call(Double a) {
        return a > 0 ? "positive" : "non-positive";
    }
}
