package io.github.beehivelab.jitudf.bench;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/** Prints the final physical plan of the benchmark's own query, for the report's evidence. */
public final class PlanDump {

    public static void main(String[] args) {
        SparkSession spark = Benchmark.session(256 * 1024 * 1024);
        try {
            for (String mode : new String[] { "B1-rapids-cpu-bridge", "B3-tornadovm-deviceview" }) {
                spark.conf().set("spark.rapids.sql.enabled", "true");
                spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(mode.startsWith("B3")));
                Dataset<Row> query = Benchmark.data(spark, 200_000).selectExpr("sum(tornado_loop(a, b)) as total");
                Row head = query.collectAsList().get(0);
                System.out.println("=== " + mode + " (sum over the UDF) ===");
                System.out.println("result: " + head.getDouble(0));
                System.out.println(query.queryExecution().executedPlan().toString());
                System.out.println();

                Dataset<Row> projection = Benchmark.data(spark, 200_000).selectExpr("tornado_loop(a, b) as value");
                projection.count();
                System.out.println("=== " + mode + " (projection only) ===");
                System.out.println(projection.queryExecution().executedPlan().toString());
                System.out.println();
            }
        } finally {
            spark.stop();
        }
    }
}
