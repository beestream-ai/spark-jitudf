package io.github.beehivelab.jitudf.bench;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/** Checks that the RAPIDS plugin initialises and puts a plain query on this laptop's GPU. */
public final class RapidsSmokeTest {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder() //
                .appName("rapids-smoke") //
                .master("local[2]") //
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin") //
                .config("spark.rapids.sql.enabled", "true") //
                .config("spark.rapids.sql.explain", "ALL") //
                .config("spark.rapids.memory.gpu.pool", "ASYNC") //
                .config("spark.rapids.memory.gpu.allocFraction", "0.3") //
                .config("spark.rapids.memory.gpu.maxAllocFraction", "0.5") //
                .config("spark.sql.shuffle.partitions", "2") //
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> df = spark.range(0, 1_000_000, 1, 2).selectExpr("id", "id * 2 as twice");
        Dataset<Row> projected = df.selectExpr("id", "twice", "id + twice as total");
        System.out.println("=== PHYSICAL PLAN ===");
        projected.explain(true);
        long count = projected.filter("total > 10").count();
        System.out.println("=== COUNT: " + count + " ===");
        Row first = projected.orderBy("id").head();
        System.out.println("=== FIRST ROW: " + first + " ===");
        spark.stop();
    }
}
