package io.github.beehivelab.jitudf.bench;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

/** Prints what Spark hands a Java UDF for an ArrayType column, so the lowering targets the truth. */
public final class ArrayUdfProbe {

    public static class ProbeUdf implements UDF1<Object, String> {
        @Override
        public String call(Object value) {
            if (value == null) {
                return "null";
            }
            StringBuilder description = new StringBuilder(value.getClass().getName());
            for (Class<?> i : value.getClass().getInterfaces()) {
                description.append(" / ").append(i.getName());
            }
            if (value.getClass().isArray()) {
                description.append(" [component=").append(value.getClass().getComponentType().getName()).append(']');
            }
            return description.toString();
        }
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("array-probe").master("local[1]").getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        spark.udf().register("probe", new ProbeUdf(), DataTypes.StringType);
        spark.range(0, 1).selectExpr("array(cast(id as double), cast(id + 1 as double)) as arr") //
                .selectExpr("probe(arr) as what").show(1, false);
        spark.stop();
    }
}
