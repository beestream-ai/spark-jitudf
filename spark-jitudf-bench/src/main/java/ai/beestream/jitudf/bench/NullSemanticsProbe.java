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
package ai.beestream.jitudf.bench;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * What Spark itself does when a null reaches a UDF. Nothing here is a kernel or a GPU: this is the
 * CPU oracle that any null handling in the plugin has to reproduce, and it is a program rather than
 * a paragraph because the answer decides whether a whole class of UDFs can be compiled at all.
 *
 * <p>
 * The question is not rhetorical. Spark short-circuits a null argument to a null result, without
 * running the UDF, only for arguments it knows to be primitive ({@code ScalaUDF.inputPrimitives}).
 * A Java {@code UDF1<Long, Integer>} has a boxed parameter whatever the column type is, so the
 * short-circuit may not apply to it, and then the UDF is called with null and does whatever its
 * own code does -- which, for a body that starts by unboxing, is throw.
 * </p>
 */
public final class NullSemanticsProbe {

    private NullSemanticsProbe() {
    }

    /** Unboxes immediately: whatever Spark does with the null decides what happens. */
    public static final class UnboxingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            long x = v;
            return (int) (x & 0xFF);
        }
    }

    /** Handles null itself, the way a careful author would. */
    public static final class NullCheckingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long v) {
            if (v == null) {
                return -1;
            }
            return (int) (v & 0xFF);
        }
    }

    private static void report(SparkSession spark, String what, String udf) {
        try {
            List<Row> rows = spark.sql("select id, " + udf + "(v) as r from t order by id").collectAsList();
            var out = new StringBuilder();
            for (Row r : rows) {
                out.append(r.isNullAt(1) ? "null" : r.get(1).toString()).append(' ');
            }
            System.out.println("  " + what + ": " + out.toString().trim());
        } catch (Exception e) {
            var root = e;
            while (root.getCause() instanceof Exception next && root.getCause() != root) {
                root = next;
            }
            System.out.println("  " + what + ": THREW " + root.getClass().getName());
        }
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("null-semantics-probe").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        // A nullable bigint column: 1, null, 3.
        StructType schema = new StructType().add("id", DataTypes.IntegerType, false).add("v", DataTypes.LongType, true);
        spark.createDataFrame(Arrays.asList(org.apache.spark.sql.RowFactory.create(1, 1L), org.apache.spark.sql.RowFactory.create(2, (Long) null),
                org.apache.spark.sql.RowFactory.create(3, 3L)), schema).createOrReplaceTempView("t");

        spark.udf().register("unboxing", new UnboxingUdf(), DataTypes.IntegerType);
        spark.udf().register("nullChecking", new NullCheckingUdf(), DataTypes.IntegerType);

        System.out.println("=== what Spark does with a null argument (values for id 1, 2, 3) ===");
        report(spark, "java UDF1<Long,Integer>, unboxes at once ", "unboxing");
        report(spark, "java UDF1<Long,Integer>, checks for null  ", "nullChecking");

        // What the planner is told, which is what the plugin gets to reason about.
        var plan = spark.sql("select unboxing(v) from t").queryExecution().analyzed();
        System.out.println("=== ScalaUDF.inputPrimitives for the java UDF ===");
        plan.collect(new scala.PartialFunction<org.apache.spark.sql.catalyst.plans.logical.LogicalPlan, Object>() {
            @Override
            public boolean isDefinedAt(org.apache.spark.sql.catalyst.plans.logical.LogicalPlan x) {
                return true;
            }

            @Override
            public Object apply(org.apache.spark.sql.catalyst.plans.logical.LogicalPlan node) {
                node.expressions().foreach(e -> {
                    e.foreach(sub -> {
                        if (sub instanceof org.apache.spark.sql.catalyst.expressions.ScalaUDF udf) {
                            System.out.println("  inputPrimitives = " + udf.inputPrimitives() + ", udf nullable = " + udf.nullable());
                        }
                        return scala.runtime.BoxedUnit.UNIT;
                    });
                    return scala.runtime.BoxedUnit.UNIT;
                });
                return scala.runtime.BoxedUnit.UNIT;
            }
        });
        spark.stop();
    }
}
