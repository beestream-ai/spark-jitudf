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
