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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * Candidate A feasibility probe: does Spark's own ALS prediction UDF reach a GPU kernel?
 *
 * <p>
 * {@code ALSModel.transform} scores with a Scala UDF that Spark defines itself
 * (mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala:311). This probe trains a
 * small model on synthetic ratings -- no dataset download is needed to answer the question -- calls
 * {@code transform}, and prints the physical plan together with every routing decision RAPIDS and
 * the TornadoVM provider made. The question is whether that UDF is accelerated as it stands, with
 * no change to Spark.
 * </p>
 */
public final class AlsProbe {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder() //
                .appName("als-feasibility-probe") //
                .master("local[2]") //
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin") //
                .config("spark.rapids.sql.enabled", "true") //
                .config("spark.rapids.sql.udfCompiler.enabled", "true") //
                .config("spark.rapids.sql.tornadovm.enabled", "true") //
                .config("spark.rapids.sql.explain", "NOT_ON_GPU") //
                .config("spark.rapids.memory.gpu.pool", "ASYNC") //
                .config("spark.rapids.memory.gpu.allocFraction", "0.3") //
                .config("spark.rapids.memory.gpu.minAllocFraction", "0.05") //
                .config("spark.rapids.memory.gpu.maxAllocFraction", "0.5") //
                .config("spark.sql.shuffle.partitions", "2") //
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        int users = 400;
        int items = 300;
        StructType schema = new StructType() //
                .add("userId", DataTypes.IntegerType, false) //
                .add("movieId", DataTypes.IntegerType, false) //
                .add("rating", DataTypes.FloatType, false);
        List<Row> rows = new ArrayList<>();
        for (int u = 0; u < users; u++) {
            for (int k = 0; k < 20; k++) {
                int item = (u * 7 + k * 13) % items;
                float r = 1.0f + ((u + item) % 5);
                rows.add(RowFactory.create(u, item, r));
            }
        }
        Dataset<Row> ratings = spark.createDataFrame(rows, schema).repartition(2).cache();
        ratings.count();

        ALS als = new ALS() //
                .setRank(16).setMaxIter(3).setRegParam(0.1) //
                .setUserCol("userId").setItemCol("movieId").setRatingCol("rating") //
                .setColdStartStrategy("drop").setSeed(42);
        ALSModel model = als.fit(ratings);
        System.out.println("=== ALS model trained: rank=" + model.rank() + " ===");

        Dataset<Row> scored = model.transform(ratings);
        long n = scored.count();
        String plan = scored.queryExecution().executedPlan().toString();

        System.out.println("=== ROWS SCORED: " + n + " ===");
        System.out.println("=== PHYSICAL PLAN ===");
        for (String line : plan.split("\n")) {
            System.out.println("PLAN| " + line);
        }

        System.out.println("=== WHERE THE PREDICT UDF RAN ===");
        System.out.println("  plan mentions GpuCpuBridge ........ " + plan.contains("GpuCpuBridge"));
        System.out.println("  plan mentions a tornado kernel .... " + plan.contains("gpu_tornado"));
        System.out.println("  plan mentions ScalaUDF (CPU) ...... " + plan.contains("ScalaUDF"));
        System.out.println("  plan mentions GpuProject .......... " + plan.contains("GpuProject"));

        IntegrationTest.printDecisions();
        spark.stop();
    }

    private AlsProbe() {
    }
}
