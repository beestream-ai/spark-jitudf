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
package ai.beestream.jitudf.bench.movielens;

import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Method;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.ScalaUDF;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;

import scala.collection.Seq;

/** Why RAPIDS cannot see NVIDIA's RapidsUDF behind Spark 4's UDF registration wrapper. */
public final class LambdaProbe {
    public static void main(String[] args) throws Exception {
        SparkSession.Builder b = SparkSession.builder().appName("lambda-probe").master("local[1]");
        if (Boolean.getBoolean("probe.plugin")) {
            b = b.config("spark.plugins", "com.nvidia.spark.SQLPlugin")
                 .config("spark.rapids.sql.enabled", "true").config("spark.rapids.sql.udfCompiler.enabled", System.getProperty("probe.udfcompiler","false"))
                 .config("spark.rapids.memory.gpu.allocFraction", "0.3").config("spark.rapids.memory.gpu.reserve", "805306368")
                 .config("spark.rapids.memory.gpu.maxAllocFraction", "0.4");
        }
        SparkSession spark = b.getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        // NVIDIA's class is not a build dependency of this repository; it is supplied on the
        // classpath by scripts/build-native-udf.sh. Loaded reflectively so that nothing here
        // ships in NVIDIA's package namespace.
        @SuppressWarnings("unchecked")
        UDF2<Seq<Float>, Seq<Float>, Float> nvidia = (UDF2<Seq<Float>, Seq<Float>, Float>)
                Class.forName("com.nvidia.spark.rapids.udf.java.CosineSimilarity")
                        .getDeclaredConstructor().newInstance();
        spark.udf().register("cosine_native", nvidia, DataTypes.FloatType);

        // Pull the ScalaUDF out of a real analysed plan rather than the catalog, so we see exactly
        // the object RAPIDS would inspect.
        org.apache.spark.sql.Dataset<org.apache.spark.sql.Row> df = spark.range(1).selectExpr(
                "cosine_native(array(cast(1.0 as float)), array(cast(1.0 as float))) as s");
        final ScalaUDF[] found = new ScalaUDF[1];
        df.queryExecution().analyzed().expressions().foreach(e -> {
            e.foreach(sub -> {
                if (sub instanceof ScalaUDF u) {
                    found[0] = u;
                }
                return null;
            });
            return null;
        });
        Object fn = found[0].function();
        System.out.println("PROBE function class: " + fn.getClass().getName());
        System.out.println("PROBE is RapidsUDF?   " + (fn instanceof com.nvidia.spark.RapidsUDF));
        Method wr = fn.getClass().getDeclaredMethod("writeReplace");
        wr.setAccessible(true);
        SerializedLambda sl = (SerializedLambda) wr.invoke(fn);
        System.out.println("PROBE capturedArgCount: " + sl.getCapturedArgCount() + "   (RAPIDS requires exactly 1)");
        for (int i = 0; i < sl.getCapturedArgCount(); i++) {
            Object c = sl.getCapturedArg(i);
            System.out.println("PROBE   capture[" + i + "] = " + (c == null ? "null" : c.getClass().getName())
                    + (c instanceof com.nvidia.spark.RapidsUDF ? "  <-- the RapidsUDF" : ""));
        }
        // Replicate RAPIDS' getRapidsUDFInstance step by step to see which step diverges.
        Class<?> lam = fn.getClass();
        System.out.println("STEP simpleName            = " + lam.getSimpleName());
        System.out.println("STEP simpleName has lambda = " + lam.getSimpleName().toLowerCase().contains("lambda"));
        Object cap = sl.getCapturedArg(0);
        Class<?> rapidsUdfHere = com.nvidia.spark.RapidsUDF.class;
        System.out.println("STEP RapidsUDF (probe)     = " + rapidsUdfHere + " @" + System.identityHashCode(rapidsUdfHere)
                + " loader=" + rapidsUdfHere.getClassLoader());
        for (Class<?> i : cap.getClass().getInterfaces()) {
            System.out.println("STEP capture implements    = " + i + " @" + System.identityHashCode(i) + " loader=" + i.getClassLoader());
        }
        System.out.println("STEP capture isInstance(probe RapidsUDF) = " + rapidsUdfHere.isInstance(cap));
        try {
            Class<?> g = Class.forName("org.apache.spark.sql.rapids.GpuScalaUDF$");
            Class<?> rapidsUdfThere = Class.forName("com.nvidia.spark.RapidsUDF", false, g.getClassLoader());
            System.out.println("STEP RapidsUDF (plugin)    = " + rapidsUdfThere + " @" + System.identityHashCode(rapidsUdfThere)
                    + " loader=" + rapidsUdfThere.getClassLoader());
            System.out.println("STEP SAME CLASS OBJECT?    = " + (rapidsUdfHere == rapidsUdfThere));
            System.out.println("STEP plugin-view isInstance= " + rapidsUdfThere.isInstance(cap));
        } catch (Throwable t) {
            System.out.println("STEP plugin RapidsUDF lookup threw " + t);
        }

        // Ask RAPIDS' own detector what it makes of this function.
        try {
            Class<?> g = Class.forName("org.apache.spark.sql.rapids.GpuScalaUDF$");
            Object mod = g.getField("MODULE$").get(null);
            Object r = g.getMethod("getRapidsUDFInstance", Object.class).invoke(mod, fn);
            System.out.println("PROBE getRapidsUDFInstance -> " + r);
        } catch (Throwable t) {
            System.out.println("PROBE getRapidsUDFInstance threw " + t);
        }
        spark.stop();
    }
    private LambdaProbe() {}
}
