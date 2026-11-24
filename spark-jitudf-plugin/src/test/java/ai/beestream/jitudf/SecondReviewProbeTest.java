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
package ai.beestream.jitudf;

import java.io.Serializable;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

/**
 * Diagnostic only: prints what the provider decides, and the CUDA it emits, for the counterexamples
 * of Codex's second review. Run with {@code -Dtest=SecondReviewProbeTest} to reproduce them; it
 * asserts nothing, so it records the state of whichever revision it runs on.
 */
class SecondReviewProbeTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    // ----------------------------------------------------------------- A: inherited dispatch

    public static class BoxedParent implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return x + 1;
        }
    }

    public static class PrimitiveOverloadChild extends BoxedParent {
        public int call(int x, int y) {
            return x + 2;
        }
    }

    public static final class Popcount implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    // ----------------------------------------------------------------- B: readResolve

    public static final class ResolvedFactor implements UDF1<Integer, Integer> {
        private final int factor;

        public ResolvedFactor(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private Object readResolve() {
            return new ResolvedFactor(Integer.getInteger("review.factor", 9));
        }
    }

    public static final class StaticFromProperty implements UDF1<Integer, Integer> {
        static final int FACTOR = Integer.getInteger("review.static", 9);

        @Override
        public Integer call(Integer x) {
            return x + FACTOR;
        }
    }

    // ----------------------------------------------------------------- C: conditional divisor

    public static final class CondFloorMod implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorMod(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondFloorDiv implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorDiv(x, x == 7 ? y.intValue() : 3);
        }
    }

    public static final class CondDivide implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return x / (x == 7 ? y.intValue() : 3);
        }
    }

    static void show(String label, ProviderHarness.Outcome outcome) {
        System.out.println("=== " + label + ": " + outcome);
        if (outcome.cuda() != null) {
            System.out.println(outcome.cuda());
        }
    }

    @Test
    void probe() throws Exception {
        TornadoUdfProvider.resetDecisions();
        // A
        System.out.println("JVM child.call via UDF2 interface (7,0) = " + ((UDF2<Integer, Integer, Integer>) new PrimitiveOverloadChild()).call(7, 0));
        show("A inherited overload, direct", ProviderHarness.cudf(new PrimitiveOverloadChild(), new DataType[] { I, I }, I, false, "a_direct"));
        show("A inherited overload, Spark adapter", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new PrimitiveOverloadChild()), new DataType[] { I, I }, I, false, "a_adapter"));
        show("A inherited, no overload", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new BoxedParent() { }), new DataType[] { I, I }, I, false, "a_inherited"));
        UDF1<Long, Integer> inner = new Popcount();
        Popcount concrete = new Popcount();
        scala.Function1<Long, Integer> typed = (scala.Function1<Long, Integer> & Serializable) x -> concrete.call(x);
        show("A typed forwarding Function1(Long)", ProviderHarness.cudf(typed, new DataType[] { L }, I, false, "a_typed"));
        UDF1<Long, Integer> typedUdf = x -> inner.call(x);
        show("A typed forwarding Java UDF1 lambda via adapter", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(typedUdf), new DataType[] { L }, I, false, "a_typed_udf"));
        UDF1<Long, Integer> javaLambda = x -> Long.bitCount(x);
        show("A plain Java UDF1 lambda via adapter", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(javaLambda), new DataType[] { L }, I, false, "a_java_lambda"));
        // B
        show("B readResolve factor", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new ResolvedFactor(9)), new DataType[] { I }, I, false, "b_resolve"));
        show("B static from property", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new StaticFromProperty()), new DataType[] { I }, I, false, "b_static"));
        // C
        show("C floorMod cond", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new CondFloorMod()), new DataType[] { I, I }, I, false, "c_mod"));
        show("C floorDiv cond", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new CondFloorDiv()), new DataType[] { I, I }, I, false, "c_div"));
        show("C / cond", ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(new CondDivide()), new DataType[] { I, I }, I, false, "c_slash"));
    }
}
