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

/** Phase-1 probe: prints what the provider does with each Codex counterexample. No assertions. */
class CodexProbeTest {

    public static final class BitCountUdf implements UDF1<Long, Integer> {
        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    public static final class StrideTwoUdf implements UDF1<long[], Integer> {
        @Override
        public Integer call(long[] arr) {
            int acc = 0;
            for (int i = 0; i < arr.length; i += 2) {
                acc += Long.bitCount(arr[i]);
            }
            return acc;
        }
    }

    public static final class BadCastUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return ((Boolean) (Object) x).booleanValue() ? 21 : 42;
        }
    }

    public static final class TransientFieldUdf implements UDF1<Integer, Integer> {
        private final transient int factor;

        public TransientFieldUdf() {
            this.factor = 9;
        }

        @Override
        public Integer call(Integer x) {
            return factor + x;
        }
    }

    public static final class FloorDivUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorDiv(x, y);
        }
    }

    public static final class FloorModUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return Math.floorMod(x, y);
        }
    }

    public static final class PlainDivUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer x, Integer y) {
            return x / y;
        }
    }

    public static final class SinInIntUdf implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return (int) (Math.sin((double) x) * 1_000_000_000.0);
        }
    }

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    static void show(String label, ProviderHarness.Outcome o) {
        System.out.println("PROBE " + label + ": " + o);
        if (o.cuda() != null) {
            System.out.println(o.cuda());
        }
    }

    @Test
    void probe() {
        TornadoUdfProvider.resetDecisions();
        BitCountUdf inner = new BitCountUdf();
        @SuppressWarnings("unchecked")
        scala.Function1<Object, Object> wrapper = (scala.Function1<Object, Object> & Serializable) x -> 1 + (Integer) inner.call((Long) x);
        show("F1 inner", ProviderHarness.cudf(inner, new DataType[] { L }, I, false, "inner"));
        show("F1 wrapper x->1+inner(x)", ProviderHarness.cudf(wrapper, new DataType[] { L }, I, false, "wrapper"));
        show("F1 genuine ToScalaUDF adapter",
                ProviderHarness.cudf(org.apache.spark.sql.internal.ToScalaUDF$.MODULE$.apply(inner), new DataType[] { L }, I, false, "adapter"));
        show("F2 stride two", ProviderHarness.cudf(new StrideTwoUdf(), new DataType[] { DataTypes.createArrayType(L, false) }, I, false, "stride2"));
        show("F4 bad cast", ProviderHarness.cudf(new BadCastUdf(), new DataType[] { I, I }, I, false, "badcast"));
        show("F5 transient", ProviderHarness.cudf(new TransientFieldUdf(), new DataType[] { I }, I, false, "transient"));
        show("F6 plain div", ProviderHarness.cudf(new PlainDivUdf(), new DataType[] { I, I }, I, false, "div"));
        show("F6 floorDiv", ProviderHarness.cudf(new FloorDivUdf(), new DataType[] { I, I }, I, false, "floorDiv"));
        show("F6 floorMod", ProviderHarness.cudf(new FloorModUdf(), new DataType[] { I, I }, I, false, "floorMod"));
        show("F7 sin in int, incompat off", ProviderHarness.cudf(new SinInIntUdf(), new DataType[] { I }, I, false, "sin"));
    }
}
