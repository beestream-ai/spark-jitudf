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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.List;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.internal.ToScalaUDF$;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Codex finding 1: the compiled method must be the computation Spark registered.
 *
 * <p>
 * The counterexample was a function {@code x -> 1 + inner.call(x)}: it captures one Java UDF, and
 * the resolver took "captures one Java UDF" to mean "is Spark's adapter", compiled {@code inner},
 * and served the wrapper inner's CUDA from the cache. Spark's adapter is now recognised from its
 * own bytecode, and everything else that captures a UDF is declined.
 * </p>
 */
class UdfResolutionTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ------------------------------------------------------------------------ genuine adapters

    @Test
    void sparksOwnAdapterIsResolvedToTheUdfItForwardsTo() throws Exception {
        CoverageCorpus.BitCount inner = new CoverageCorpus.BitCount();
        scala.Function1<Object, Object> adapter = ToScalaUDF$.MODULE$.apply(inner);
        ProviderHarness.Outcome viaAdapter = ProviderHarness.cudf(adapter, new DataType[] { L }, I, false, "adapter");
        assertTrue(viaAdapter.accepted(), viaAdapter::decision);
        TornadoUdfProvider.resetDecisions();
        ProviderHarness.Outcome direct = ProviderHarness.cudf(inner, new DataType[] { L }, I, false, "direct");
        assertEquals(direct.cuda(), viaAdapter.cuda(), "a pure forwarding adapter computes exactly what it forwards to");

        Method body = LoweredOracle.scalarBody(adapter, List.of(L), I);
        for (long x : new long[] { 0, 7, -1, Long.MIN_VALUE, 0x5555_5555_5555_5555L }) {
            assertEquals(adapter.apply(x), LoweredOracle.run(body, x), "x=" + x);
        }
    }

    @Test
    void aTwoInputAdapterKeepsArgumentOrder() throws Exception {
        UDF2<Integer, Integer, Integer> minus = new Minus();
        scala.Function2<Object, Object, Object> adapter = ToScalaUDF$.MODULE$.apply(minus);
        ProviderHarness.Outcome o = ProviderHarness.cudf(adapter, new DataType[] { I, I }, I, false, "minus");
        assertTrue(o.accepted(), o::decision);
        Method body = LoweredOracle.scalarBody(adapter, List.of(I, I), I);
        assertEquals(adapter.apply(10, 3), LoweredOracle.run(body, 10, 3));
        assertEquals(7, LoweredOracle.run(body, 10, 3));
    }

    public static final class Minus implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return a - b;
        }
    }

    @Test
    void nestedHelpersBehindAnAdapterStaySupported() {
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new CoverageCorpus.NestedHelpers()), new DataType[] { L }, L, false, "nested");
        assertTrue(o.accepted(), o::decision);
    }

    // ------------------------------------------------------------------------- wrappers that compute

    @Test
    void aComposedFunctionIsNotCompiledAsItsInnerUdf() {
        CoverageCorpus.BitCount concrete = new CoverageCorpus.BitCount();
        @SuppressWarnings("unchecked")
        UDF1<Object, Object> inner = (UDF1<Object, Object>) (UDF1<?, ?>) concrete;
        scala.Function1<Object, Object> composed = (scala.Function1<Object, Object> & Serializable) x -> 1 + (Integer) call1(inner, x);
        assertEquals(4, composed.apply(7L), "the registered function is 1 + popcount");
        assertEquals(3, concrete.call(7L));

        ProviderHarness.Outcome o = ProviderHarness.cudf(composed, new DataType[] { L }, I, false, "composed");
        assertFalse(o.accepted(), "a wrapper that adds 1 must not be compiled as the inner UDF: " + o.decision());
        assertTrue(o.decision().contains("does more than forward"), o::decision);
    }

    @Test
    void aWrapperCapturingTheConcreteClassIsDeclined() {
        CoverageCorpus.BitCount inner = new CoverageCorpus.BitCount();
        scala.Function1<Object, Object> composed = (scala.Function1<Object, Object> & Serializable) x -> 1 + (Integer) call1(inner, x);
        ProviderHarness.Outcome o = ProviderHarness.cudf(composed, new DataType[] { L }, I, false, "composedConcrete");
        assertFalse(o.accepted(), o::decision);
    }

    @Test
    void aWrapperThatReordersArgumentsIsDeclined() {
        @SuppressWarnings("unchecked")
        UDF2<Object, Object, Object> inner = (UDF2<Object, Object, Object>) (UDF2<?, ?, ?>) new Minus();
        scala.Function2<Object, Object, Object> swapped = (scala.Function2<Object, Object, Object> & Serializable) (a, b) -> call2(inner, b, a);
        ProviderHarness.Outcome o = ProviderHarness.cudf(swapped, new DataType[] { I, I }, I, false, "swapped");
        assertFalse(o.accepted(), o::decision);
        assertTrue(o.decision().contains("unchanged and in order"), o::decision);
    }

    @Test
    void aWrapperWithSeveralCapturedValuesIsDeclined() {
        @SuppressWarnings("unchecked")
        UDF1<Object, Object> inner = (UDF1<Object, Object>) (UDF1<?, ?>) new CoverageCorpus.BitCount();
        int offset = 5;
        scala.Function1<Object, Object> twoCaptures = (scala.Function1<Object, Object> & Serializable) x -> (Integer) call1(inner, x) + offset;
        ProviderHarness.Outcome o = ProviderHarness.cudf(twoCaptures, new DataType[] { L }, I, false, "twoCaptures");
        assertFalse(o.accepted(), o::decision);
        assertTrue(o.decision().contains("captures 2 values"), o::decision);
    }

    @SuppressWarnings("unchecked")
    static Object call1(UDF1<?, ?> f, Object x) {
        try {
            return ((UDF1<Object, Object>) f).call(x);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    static Object call2(UDF2<?, ?, ?> f, Object a, Object b) {
        try {
            return ((UDF2<Object, Object, Object>) f).call(a, b);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ---------------------------------------------------------------------- capture-free lambdas

    @Test
    void aCaptureFreeLambdaIsMatchedByItsFullDescriptorNotItsName() throws Exception {
        // Two static plusOne overloads: (J)J adds 1, (I)J adds 2. getDeclaredMethods order is
        // unspecified, so a name match could compile either one.
        scala.Function1<Long, Long> longLambda = (scala.Function1<Long, Long> & Serializable) CoverageCorpus.ScalaShapes::plusOne;
        scala.Function1<Integer, Long> intLambda = (scala.Function1<Integer, Long> & Serializable) CoverageCorpus.ScalaShapes::plusOne;

        ProviderHarness.Outcome a = ProviderHarness.cudf(longLambda, new DataType[] { L }, L, false, "plusOneLong");
        ProviderHarness.Outcome b = ProviderHarness.cudf(intLambda, new DataType[] { I }, L, false, "plusOneInt");
        assertTrue(a.accepted(), a::decision);
        assertTrue(b.accepted(), b::decision);
        assertNotEquals(a.cuda(), b.cuda());

        Method longBody = LoweredOracle.scalarBody(longLambda, List.of(L), L);
        Method intBody = LoweredOracle.scalarBody(intLambda, List.of(I), L);
        assertEquals(longLambda.apply(10L), LoweredOracle.run(longBody, 10L));
        assertEquals(11L, LoweredOracle.run(longBody, 10L));
        assertEquals(intLambda.apply(10), LoweredOracle.run(intBody, 10));
        assertEquals(12L, LoweredOracle.run(intBody, 10));
    }
}
