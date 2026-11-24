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
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
 * Codex's second review, finding A: the method compiled must be the one the JVM runs when Spark
 * calls the UDF through its interface -- selected by virtual dispatch on the exact erased
 * descriptor and followed through compiler bridges -- not "the call declared on the most-derived
 * class". And the typed forwarding wrappers the first remediation stopped recognising are
 * recognised again, by what their bytecode does rather than by one erased shape.
 *
 * <p>
 * Every accepted case is checked against the Java reference called <em>through the interface</em>
 * ({@link LoweredOracle#reference}), which is what Spark does.
 * </p>
 */
class DispatchResolutionTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ============================================================================ fixtures

    public static class BoxedParent implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return x + 1;
        }
    }

    /** Codex's counterexample: an overload whose descriptor is not the one Spark calls. */
    public static class PrimitiveOverloadChild extends BoxedParent {
        private static final long serialVersionUID = 1L;

        public int call(int x, int y) {
            return x + 2;
        }
    }

    /** Inherits its implementation unchanged. */
    public static final class InheritingChild extends BoxedParent {
        private static final long serialVersionUID = 1L;
    }

    /** Overrides the implementation: the child's body is what runs. */
    public static final class OverridingChild extends BoxedParent {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return x * y;
        }
    }

    /** Overrides, and adds overloads on both sides of it. */
    public static final class OverridingWithOverloads extends PrimitiveOverloadChild {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x, Integer y) {
            return x - y;
        }

        public long call(long x, long y) {
            return x + y + 1000;
        }
    }

    /** A generic base: the bridge {@code call(Object)Object} lives in the child, forwarding to {@code call(Long)}. */
    public abstract static class GenericBase<T> implements UDF1<T, Integer> {
        private static final long serialVersionUID = 1L;
    }

    public static final class BridgedChild extends GenericBase<Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return Long.bitCount(x) * 3;
        }

        public Integer call(Integer x) {
            return -1;
        }
    }

    /** The implementation is in a generic grandparent, the bridge in the parent that fixes T. */
    public abstract static class GenericImpl<T extends Number> implements UDF1<T, Integer> {
        private static final long serialVersionUID = 1L;
    }

    public static class LongImpl extends GenericImpl<Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return (int) (x % 1000);
        }
    }

    public static final class LongImplChild extends LongImpl {
        private static final long serialVersionUID = 1L;
    }

    public static final class Popcount implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    /** Implements two UDF interfaces at once: without Spark's adapter there is no saying which is called. */
    public static final class TwoInterfaces implements UDF1<Long, Integer>, UDF2<Long, Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return 1;
        }

        @Override
        public Integer call(Long x, Long y) {
            return 2;
        }
    }

    /** An interface default method is the implementation. */
    public interface DefaultCall extends UDF1<Long, Integer> {
        @Override
        default Integer call(Long x) {
            return Long.bitCount(x) + 7;
        }
    }

    public static final class UsesDefault implements DefaultCall {
        private static final long serialVersionUID = 1L;
    }

    /** The inherited body reaches an overridable instance method: dynamic dispatch inside the body. */
    public static class TemplateParent implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;

        protected long step(long x) {
            return x + 1;
        }

        @Override
        public Long call(Long x) {
            return step(x);
        }
    }

    public static final class TemplateChild extends TemplateParent {
        private static final long serialVersionUID = 1L;

        @Override
        protected long step(long x) {
            return x * 2;
        }
    }

    static final java.util.concurrent.atomic.AtomicInteger SIDE_EFFECTS = new java.util.concurrent.atomic.AtomicInteger();

    static int sideEffect() {
        return SIDE_EFFECTS.incrementAndGet();
    }

    // ============================================================================ positive

    @Test
    void codexCounterexampleCompilesTheInheritedInterfaceMethod() throws Exception {
        UDF2<Integer, Integer, Integer> viaInterface = new PrimitiveOverloadChild();
        assertEquals(8, viaInterface.call(7, 0), "Java runs the inherited call(Integer,Integer)");
        for (Object function : new Object[] { new PrimitiveOverloadChild(), ToScalaUDF$.MODULE$.apply(new PrimitiveOverloadChild()) }) {
            ResolvedUdf resolved = UdfResolver.resolve(function).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
            assertEquals(BoxedParent.class, resolved.declaringClass());
            assertEquals("(Ljava/lang/Integer;Ljava/lang/Integer;)Ljava/lang/Integer;", org.objectweb.asm.Type.getMethodDescriptor(resolved.method()));
        }
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new PrimitiveOverloadChild()), new DataType[] { I, I }, I, false, "codex_a");
        assertTrue(o.accepted(), o::decision);
        assertTrue(o.cuda().contains("(unsigned int) 1"), o::cuda);
        assertFalse(o.cuda().contains("(unsigned int) 2"), o::cuda);
        agreesWithJava(new PrimitiveOverloadChild(), List.of(I, I), I, new Object[][] { { 7, 0 }, { -3, 9 }, { Integer.MAX_VALUE, 1 } });
    }

    @Test
    void inheritedOverriddenAndOverloadedImplementations() throws Exception {
        agreesWithJava(new InheritingChild(), List.of(I, I), I, new Object[][] { { 7, 0 }, { 1, 2 } });
        agreesWithJava(new OverridingChild(), List.of(I, I), I, new Object[][] { { 7, 3 }, { -2, 5 } });
        agreesWithJava(new OverridingWithOverloads(), List.of(I, I), I, new Object[][] { { 7, 3 }, { -2, 5 } });
        assertEquals(OverridingWithOverloads.class, UdfResolver.resolve(new OverridingWithOverloads()).orElseThrow().declaringClass());
        for (BoxedParent udf : new BoxedParent[] { new InheritingChild(), new OverridingChild(), new OverridingWithOverloads() }) {
            assertTrue(ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(udf), new DataType[] { I, I }, I, false, udf.getClass().getSimpleName()).accepted());
        }
    }

    @Test
    void bridgesAreFollowedByDispatch() throws Exception {
        agreesWithJava(new BridgedChild(), List.of(L), I, new Object[][] { { 7L }, { -1L }, { 0L } });
        assertEquals("(Ljava/lang/Long;)Ljava/lang/Integer;", org.objectweb.asm.Type.getMethodDescriptor(UdfResolver.resolve(new BridgedChild()).orElseThrow().method()));
        agreesWithJava(new LongImplChild(), List.of(L), I, new Object[][] { { 123456L }, { -7L } });
        assertEquals(LongImpl.class, UdfResolver.resolve(new LongImplChild()).orElseThrow().declaringClass());
    }

    @Test
    void typedAndErasedForwardersResolveToWhatTheyForwardTo() throws Exception {
        Popcount concrete = new Popcount();
        UDF1<Long, Integer> asInterface = concrete;
        // Codex's regression: a typed forwarder, Long rather than erased Object.
        scala.Function1<Long, Integer> typed = (scala.Function1<Long, Integer> & Serializable) x -> concrete.call(x);
        // The same, erased, with the cast the target needs.
        scala.Function1<Object, Object> erased = (scala.Function1<Object, Object> & Serializable) x -> concrete.call((Long) x);
        // A Java UDF lambda that forwards through the interface, registered as Spark would.
        UDF1<Long, Integer> javaForwarder = x -> asInterface.call(x);
        // A plain Java UDF lambda: its static implementation is the computation.
        UDF1<Long, Integer> javaLambda = x -> Long.bitCount(x);
        // The ordinary case, and Spark's adapter around it.
        Object[] functions = { typed, erased, ToScalaUDF$.MODULE$.apply(javaForwarder), ToScalaUDF$.MODULE$.apply(javaLambda), concrete,
                ToScalaUDF$.MODULE$.apply(concrete) };
        String cuda = null;
        for (Object function : functions) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(function, new DataType[] { L }, I, false, "fwd" + System.identityHashCode(function));
            assertTrue(o.accepted(), () -> function + ": " + o.decision());
            Method body = LoweredOracle.scalarBody(function, List.of(L), I);
            for (long x : new long[] { 0, 7, -1, Long.MIN_VALUE }) {
                assertEquals(Long.bitCount(x), LoweredOracle.run(body, x));
            }
            if (function != javaLambda && cuda != null && !(function instanceof Popcount)) {
                assertNotNull(o.cuda());
            }
            cuda = o.cuda();
        }
    }

    /** Bound method references are forwarders by construction. */
    @Test
    void boundMethodReferencesResolveToTheReferencedMethod() throws Exception {
        Popcount concrete = new Popcount();
        UDF1<Long, Integer> viaInterface = concrete::call;
        scala.Function1<Long, Integer> viaFunction = (scala.Function1<Long, Integer> & Serializable) concrete::call;
        PrimitiveOverloadChild child = new PrimitiveOverloadChild();
        UDF2<Integer, Integer, Integer> inherited = child::call;
        for (Object function : new Object[] { ToScalaUDF$.MODULE$.apply(viaInterface), viaFunction }) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(function, new DataType[] { L }, I, false, "ref" + System.identityHashCode(function));
            assertTrue(o.accepted(), o::decision);
            assertEquals(3, LoweredOracle.run(LoweredOracle.scalarBody(function, List.of(L), I), 7L));
        }
        // child::call with (Integer, Integer) target type binds the inherited boxed method.
        assertEquals(8, inherited.call(7, 0));
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(inherited), new DataType[] { I, I }, I, false, "ref_inherited");
        assertTrue(o.accepted(), o::decision);
        assertEquals(8, LoweredOracle.run(LoweredOracle.scalarBody(ToScalaUDF$.MODULE$.apply(inherited), List.of(I, I), I), 7, 0));
    }

    /** A Scala-shaped forwarder with a primitive parameter: exact except for null, so declined only over nulls. */
    @Test
    void primitiveForwarderIsDeclinedOnlyWhereNullsArise() throws Exception {
        Popcount concrete = new Popcount();
        scala.runtime.java8.JFunction1$mcIJ$sp primitive = (scala.runtime.java8.JFunction1$mcIJ$sp & Serializable) x -> concrete.call(x);
        ProviderHarness.Outcome plain = ProviderHarness.cudf(primitive, new DataType[] { L }, I, false, "prim");
        assertTrue(plain.accepted(), plain::decision);
        Method body = LoweredOracle.scalarBody(primitive, List.of(L), I);
        assertEquals(3, LoweredOracle.run(body, 7L));
        ProviderHarness.Outcome nullable = ProviderHarness.cudf(primitive, new DataType[] { L }, I, true, "prim_null");
        assertFalse(nullable.accepted(), nullable::decision);
        assertTrue(nullable.decision().contains("unboxes"), nullable::decision);
    }

    // ============================================================================ negative

    @Test
    void wrappersThatComputeOrHaveSideEffectsAreDeclined() {
        Popcount concrete = new Popcount();
        scala.Function1<Long, Integer> plusOne = (scala.Function1<Long, Integer> & Serializable) x -> concrete.call(x) + 1;
        scala.Function1<Long, Integer> negated = (scala.Function1<Long, Integer> & Serializable) x -> -concrete.call(x);
        scala.Function1<Long, Integer> shifted = (scala.Function1<Long, Integer> & Serializable) x -> concrete.call(x + 1);
        scala.Function1<Long, Integer> effect = (scala.Function1<Long, Integer> & Serializable) x -> {
            sideEffect();
            return concrete.call(x);
        };
        for (Object f : new Object[] { plusOne, negated, shifted, effect }) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(f, new DataType[] { L }, I, false, "neg" + System.identityHashCode(f));
            assertFalse(o.accepted(), o::decision);
            assertTrue(o.decision().contains("does more than forward"), o::decision);
        }
    }

    @Test
    void aTypedWrapperWhoseTypeDisagreesWithTheTargetIsDeclined() {
        // UDF1<Long,..> called with an Integer: Java raises ClassCastException in the bridge.
        UDF1<Object, Object> raw = cast(new Popcount());
        scala.Function1<Integer, Object> wrong = (scala.Function1<Integer, Object> & Serializable) x -> callUnchecked(raw, x);
        ProviderHarness.Outcome o = ProviderHarness.cudf(wrong, new DataType[] { I }, I, false, "wrong_type");
        assertFalse(o.accepted(), o::decision);
    }

    @Test
    void forwardingMustPreserveAnExplicitArgumentCast() throws Exception {
        UDF1<Object, Object> raw = cast(new Popcount());
        UDF1<Object, Object> wrapper = value -> raw.call((String) value);
        org.junit.jupiter.api.Assertions.assertThrows(ClassCastException.class, () -> wrapper.call(7L));
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(wrapper), new DataType[] { L }, I, false, "wrong_cast");
        assertFalse(o.accepted(), o::decision);
    }

    static scala.Function1<Long, Integer> typedInterfaceForwarder(UDF1<Long, Integer> inner) {
        return (scala.Function1<Long, Integer> & Serializable) x -> {
            try {
                return inner.call(x);
            } catch (Exception failure) {
                if (failure instanceof RuntimeException runtime) {
                    throw runtime;
                }
                throw new RuntimeException(failure);
            }
        };
    }

    @Test
    void typedInterfaceForwarderPreservesRuntimeExceptions() {
        var wrapper = typedInterfaceForwarder(new Popcount());
        assertEquals(3, wrapper.apply(7L));
        org.junit.jupiter.api.Assertions.assertThrows(NullPointerException.class, () -> wrapper.apply(null));
        ProviderHarness.Outcome o = ProviderHarness.cudf(wrapper, new DataType[] { L }, I, false, "typed_interface_forwarder");
        assertTrue(o.accepted(), o::decision);
        UDF1<Long, Integer> inner = new Popcount();
        scala.Function1<Long, Integer> recovering = (scala.Function1<Long, Integer> & Serializable) x -> {
            try {
                return inner.call(x);
            } catch (Exception failure) {
                return -1;
            }
        };
        assertEquals(-1, recovering.apply(null));
        assertFalse(ProviderHarness.cudf(recovering, new DataType[] { L }, I, true, "recovers").accepted());
    }

    private static int widePopcount(long x) {
        return Long.bitCount(x);
    }

    @Test
    void methodReferenceCannotBypassItsInstantiatedArgumentCast() throws Exception {
        UDF1<Integer, Integer> widening = DispatchResolutionTest::widePopcount;
        UDF1<Object, Integer> erased = cast(widening);
        assertEquals(3, widening.call(7));
        org.junit.jupiter.api.Assertions.assertThrows(ClassCastException.class, () -> erased.call(7L));
        assertFalse(ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(widening), new DataType[] { L }, I, false, "widened").accepted());
    }

    @Test
    void ambiguousAndUnsupportedDispatchIsDeclined() {
        ProviderHarness.Outcome two = ProviderHarness.cudf(new TwoInterfaces(), new DataType[] { L }, I, false, "two");
        assertFalse(two.accepted(), two::decision);
        assertTrue(two.decision().contains("exactly one Spark Java UDF interface"), two::decision);
        // Through Spark's adapter the interface is known, and each one dispatches exactly.
        assertTrue(ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply((UDF1<Long, Integer>) new TwoInterfaces()), new DataType[] { L }, I, false, "two1").accepted());

        ProviderHarness.Outcome viaDefault = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new UsesDefault()), new DataType[] { L }, I, false, "default");
        assertFalse(viaDefault.accepted(), viaDefault::decision);
        assertTrue(viaDefault.decision().contains("interface default method"), viaDefault::decision);

        ProviderHarness.Outcome template = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new TemplateChild()), new DataType[] { L }, L, false, "template");
        assertFalse(template.accepted(), template::decision);
        assertTrue(template.decision().contains("instance method"), template::decision);
        assertEquals(14L, LoweredOracle.reference(new TemplateChild(), 7L), "Java runs the child's step");
    }

    // ============================================================================ helpers

    @SuppressWarnings("unchecked")
    private static <T> T cast(Object o) {
        return (T) o;
    }

    private static Object callUnchecked(UDF1<Object, Object> udf, Object x) {
        try {
            return udf.call(x);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** Accepted by the provider, and the lowered body equals Java called through the interface. */
    static void agreesWithJava(Object udf, List<DataType> inputs, DataType output, Object[][] samples) throws Exception {
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, inputs.toArray(new DataType[0]), output, false, "j" + System.identityHashCode(udf));
        assertTrue(o.accepted(), () -> udf.getClass().getSimpleName() + ": " + o.decision());
        Method body = LoweredOracle.scalarBody(udf, inputs, output);
        for (Object[] sample : samples) {
            assertEquals(LoweredOracle.reference(udf, sample), LoweredOracle.run(body, sample), () -> udf.getClass().getSimpleName() + java.util.Arrays.toString(sample));
        }
    }
}
