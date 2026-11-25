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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.internal.ToScalaUDF$;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Codex's second review, finding B: a field is folded into the CUDA only when its value in an
 * executor's JVM is fixed by what the driver serializes -- not when two round trips on the driver
 * happen to agree.
 *
 * <p>
 * The central counterexample is run across two genuinely separate JVM processes: this one, as the
 * driver, and a child started with a different {@code review.factor}, as the executor. The child
 * deserializes exactly the bytes Spark would ship (the {@code ToScalaUDF} adapter around the UDF)
 * and calls it through {@code Function1.apply}, as an executor's task does.
 * </p>
 */
class ExecutorInvariantFieldsTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    @TempDir
    Path dir;

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    // ============================================================================ fixtures: safe

    public static final class Offset implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int offset;

        public Offset(int offset) {
            this.offset = offset;
        }

        @Override
        public Integer call(Integer x) {
            return x + offset;
        }
    }

    /** A long, a boolean and an int, all from the constructor. */
    public static final class Masked implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;
        private final long mask;
        private final boolean negate;
        private final int shift;

        public Masked(long mask, boolean negate, int shift) {
            this.mask = mask;
            this.negate = negate;
            this.shift = shift;
        }

        @Override
        public Long call(Long x) {
            long v = (x & mask) >>> shift;
            return negate ? -v : v;
        }
    }

    /** Fields declared in a serializable parent: default serialization carries them too. */
    public static class SerializableBase implements Serializable {
        private static final long serialVersionUID = 1L;
        protected final int base;

        SerializableBase(int base) {
            this.base = base;
        }
    }

    public static final class DerivedOffset extends SerializableBase implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int extra;

        public DerivedOffset(int base, int extra) {
            super(base);
            this.extra = extra;
        }

        @Override
        public Integer call(Integer x) {
            return x * base + extra;
        }
    }

    // ============================================================================ fixtures: executor-dependent

    /** Codex's counterexample. */
    public static final class ResolvedFactor implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
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

    /** readObject overwrites the (final) field from the executor's environment. */
    public static final class ReadObjectFactor implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int factor;

        public ReadObjectFactor(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            try {
                java.lang.reflect.Field f = ReadObjectFactor.class.getDeclaredField("factor");
                f.setAccessible(true);
                f.setInt(this, Integer.getInteger("review.factor", factor));
            } catch (ReflectiveOperationException e) {
                throw new IOException(e);
            }
        }
    }

    /** writeObject puts a different value in the stream than the driver instance holds. */
    public static final class WrittenFactor implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int factor;

        public WrittenFactor(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private void writeObject(ObjectOutputStream out) throws IOException {
            ObjectOutputStream.PutField fields = out.putFields();
            fields.put("factor", factor * 100);
            out.writeFields();
        }
    }

    /** readObject that only restores the default fields: the executor sees the driver's values. */
    public static final class DefaultReadObject implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int factor;

        public DefaultReadObject(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
        }
    }

    /** A field of a non-serializable superclass is set by its no-arg constructor, on the executor. */
    public static class EnvironmentBase {
        protected final int base;

        public EnvironmentBase() {
            this.base = Integer.getInteger("review.factor", 9);
        }
    }

    public static final class EnvironmentDerived extends EnvironmentBase implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return x + base;
        }
    }

    /** readResolve picks the class from the executor's environment; call reads no field at all. */
    public static final class ClassSwitch implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return x + 1;
        }

        private Object readResolve() {
            return Integer.getInteger("review.factor", 9) == 9 ? this : new Minus();
        }
    }

    public static final class Minus implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return x - 1;
        }
    }

    /** A pure readResolve over a UDF that reads no field: the class cannot change, so it compiles. */
    public static final class PureResolveNoFields implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return x * 3;
        }

        private Object readResolve() {
            return this;
        }
    }

    /** A final reference to a mutable object: not a constant, whatever the reference is. */
    public static final class MutableReference implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int[] factor = { 9 };

        @Override
        public Integer call(Integer x) {
            return x + factor[0];
        }
    }

    /** A transient final field: zero on every executor, whatever the driver's constructor set. */
    public static final class TransientFactor implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final transient int factor;

        public TransientFactor() {
            this.factor = 9;
        }

        @Override
        public Integer call(Integer x) {
            return x + factor;
        }
    }

    /** Two final fields of an inherited implementation, read through the parent's call. */
    public static class ParentWithFields implements UDF2<Integer, Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int a;
        private final int b;

        public ParentWithFields(int a, int b) {
            this.a = a;
            this.b = b;
        }

        @Override
        public Integer call(Integer x, Integer y) {
            return x * a + y * b;
        }
    }

    public static final class ChildOfFields extends ParentWithFields {
        private static final long serialVersionUID = 1L;

        public ChildOfFields() {
            super(3, 5);
        }

        public int call(int x, int y) {
            return 0;
        }
    }

    // ============================================================================ the executor JVM

    /** Deserializes the shipped function and applies it, as a task on an executor does. */
    public static final class ExecutorMain {
        public static void main(String[] args) throws Exception {
            Object function;
            try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(Path.of(args[0])))) {
                function = in.readObject();
            }
            @SuppressWarnings("unchecked")
            scala.Function1<Object, Object> f = (scala.Function1<Object, Object>) function;
            StringBuilder out = new StringBuilder();
            for (int i = 1; i < args.length; i++) {
                out.append(f.apply(Integer.valueOf(args[i]))).append(i + 1 < args.length ? "," : "");
            }
            System.out.println("RESULT " + out);
        }
    }

    /** What a separate JVM started with {@code -Dreview.factor=property} computes for {@code xs}. */
    private String inExecutorJvm(UDF1<Integer, Integer> udf, String property, int... xs) throws Exception {
        return inExecutorJvm(udf, property, null, xs);
    }

    private String inExecutorJvm(UDF1<Integer, Integer> udf, String property, String expectedFailure, int... xs) throws Exception {
        Path shipped = dir.resolve("udf-" + System.nanoTime() + ".bin");
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(ToScalaUDF$.MODULE$.apply(udf));
        }
        Files.write(shipped, bytes.toByteArray());
        List<String> command = new ArrayList<>(List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(), "-cp", System.getProperty("java.class.path")));
        if (property != null) {
            command.add("-Dreview.factor=" + property);
        }
        command.add(ExecutorMain.class.getName());
        command.add(shipped.toString());
        for (int x : xs) {
            command.add(Integer.toString(x));
        }
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes());
        assertTrue(process.waitFor(120, TimeUnit.SECONDS), "the executor JVM did not finish");
        assertNotEquals(ProcessHandle.current().pid(), process.pid());
        if (expectedFailure != null) {
            assertNotEquals(0, process.exitValue(), output);
            assertTrue(output.contains(expectedFailure), output);
            return output;
        }
        assertEquals(0, process.exitValue(), output);
        return output.lines().filter(l -> l.startsWith("RESULT ")).findFirst().orElseThrow(() -> new AssertionError(output)).substring(7);
    }

    // ============================================================================ tests

    @Test
    void recordConstructorsRunOnTheDeserializingJvm() throws Exception {
        var worker = new CoverageCorpus.WorkerOffset(9);
        assertEquals("16", inExecutorJvm(worker, null, 7));
        assertEquals("84", inExecutorJvm(worker, "77", 7));
        assertEquals("7", inExecutorJvm(new CoverageCorpus.GuardedMetadata(-1, "label"), "77", 7));
        var nested = new CoverageCorpus.WithGuard(3, new CoverageCorpus.WorkerGuard(1));
        assertEquals("10", inExecutorJvm(nested, null, 7));
        inExecutorJvm(nested, "77", "IllegalStateException: worker guard", 7);
        assertEquals("7", inExecutorJvm(new CoverageCorpus.Nonnegative(-1), "77", 7));
    }

    @Test
    void codexReadResolveCounterexampleAcrossTwoJvms() throws Exception {
        assertEquals(null, System.getProperty("review.factor"), "the driver JVM runs without the property");
        ResolvedFactor udf = new ResolvedFactor(9);
        // The two processes disagree, which is the whole point: the executor's value is decided there.
        assertEquals("16", inExecutorJvm(udf, null, 7));
        assertEquals("84", inExecutorJvm(udf, "77", 7));
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(udf), new DataType[] { I }, I, false, "codex_b");
        assertFalse(o.accepted(), () -> "must not compile x + 9: " + o.cuda());
        assertTrue(o.decision().contains("instance field"), o::decision);
    }

    @Test
    void otherExecutorDependentStateIsNotFolded() throws Exception {
        // Each of these computes something else in a JVM with the property set; none may be compiled
        // with the driver's value.
        assertEquals("84", inExecutorJvm(new ReadObjectFactor(9), "77", 7));
        assertEquals("84", inExecutorJvm(new EnvironmentDerived(), "77", 7));
        assertEquals("6", inExecutorJvm(new ClassSwitch(), "77", 7));
        for (UDF1<Integer, Integer> udf : List.<UDF1<Integer, Integer>>of(new ReadObjectFactor(9), new EnvironmentDerived(), new ClassSwitch(),
                new MutableReference())) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(udf), new DataType[] { I }, I, false, udf.getClass().getSimpleName());
            assertFalse(o.accepted(), () -> udf.getClass().getSimpleName() + " compiled: " + o.cuda());
        }
        String classSwitch = TornadoUdfProvider.decisions().get("ClassSwitch");
        assertTrue(classSwitch.contains("readResolve"), classSwitch);
    }

    static class EmptyParent {
    }

    public static final class StableChild extends EmptyParent implements UDF1<Integer, Integer> {
        private final int offset;

        StableChild(int offset) {
            this.offset = offset;
        }

        @Override
        public Integer call(Integer x) {
            return x + offset;
        }
    }

    public static final class StatefulWriter implements UDF1<Integer, Integer> {
        private final int factor;
        private int writes;

        public StatefulWriter(int factor) {
            this.factor = factor;
        }

        @Override
        public Integer call(Integer x) {
            return factor + x;
        }

        private void writeObject(ObjectOutputStream out) throws IOException {
            out.putFields().put("factor", ++writes);
            out.writeFields();
        }
    }

    @Test
    void planningCannotAdvanceStatefulSerialization() {
        StatefulWriter udf = new StatefulWriter(9);
        ProviderHarness.Outcome o = ProviderHarness.cudf(udf, new DataType[] { I }, I, false, "writer");
        assertFalse(o.accepted(), o::decision);
        assertEquals(0, udf.writes);
    }

    public static final class ExecutorOnlyFailure implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer x) {
            return x + 1;
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            if (Integer.getInteger("review.factor", 9) == 77) {
                throw new java.io.InvalidObjectException("executor refused the UDF");
            }
        }
    }

    public static final class WithCallback implements UDF1<Integer, Integer> {
        private final ExecutorOnlyFailure child = new ExecutorOnlyFailure();

        @Override
        public Integer call(Integer x) {
            return x + 1;
        }
    }

    @Test
    void anUnusedFieldDoesNotMakeExecutorDeserializationEffectsUnobservable() throws Exception {
        for (UDF1<Integer, Integer> udf : List.of(new ExecutorOnlyFailure(), new WithCallback())) {
            assertEquals("8", inExecutorJvm(udf, null, 7));
            inExecutorJvm(udf, "77", "java.io.InvalidObjectException: executor refused the UDF", 7);
            ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(udf), new DataType[] { I }, I, false, "serialization_effect");
            assertFalse(o.accepted(), o::decision);
        }
    }

    @Test
    void constructorConstantsStayEligibleAndMatchAnExecutorJvm() throws Exception {
        StableChild stable = new StableChild(12);
        assertEquals("19", inExecutorJvm(stable, "77", 7));
        assertTrue(ProviderHarness.cudf(stable, new DataType[] { I }, I, false, "empty_parent").accepted());
        Offset offset = new Offset(41);
        ProviderHarness.Outcome o = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(offset), new DataType[] { I }, I, false, "offset");
        assertTrue(o.accepted(), o::decision);
        Method body = LoweredOracle.scalarBody(offset, List.of(I), I);
        String executor = inExecutorJvm(offset, "77", 1, -50, Integer.MAX_VALUE);
        assertEquals(LoweredOracle.run(body, 1) + "," + LoweredOracle.run(body, -50) + "," + LoweredOracle.run(body, Integer.MAX_VALUE), executor);

        // A pure readResolve, and a transient field, are decided identically in the executor JVM.
        assertEquals("21", inExecutorJvm(new PureResolveNoFields(), "77", 7));
        assertTrue(ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new PureResolveNoFields()), new DataType[] { I }, I, false, "pure").accepted());
        TransientFactor transientFactor = new TransientFactor();
        assertEquals("7", inExecutorJvm(transientFactor, "77", 7));
        assertEquals(7, LoweredOracle.run(LoweredOracle.scalarBody(transientFactor, List.of(I), I), 7));
    }

    /** Deterministic hooks are followed: the folded value is the one the executor JVM computes. */
    @Test
    void deterministicHooksFoldTheExecutorsValue() throws Exception {
        WrittenFactor written = new WrittenFactor(9);
        assertEquals("907", inExecutorJvm(written, "77", 7), "the stream carries 900, not the instance's 9");
        ProviderHarness.Outcome w = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(written), new DataType[] { I }, I, false, "written");
        assertTrue(w.accepted(), w::decision);
        assertTrue(w.cuda().contains("900"), w::cuda);
        assertEquals(907, LoweredOracle.run(LoweredOracle.scalarBody(written, List.of(I), I), 7));

        DefaultReadObject plain = new DefaultReadObject(12);
        assertEquals("19", inExecutorJvm(plain, "77", 7));
        assertEquals(19, LoweredOracle.run(LoweredOracle.scalarBody(plain, List.of(I), I), 7));
    }

    @Test
    void longBooleanAndSeveralConstants() throws Exception {
        for (Masked udf : new Masked[] { new Masked(0xFFFF_0000L, false, 16), new Masked(-1L, true, 3) }) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(udf, new DataType[] { L }, L, false, "masked" + udf.shift);
            assertTrue(o.accepted(), o::decision);
            Method body = LoweredOracle.scalarBody(udf, List.of(L), L);
            for (long x : new long[] { 0, 0x1234_5678_9ABCL, -1, Long.MIN_VALUE }) {
                assertEquals(LoweredOracle.reference(udf, x), LoweredOracle.run(body, x));
            }
        }
        DerivedOffset derived = new DerivedOffset(4, -3);
        assertTrue(ProviderHarness.cudf(derived, new DataType[] { I }, I, false, "derived").accepted());
        assertEquals(LoweredOracle.reference(derived, 10), LoweredOracle.run(LoweredOracle.scalarBody(derived, List.of(I), I), 10));
        ChildOfFields child = new ChildOfFields();
        assertTrue(ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(child), new DataType[] { I, I }, I, false, "child_fields").accepted());
        assertEquals(LoweredOracle.reference(child, 2, 7), LoweredOracle.run(LoweredOracle.scalarBody(child, List.of(I, I), I), 2, 7));
    }

    @Test
    void differentInstancesNeverShareCompiledCode() {
        ProviderHarness.Outcome three = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new Offset(3)), new DataType[] { I }, I, false, "o3");
        ProviderHarness.Outcome five = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new Offset(5)), new DataType[] { I }, I, false, "o5");
        ProviderHarness.Outcome threeAgain = ProviderHarness.cudf(ToScalaUDF$.MODULE$.apply(new Offset(3)), new DataType[] { I }, I, false, "o3b");
        assertTrue(three.accepted() && five.accepted() && threeAgain.accepted());
        assertNotEquals(three.cuda(), five.cuda());
        assertFalse(five.decision().contains("cached"), five::decision);
        assertEquals(three.cuda(), threeAgain.cuda());
        ResolvedUdf a = ConstantFieldFolding.fold(UdfResolver.resolve(new Offset(3)).orElseThrow());
        ResolvedUdf b = ConstantFieldFolding.fold(UdfResolver.resolve(new Offset(5)).orElseThrow());
        List<InputSpec> in = List.of(InputSpec.scalar(PrimitiveKind.INT));
        assertNotEquals(KernelIdentity.of(a, in, PrimitiveKind.INT), KernelIdentity.of(b, in, PrimitiveKind.INT));
    }
}
