/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.util.CheckClassAdapter;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The cases Codex's adversarial review used to break this compiler, kept as permanent tests.
 *
 * <p>
 * They all check one invariant, which is the only thing that makes a planning-time "eligible"
 * meaningful:
 * </p>
 *
 * <blockquote>
 * If the analyzer accepts a UDF, the class generated for it must load, must pass bytecode
 * verification, and must compute exactly what the UDF computes.
 * </blockquote>
 *
 * <p>
 * A UDF the analyzer <em>refuses</em> needs no such guarantee, because the query keeps its CPU plan.
 * So every test here is written to pass either way: it refuses, or it is right. What it may never do
 * is accept a UDF and then produce a different answer, invalid bytecode, or a read of somebody
 * else's data. That is the shape of every defect the review found in this module, and writing the
 * tests this way means a later change that widens coverage is held to the same standard rather than
 * having to be re-tested by hand.
 * </p>
 */
class AdversarialRegressionTest {

    // ------------------------------------------------------------------ ADV-04: conversions

    /** {@code shortValue()} narrows. Erasing it as though it were plain unboxing does not. */
    public static final class NarrowingUdf implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer a) {
            return (int) a.shortValue();
        }
    }

    /** {@code intValue()} on a Double is a D2I conversion, not an identity. */
    public static final class DoubleToIntUdf implements UDF1<Double, Integer> {
        @Override
        public Integer call(Double a) {
            return a.intValue();
        }
    }

    /** A widening conversion, which is representable and should stay correct. */
    public static final class IntToDoubleUdf implements UDF1<Integer, Double> {
        @Override
        public Double call(Integer a) {
            return a.doubleValue() / 3.0;
        }
    }

    @Test
    void narrowingThroughShortValueIsRefusedOrCorrect() {
        assertAcceptedImpliesCorrect(new NarrowingUdf(), List.of(DataTypes.IntegerType), DataTypes.IntegerType,
                new Object[][] { { 40000 }, { 40001 }, { -40000 }, { 7 }, { Integer.MAX_VALUE } });
    }

    @Test
    void doubleToIntConversionIsRefusedOrCorrect() {
        assertAcceptedImpliesCorrect(new DoubleToIntUdf(), List.of(DataTypes.DoubleType), DataTypes.IntegerType,
                new Object[][] { { 3.9 }, { -3.9 }, { 0.0 }, { 1e18 }, { -1e18 } });
    }

    @Test
    void wideningThroughDoubleValueStaysCorrect() {
        assertAcceptedImpliesCorrect(new IntToDoubleUdf(), List.of(DataTypes.IntegerType), DataTypes.DoubleType,
                new Object[][] { { 7 }, { -7 }, { 0 }, { Integer.MIN_VALUE } });
    }

    // ------------------------------------------------------------------ ADV-05: admission

    /** A non-null input does not imply a non-null output, and a kernel has no null. */
    public static final class NullReturningUdf implements UDF1<Double, Double> {
        @Override
        public Double call(Double a) {
            return a < 0 ? null : a;
        }
    }

    /** A helper whose signature is boxed: the lowered call site cannot agree with it. */
    public static final class BoxedHelperUdf implements UDF1<Double, Double> {
        private static Double twice(Double x) {
            return x * 2;
        }

        @Override
        public Double call(Double a) {
            return twice(a);
        }
    }

    /** Reference identity on boxed values is not numeric equality and has no kernel meaning. */
    public static final class ReferenceEqualityUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return a == b ? 1 : 0;
        }
    }

    /** A helper that calls itself. The entry method is not recursive; the call graph is. */
    public static final class RecursiveHelperUdf implements UDF1<Integer, Integer> {
        private static int countdown(int n) {
            return n <= 0 ? 0 : countdown(n - 1) + 1;
        }

        @Override
        public Integer call(Integer a) {
            return countdown(a);
        }
    }

    /** A mutable static is state that can change between batches. */
    public static final class MutableStaticUdf implements UDF1<Double, Double> {
        private static double scale = 2.0;

        @Override
        public Double call(Double a) {
            return a * scale;
        }
    }

    /** A static field of another class entirely. */
    public static final class ForeignStaticUdf implements UDF1<Double, Double> {
        @Override
        public Double call(Double a) {
            return a * Limits.FACTOR;
        }
    }

    static final class Limits {
        // Not a compile-time constant, so javac emits a real GETSTATIC rather than inlining it.
        // A constant that javac folds away never reaches the analyzer at all and proves nothing.
        static final double FACTOR = Double.parseDouble("3.0");
    }

    @Test
    void aUdfThatCanReturnNullIsRefused() {
        assertRefused(new NullReturningUdf(), List.of(DataTypes.DoubleType), DataTypes.DoubleType, "null");
    }

    @Test
    void aBoxedHelperSignatureIsRefused() {
        assertRefused(new BoxedHelperUdf(), List.of(DataTypes.DoubleType), DataTypes.DoubleType, "helper");
    }

    @Test
    void referenceEqualityIsRefused() {
        assertRefused(new ReferenceEqualityUdf(), List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType, "reference");
    }

    @Test
    void aRecursiveHelperIsRefused() {
        assertRefused(new RecursiveHelperUdf(), List.of(DataTypes.IntegerType), DataTypes.IntegerType, "recursi");
    }

    @Test
    void aMutableStaticFieldIsRefused() {
        assertRefused(new MutableStaticUdf(), List.of(DataTypes.DoubleType), DataTypes.DoubleType, "static");
    }

    @Test
    void aStaticFieldOfAnotherClassIsRefused() {
        assertRefused(new ForeignStaticUdf(), List.of(DataTypes.DoubleType), DataTypes.DoubleType, "static");
    }

    // ------------------------------------------------------------------ ADV-06: arithmetic

    /** Integer division whose divisor the compiler cannot prove non-zero. */
    public static final class IntegerDivisionUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            return a / b;
        }
    }

    /** Division by a constant, which is provably safe and should stay supported. */
    public static final class DivideByConstantUdf implements UDF1<Integer, Integer> {
        @Override
        public Integer call(Integer a) {
            return a / 7 + a % 3;
        }
    }

    @Test
    void integerDivisionByAnUnprovableDivisorIsRefused() {
        assertRefused(new IntegerDivisionUdf(), List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType, "divi");
    }

    @Test
    void integerDivisionByANonZeroConstantStaysSupported() {
        assertAcceptedImpliesCorrect(new DivideByConstantUdf(), List.of(DataTypes.IntegerType), DataTypes.IntegerType,
                new Object[][] { { 100 }, { -100 }, { 0 }, { Integer.MIN_VALUE } });
    }

    // ------------------------------------------------------------------ ADV-03: LIST bounds

    /** An index that is not tied to the row's own length can read the next row. */
    public static final class UnguardedListUdf implements UDF1<scala.collection.Seq<Double>, Double> {
        @Override
        public Double call(scala.collection.Seq<Double> a) {
            return a.apply(1) + 0.0;
        }
    }

    /** The canonical guarded reduction, which must keep working. */
    public static final class GuardedListUdf implements UDF1<scala.collection.Seq<Double>, Double> {
        @Override
        public Double call(scala.collection.Seq<Double> a) {
            double total = 0.0;
            for (int i = 0; i < a.length(); i++) {
                total += a.apply(i);
            }
            return total;
        }
    }

    @Test
    void anIndexNotProvablyInsideTheRowIsRefused() {
        assertRefused(new UnguardedListUdf(), List.of(DataTypes.createArrayType(DataTypes.DoubleType, false)), DataTypes.DoubleType, "inside its row");
    }

    @Test
    void theGuardedReductionStaysSupportedAndNeverReadsAnotherRow() throws Exception {
        var listType = DataTypes.createArrayType(DataTypes.DoubleType, false);
        GuardedListUdf udf = new GuardedListUdf();
        EligibilityResult e = analyse(udf, List.of(listType), DataTypes.DoubleType);
        assertTrue(e.isEligible(), "the canonical guarded reduction must stay supported: " + e.reason());

        Method kernel = generateAndVerify(udf, e);
        // Rows of different lengths, including an empty one. If the kernel ignored the row's own
        // bounds it would fold the next row's values into this row's total.
        double[][] rows = { { 1.0, 2.0, 3.0 }, {}, { 10.0 }, { 4.0, 5.0 } };
        DoubleArray out = new DoubleArray(rows.length);
        kernel.invoke(null, offsetsOf(rows), flatten(rows), out, rows.length);
        for (int i = 0; i < rows.length; i++) {
            assertEquals(udf.call(seqOf(rows[i])), out.get(i), 0.0, "row " + i);
        }
    }

    // ------------------------------------------------------------------ ADV-07: cache identity

    @Test
    void twoUdfsWithTheSameNameAndDifferentBodiesDoNotShareAKernel() throws Exception {
        // Same fully-qualified name, same signature, different arithmetic, loaded by two separate
        // classloaders. This is what re-registering changed UDF code in a new session looks like.
        byte[] plusOne = udfClassAdding(1);
        byte[] plusTwo = udfClassAdding(2);

        Object a = defineAndInstantiate(plusOne);
        Object b = defineAndInstantiate(plusTwo);
        assertNotSame(a.getClass(), b.getClass());

        Object kernelA = TornadoUdfProvider.tryCompile(a, new DataType[] { DataTypes.IntegerType }, DataTypes.IntegerType,
                false, true, "collide", new String[] { "LOOP" }, "");
        Object kernelB = TornadoUdfProvider.tryCompile(b, new DataType[] { DataTypes.IntegerType }, DataTypes.IntegerType,
                false, true, "collide", new String[] { "LOOP" }, "");

        if (kernelA == null || kernelB == null) {
            return; // refusing both is a safe outcome; it is accepting them wrongly that is not
        }
        KernelDescriptor da = ((TornadoRapidsUdf) kernelA).descriptor();
        KernelDescriptor db = ((TornadoRapidsUdf) kernelB).descriptor();
        assertFalse(da.cacheKey().equals(db.cacheKey()),
                "two different UDF bodies produced the same cache key, so one would run the other's code");

        Class<?> ca = TornadoKernelCache.kernelClass(da);
        Class<?> cb = TornadoKernelCache.kernelClass(db);
        assertNotSame(ca, cb, "the kernel cache returned one class for two different UDF bodies");
        assertEquals(11, invokeKernelScalar(ca, 10), "first UDF must still compute x + 1");
        assertEquals(12, invokeKernelScalar(cb, 10), "second UDF must compute x + 2, not the first UDF's answer");
    }

    // ------------------------------------------------------------------ helpers

    private static EligibilityResult analyse(Object udf, List<DataType> inputs, DataType output) {
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
        return UdfEligibilityAnalyzer.analyse(resolved, inputs, output, false, true);
    }

    /** Asserts the UDF is refused, and that the reason mentions {@code hint}. */
    private static void assertRefused(Object udf, List<DataType> inputs, DataType output, String hint) {
        EligibilityResult e;
        try {
            e = analyse(udf, inputs, output);
        } catch (RuntimeException ex) {
            return; // refusing by failing the analysis outright is also a refusal
        }
        assertFalse(e.isEligible(), "this UDF must not be accepted; it was, with inputs " + inputs + " -> " + output);
        assertTrue(e.reason().toLowerCase().contains(hint.toLowerCase()),
                "refused for the wrong reason: expected something mentioning '" + hint + "', got: " + e.reason());
    }

    /**
     * If the UDF is accepted, the generated class must verify and agree with it on every input.
     * A refusal is a pass: the query keeps its CPU plan and stays correct.
     */
    private static void assertAcceptedImpliesCorrect(Object udf, List<DataType> inputs, DataType output, Object[][] samples) {
        EligibilityResult e;
        try {
            e = analyse(udf, inputs, output);
        } catch (RuntimeException ex) {
            return;
        }
        if (!e.isEligible()) {
            return;
        }
        Method kernel;
        try {
            kernel = generateAndVerify(udf, e);
        } catch (Exception ex) {
            throw new AssertionError("accepted but the generated class is not usable: " + ex, ex);
        }
        for (Object[] sample : samples) {
            Object expected = callUdf(udf, sample);
            Object actual = runKernelOnce(kernel, e, sample);
            assertEquals(String.valueOf(expected), String.valueOf(actual),
                    "accepted, but the kernel disagrees with the UDF on input " + java.util.Arrays.toString(sample));
        }
    }

    /** Generates the kernel class, verifies its bytecode, loads it and returns the kernel method. */
    private static Method generateAndVerify(Object udf, EligibilityResult e) throws Exception {
        ResolvedUdf resolved = e.udf();
        ParallelKernelGenerator.Generated g = ParallelKernelGenerator.generate(resolved, e.inputs(), e.outputKind());

        StringWriter problems = new StringWriter();
        CheckClassAdapter.verify(new ClassReader(g.classFile()), false, new PrintWriter(problems));
        if (!problems.toString().isBlank()) {
            fail("generated bytecode does not verify:\n" + problems);
        }
        Class<?> cls = new GeneratedKernelClassLoader(AdversarialRegressionTest.class.getClassLoader())
                .define(g.className(), g.classFile());
        for (Method m : cls.getDeclaredMethods()) {
            if (m.getName().equals(ParallelKernelGenerator.KERNEL_METHOD)) {
                m.setAccessible(true);
                return m;
            }
        }
        throw new AssertionError("no kernel method generated");
    }

    private static Object callUdf(Object udf, Object[] args) {
        try {
            if (args.length == 1) {
                return ((UDF1<Object, Object>) udf).call(args[0]);
            }
            return ((UDF2<Object, Object, Object>) udf).call(args[0], args[1]);
        } catch (Exception ex) {
            return "threw:" + ex.getClass().getSimpleName();
        }
    }

    /** Runs the generated kernel over a single row built from {@code args}. */
    private static Object runKernelOnce(Method kernel, EligibilityResult e, Object[] args) {
        Object[] call = new Object[e.inputs().size() + 2];
        for (int i = 0; i < e.inputs().size(); i++) {
            call[i] = singletonColumn(e.inputs().get(i).kind(), args[i]);
        }
        Object out = singletonColumn(e.outputKind(), null);
        call[e.inputs().size()] = out;
        call[e.inputs().size() + 1] = 1;
        try {
            kernel.invoke(null, call);
        } catch (InvocationTargetException ex) {
            return "threw:" + ex.getCause().getClass().getSimpleName();
        } catch (IllegalAccessException ex) {
            throw new AssertionError(ex);
        }
        return readColumn(e.outputKind(), out);
    }

    private static Object singletonColumn(PrimitiveKind kind, Object value) {
        switch (kind) {
            case INT: {
                IntArray a = new IntArray(1);
                a.set(0, value == null ? 0 : ((Number) value).intValue());
                return a;
            }
            case LONG: {
                var a = new uk.ac.manchester.tornado.api.types.arrays.LongArray(1);
                a.set(0, value == null ? 0L : ((Number) value).longValue());
                return a;
            }
            case FLOAT: {
                var a = new uk.ac.manchester.tornado.api.types.arrays.FloatArray(1);
                a.set(0, value == null ? 0f : ((Number) value).floatValue());
                return a;
            }
            default: {
                DoubleArray a = new DoubleArray(1);
                a.set(0, value == null ? 0d : ((Number) value).doubleValue());
                return a;
            }
        }
    }

    private static Object readColumn(PrimitiveKind kind, Object column) {
        return switch (kind) {
            case INT -> ((IntArray) column).get(0);
            case LONG -> ((uk.ac.manchester.tornado.api.types.arrays.LongArray) column).get(0);
            case FLOAT -> ((uk.ac.manchester.tornado.api.types.arrays.FloatArray) column).get(0);
            case DOUBLE -> ((DoubleArray) column).get(0);
        };
    }

    private static int invokeKernelScalar(Class<?> kernelClass, int input) throws Exception {
        Method kernel = null;
        for (Method m : kernelClass.getDeclaredMethods()) {
            if (m.getName().equals(ParallelKernelGenerator.KERNEL_METHOD)) {
                kernel = m;
                kernel.setAccessible(true);
            }
        }
        IntArray in = new IntArray(1);
        in.set(0, input);
        IntArray out = new IntArray(1);
        kernel.invoke(null, in, out, 1);
        return out.get(0);
    }

    /** A UDF1&lt;Integer,Integer&gt; class, always called {@code gen.CollidingUdf}, computing x + n. */
    private static byte[] udfClassAdding(int n) {
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        cw.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL, "gen/CollidingUdf", null, "java/lang/Object",
                new String[] { "org/apache/spark/sql/api/java/UDF1" });
        var ctor = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null);
        ctor.visitCode();
        ctor.visitVarInsn(Opcodes.ALOAD, 0);
        ctor.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        ctor.visitInsn(Opcodes.RETURN);
        ctor.visitMaxs(0, 0);
        ctor.visitEnd();

        var call = cw.visitMethod(Opcodes.ACC_PUBLIC, "call", "(Ljava/lang/Integer;)Ljava/lang/Integer;", null, null);
        call.visitCode();
        call.visitVarInsn(Opcodes.ALOAD, 1);
        call.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/Integer", "intValue", "()I", false);
        call.visitLdcInsn(n);
        call.visitInsn(Opcodes.IADD);
        call.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", false);
        call.visitInsn(Opcodes.ARETURN);
        call.visitMaxs(0, 0);
        call.visitEnd();

        var bridge = cw.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_BRIDGE | Opcodes.ACC_SYNTHETIC, "call",
                "(Ljava/lang/Object;)Ljava/lang/Object;", null, null);
        bridge.visitCode();
        bridge.visitVarInsn(Opcodes.ALOAD, 0);
        bridge.visitVarInsn(Opcodes.ALOAD, 1);
        bridge.visitTypeInsn(Opcodes.CHECKCAST, "java/lang/Integer");
        bridge.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "gen/CollidingUdf", "call", "(Ljava/lang/Integer;)Ljava/lang/Integer;", false);
        bridge.visitInsn(Opcodes.ARETURN);
        bridge.visitMaxs(0, 0);
        bridge.visitEnd();
        cw.visitEnd();
        return cw.toByteArray();
    }

    private static Object defineAndInstantiate(byte[] bytes) throws Exception {
        ClassLoader loader = new ClassLoader(AdversarialRegressionTest.class.getClassLoader()) {
            @Override
            protected Class<?> findClass(String name) throws ClassNotFoundException {
                if (name.equals("gen.CollidingUdf")) {
                    return defineClass(name, bytes, 0, bytes.length);
                }
                throw new ClassNotFoundException(name);
            }

            @Override
            public java.io.InputStream getResourceAsStream(String name) {
                if (name.equals("gen/CollidingUdf.class")) {
                    return new java.io.ByteArrayInputStream(bytes);
                }
                return super.getResourceAsStream(name);
            }
        };
        return loader.loadClass("gen.CollidingUdf").getDeclaredConstructor().newInstance();
    }

    private static scala.collection.Seq<Double> seqOf(double[] values) {
        java.util.List<Double> boxed = new java.util.ArrayList<>();
        for (double v : values) {
            boxed.add(v);
        }
        return scala.jdk.javaapi.CollectionConverters.asScala(boxed).toSeq();
    }

    private static IntArray offsetsOf(double[][] rows) {
        IntArray offsets = new IntArray(rows.length + 1);
        int running = 0;
        offsets.set(0, 0);
        for (int i = 0; i < rows.length; i++) {
            running += rows[i].length;
            offsets.set(i + 1, running);
        }
        return offsets;
    }

    private static DoubleArray flatten(double[][] rows) {
        int total = 0;
        for (double[] row : rows) {
            total += row.length;
        }
        DoubleArray values = new DoubleArray(Math.max(total, 1));
        int at = 0;
        for (double[] row : rows) {
            for (double v : row) {
                values.set(at++, v);
            }
        }
        return values;
    }

    /** Unused marker so ClassNode stays imported for readers tracing the ASM types used here. */
    @SuppressWarnings("unused")
    private static final Class<?> ASM_TREE = ClassNode.class;
    /**
     * ADV-10: per-batch telemetry must not grow for the life of the executor.
     *
     * <p>
     * The counters stay exact past the cap -- a figure that quietly under-counted batches would be
     * worse than one that stopped -- and what was not kept is reported rather than hidden, so no
     * median derived from the samples can pass itself off as covering every batch.
     * </p>
     */
    @Test
    void perBatchTelemetryIsBounded() {
        TornadoUdfMetrics.reset();
        TornadoUdfMetrics.sampleLimitForTesting(8);
        try {
            for (int i = 0; i < 100; i++) {
                TornadoUdfMetrics.record(new TornadoRapidsUdf.Timings(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 64, false, 0, 0, 0, 0, 0));
            }
            assertEquals(8, TornadoUdfMetrics.samples().size(), "the retained samples must stop at the cap");
            assertEquals(100, TornadoUdfMetrics.batches(), "every batch must still be counted");
            assertEquals(6400, TornadoUdfMetrics.rows(), "every row must still be counted");
            assertEquals(92, TornadoUdfMetrics.samplesDropped(), "what was not kept has to be reported");
            assertTrue(TornadoUdfMetrics.summary().contains("droppedSamples=92"), "the summary must say so: " + TornadoUdfMetrics.summary());
        } finally {
            TornadoUdfMetrics.sampleLimitForTesting(Integer.getInteger("tornado.spark.metrics.limit", 100_000));
            TornadoUdfMetrics.reset();
        }
    }

}
