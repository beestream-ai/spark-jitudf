/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.util.List;

import org.junit.jupiter.api.Test;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;

import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Checks the lowering and the generated kernel without a GPU, by running the generated method on
 * the CPU over ordinary host-allocated arrays and comparing it against the UDF itself.
 *
 * <p>
 * This is where a mistake in the bytecode transform shows up as a wrong number rather than as a
 * mysterious kernel result: the generated {@code kernel} method is plain Java until TornadoVM
 * compiles it, so invoking it directly exercises the slot remapping, the unboxing removal and the
 * return-opcode rewrite against the original method's own answers.
 * </p>
 */
class KernelGenerationTest {

    /** A loop, which is what the RAPIDS compiler declines and this path exists for. */
    public static final class LoopUdf implements UDF2<Double, Double, Double> {
        @Override
        public Double call(Double a, Double b) {
            double acc = 0.0;
            double x = a;
            double y = b;
            for (int i = 0; i < 17; i++) {
                acc = acc * 0.5 + x * y;
                x = x * 1.000001 + 0.5;
                y = y * 0.999999 - 0.25;
            }
            return acc;
        }
    }

    /** Branches, maths calls and a long-lived local, to exercise the slot remapping. */
    public static final class BranchyUdf implements UDF2<Double, Double, Double> {
        @Override
        public Double call(Double a, Double b) {
            double scale = a > b ? 2.0 : 0.5;
            double norm = Math.sqrt(a * a + b * b);
            if (norm > 100.0) {
                norm = Math.cbrt(norm);
            }
            return scale * norm + Math.min(a, b);
        }
    }

    /** A static helper, copied into the generated class and called from the lowered body. */
    public static final class HelperUdf implements UDF2<Double, Double, Double> {
        private static double blend(double a, double b, int i) {
            return a * 0.5 + b * 0.25 + i * 0.125;
        }

        @Override
        public Double call(Double a, Double b) {
            double acc = 0.0;
            for (int i = 0; i < 9; i++) {
                acc = blend(acc + a, b, i);
            }
            return acc;
        }
    }

    /** Ints in and ints out, so the lowering is exercised on one-slot primitives too. */
    public static final class IntUdf implements UDF2<Integer, Integer, Integer> {
        @Override
        public Integer call(Integer a, Integer b) {
            int acc = 0;
            for (int i = 0; i < 11; i++) {
                acc = acc * 3 + a - b + i;
            }
            return acc;
        }
    }

    /** A string result: nothing to generate, and the analysis has to say so. */
    public static final class StringUdf implements UDF1<Double, String> {
        @Override
        public String call(Double a) {
            return a > 0 ? "yes" : "no";
        }
    }

    /** Allocation, which a kernel cannot do. */
    public static final class AllocatingUdf implements UDF1<Double, Double> {
        @Override
        public Double call(Double a) {
            double[] buffer = new double[4];
            buffer[0] = a;
            return buffer[0] * 2;
        }
    }

    /** State the kernel does not have. */
    public static final class StatefulUdf implements UDF1<Double, Double> {
        private double accumulated;

        @Override
        public Double call(Double a) {
            accumulated += a;
            return accumulated;
        }
    }

    @Test
    void loopUdfLowersToTheSameArithmetic() throws Exception {
        assertMatches(new LoopUdf(), new double[] { 1.5, -3.25, 0.0, 1e6, -1e-6 }, new double[] { 2.0, 7.5, -1.0, 3.5, 2.25 });
    }

    @Test
    void branchesAndMathCallsLowerCorrectly() throws Exception {
        assertMatches(new BranchyUdf(), new double[] { 1.5, 300.0, -2.0, 0.5 }, new double[] { 2.0, 5.0, -8.0, 0.25 });
    }

    @Test
    void staticHelpersAreCopiedIntoTheGeneratedClass() throws Exception {
        assertMatches(new HelperUdf(), new double[] { 1.0, 2.5, -4.0 }, new double[] { 3.0, -1.5, 0.25 });
    }

    @Test
    void intUdfLowersToTheSameArithmetic() throws Exception {
        UDF2<Integer, Integer, Integer> udf = new IntUdf();
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow();
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.IntegerType, DataTypes.IntegerType), DataTypes.IntegerType, false, true);
        assertTrue(eligibility.isEligible(), eligibility.reason());

        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        Class<?> kernelClass = new GeneratedKernelClassLoader(getClass().getClassLoader()).define(generated.className(), generated.classFile());
        Method kernel = kernelMethod(kernelClass);

        int rows = 4;
        IntArray a = new IntArray(rows);
        IntArray b = new IntArray(rows);
        IntArray out = new IntArray(rows);
        int[] left = { 1, -5, 1000, 0 };
        int[] right = { 2, 7, -3, 0 };
        for (int i = 0; i < rows; i++) {
            a.set(i, left[i]);
            b.set(i, right[i]);
        }
        kernel.invoke(null, a, b, out, rows);
        for (int i = 0; i < rows; i++) {
            assertEquals(udf.call(left[i], right[i]).intValue(), out.get(i), "row " + i);
        }
    }

    @Test
    void theGeneratedKernelLoopCarriesTheParallelAnnotation() throws Exception {
        ResolvedUdf resolved = UdfResolver.resolve(new LoopUdf()).orElseThrow();
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType, DataTypes.DoubleType), DataTypes.DoubleType, false, true);
        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        // TornadoVM finds @Parallel in RuntimeVisibleTypeAnnotations on a local variable; without it
        // the kernel would compile and then run on one thread.
        String constants = new String(generated.classFile(), java.nio.charset.StandardCharsets.ISO_8859_1);
        assertTrue(constants.contains("Luk/ac/manchester/tornado/api/annotations/Parallel;"), "the generated kernel does not reference @Parallel");
        assertTrue(constants.contains("RuntimeVisibleTypeAnnotations"), "the generated kernel has no type annotations attribute");
    }

    @Test
    void aStringReturnIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new StringUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType), DataTypes.StringType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("fixed-width numeric"), result.reason());
    }

    @Test
    void allocationIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new AllocatingUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("allocates"), result.reason());
    }

    @Test
    void instanceStateIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new StatefulUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("instance field"), result.reason());
    }

    @Test
    void nullableInputIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new LoopUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType, DataTypes.DoubleType), DataTypes.DoubleType, true, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("nullable"), result.reason());
    }

    @Test
    void aNonDeterministicUdfIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new LoopUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType, DataTypes.DoubleType), DataTypes.DoubleType, false, false);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("deterministic"), result.reason());
    }

    @Test
    void aSchemaThatDisagreesWithTheMethodIsRefused() {
        ResolvedUdf resolved = UdfResolver.resolve(new LoopUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType, DataTypes.IntegerType), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("argument 1"), result.reason());
    }

    /** NVIDIA's published example shape: two per-row vectors reduced to one number. */
    public static final class CosineSimilarityUdf implements UDF2<double[], double[], Double> {
        @Override
        public Double call(double[] a, double[] b) {
            int n = Math.min(a.length, b.length);
            double dot = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            for (int i = 0; i < n; i++) {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            double denominator = Math.sqrt(normA) * Math.sqrt(normB);
            return denominator == 0.0 ? 0.0 : dot / denominator;
        }
    }

    /** One list argument and one scalar, to check the mixed shape. */
    public static final class ScaledSumUdf implements UDF2<double[], Double, Double> {
        @Override
        public Double call(double[] values, Double scale) {
            double total = 0.0;
            for (int i = 0; i < values.length; i++) {
                total += values[i] * scale;
            }
            return total;
        }
    }

    /** An array used as a value rather than indexed: it has no device representation. */
    public static final class EscapingArrayUdf implements UDF1<double[], Double> {
        private static double first(double[] values) {
            return values[0];
        }

        @Override
        public Double call(double[] values) {
            return first(values);
        }
    }

    @Test
    void cosineSimilarityOverTwoListsLowersCorrectly() throws Exception {
        CosineSimilarityUdf udf = new CosineSimilarityUdf();
        var listOfDouble = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(listOfDouble, listOfDouble), DataTypes.DoubleType, false, true);
        assertTrue(eligibility.isEligible(), eligibility.reason());
        assertTrue(eligibility.inputs().get(0).isList());

        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        Class<?> kernelClass = new GeneratedKernelClassLoader(getClass().getClassLoader()).define(generated.className(), generated.classFile());
        Method kernel = kernelMethod(kernelClass);

        // Three rows of different lengths, laid out exactly as cuDF would: offsets plus one
        // flattened child per column.
        double[][] left = { { 1.0, 2.0, 3.0 }, { 0.5 }, { 4.0, -1.0, 0.25, 8.0 } };
        double[][] right = { { 4.0, 5.0, 6.0 }, { 2.0 }, { 1.0, 1.0, 1.0, 1.0 } };
        int rows = left.length;

        IntArray leftOffsets = offsetsOf(left);
        DoubleArray leftValues = flatten(left);
        IntArray rightOffsets = offsetsOf(right);
        DoubleArray rightValues = flatten(right);
        DoubleArray out = new DoubleArray(rows);

        kernel.invoke(null, leftOffsets, leftValues, rightOffsets, rightValues, out, rows);
        for (int i = 0; i < rows; i++) {
            assertEquals(udf.call(left[i], right[i]).doubleValue(), out.get(i), 0.0, "row " + i);
        }
    }

    @Test
    void aListAndAScalarTogetherLowerCorrectly() throws Exception {
        ScaledSumUdf udf = new ScaledSumUdf();
        var listOfDouble = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow();
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(listOfDouble, DataTypes.DoubleType), DataTypes.DoubleType, false, true);
        assertTrue(eligibility.isEligible(), eligibility.reason());

        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        Class<?> kernelClass = new GeneratedKernelClassLoader(getClass().getClassLoader()).define(generated.className(), generated.classFile());
        Method kernel = kernelMethod(kernelClass);

        double[][] lists = { { 1.0, 2.0 }, {}, { 3.0, 4.0, 5.0 } };
        double[] scales = { 2.0, 7.0, 0.5 };
        int rows = lists.length;
        DoubleArray scaleColumn = new DoubleArray(rows);
        for (int i = 0; i < rows; i++) {
            scaleColumn.set(i, scales[i]);
        }
        DoubleArray out = new DoubleArray(rows);
        kernel.invoke(null, offsetsOf(lists), flatten(lists), scaleColumn, out, rows);
        for (int i = 0; i < rows; i++) {
            assertEquals(udf.call(lists[i], scales[i]).doubleValue(), out.get(i), 0.0, "row " + i);
        }
    }

    @Test
    void anArrayThatEscapesIsRefused() {
        var listOfDouble = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ResolvedUdf resolved = UdfResolver.resolve(new EscapingArrayUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(listOfDouble), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("LIST argument"), result.reason());
    }

    @Test
    void anArrayColumnThatMayContainNullsIsRefused() {
        var nullableList = DataTypes.createArrayType(DataTypes.DoubleType, true);
        ResolvedUdf resolved = UdfResolver.resolve(new CosineSimilarityUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(nullableList, nullableList), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("null elements"), result.reason());
    }

    /**
     * The shape Spark actually delivers: an {@code ArrayType} column reaches a Java UDF as a
     * {@code scala.collection.Seq}, not as a {@code double[]}.
     */
    public static final class SeqCosineSimilarityUdf implements UDF2<scala.collection.Seq<Double>, scala.collection.Seq<Double>, Double> {
        @Override
        public Double call(scala.collection.Seq<Double> a, scala.collection.Seq<Double> b) {
            int n = Math.min(a.length(), b.length());
            double dot = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            for (int i = 0; i < n; i++) {
                double x = a.apply(i);
                double y = b.apply(i);
                dot += x * y;
                normA += x * x;
                normB += y * y;
            }
            double denominator = Math.sqrt(normA) * Math.sqrt(normB);
            return denominator == 0.0 ? 0.0 : dot / denominator;
        }
    }

    /** A sequence whose elements the UDF believes are ints, over a double column. */
    public static final class WrongElementTypeUdf implements UDF1<scala.collection.Seq<Integer>, Double> {
        @Override
        public Double call(scala.collection.Seq<Integer> values) {
            double total = 0.0;
            for (int i = 0; i < values.length(); i++) {
                total += values.apply(i);
            }
            return total;
        }
    }

    @Test
    void cosineSimilarityOverScalaSequencesLowersCorrectly() throws Exception {
        SeqCosineSimilarityUdf udf = new SeqCosineSimilarityUdf();
        var listOfDouble = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(listOfDouble, listOfDouble), DataTypes.DoubleType, false, true);
        assertTrue(eligibility.isEligible(), eligibility.reason());

        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        Class<?> kernelClass = new GeneratedKernelClassLoader(getClass().getClassLoader()).define(generated.className(), generated.classFile());
        Method kernel = kernelMethod(kernelClass);

        double[][] left = { { 1.0, 2.0, 3.0 }, { 0.5 }, { 4.0, -1.0, 0.25, 8.0 }, {} };
        double[][] right = { { 4.0, 5.0, 6.0 }, { 2.0 }, { 1.0, 1.0, 1.0, 1.0 }, {} };
        int rows = left.length;
        DoubleArray out = new DoubleArray(rows);
        kernel.invoke(null, offsetsOf(left), flatten(left), offsetsOf(right), flatten(right), out, rows);
        for (int i = 0; i < rows; i++) {
            // Compared against the UDF run as Java, on the same values, through a real Scala Seq.
            assertEquals(udf.call(seqOf(left[i]), seqOf(right[i])).doubleValue(), out.get(i), 0.0, "row " + i);
        }
    }

    @Test
    void aSequenceReadAsTheWrongElementTypeIsRefused() {
        var listOfDouble = DataTypes.createArrayType(DataTypes.DoubleType, false);
        ResolvedUdf resolved = UdfResolver.resolve(new WrongElementTypeUdf()).orElseThrow();
        EligibilityResult result = UdfEligibilityAnalyzer.analyse(resolved, List.of(listOfDouble), DataTypes.DoubleType, false, true);
        assertFalse(result.isEligible());
        assertTrue(result.reason().contains("Double"), result.reason());
    }

    private static scala.collection.Seq<Double> seqOf(double[] values) {
        java.util.List<Double> boxed = new java.util.ArrayList<>();
        for (double value : values) {
            boxed.add(value);
        }
        return scala.jdk.javaapi.CollectionConverters.asScala(boxed).toSeq();
    }

    /** cuDF's offsets layout: {@code numRows + 1} entries, each the start of a row in the child. */
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

    /** cuDF's child layout: every row's elements end to end. */
    private static DoubleArray flatten(double[][] rows) {
        int total = 0;
        for (double[] row : rows) {
            total += row.length;
        }
        DoubleArray values = new DoubleArray(Math.max(total, 1));
        int at = 0;
        for (double[] row : rows) {
            for (double value : row) {
                values.set(at++, value);
            }
        }
        return values;
    }

    private static void assertMatches(UDF2<Double, Double, Double> udf, double[] left, double[] right) throws Exception {
        ResolvedUdf resolved = UdfResolver.resolve(udf).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
        EligibilityResult eligibility = UdfEligibilityAnalyzer.analyse(resolved, List.of(DataTypes.DoubleType, DataTypes.DoubleType), DataTypes.DoubleType, false, true);
        assertTrue(eligibility.isEligible(), eligibility.reason());

        ParallelKernelGenerator.Generated generated = ParallelKernelGenerator.generate(resolved, eligibility.inputs(), eligibility.outputKind());
        Class<?> kernelClass = new GeneratedKernelClassLoader(KernelGenerationTest.class.getClassLoader()).define(generated.className(), generated.classFile());
        Method kernel = kernelMethod(kernelClass);

        int rows = left.length;
        DoubleArray a = new DoubleArray(rows);
        DoubleArray b = new DoubleArray(rows);
        DoubleArray out = new DoubleArray(rows);
        for (int i = 0; i < rows; i++) {
            a.set(i, left[i]);
            b.set(i, right[i]);
        }
        kernel.invoke(null, a, b, out, rows);
        for (int i = 0; i < rows; i++) {
            // Bit-for-bit: the lowering only removes boxing, so the arithmetic is identical.
            assertEquals(udf.call(left[i], right[i]).doubleValue(), out.get(i), 0.0, "row " + i);
        }
    }

    private static Method kernelMethod(Class<?> kernelClass) {
        for (Method method : kernelClass.getDeclaredMethods()) {
            if (method.getName().equals(ParallelKernelGenerator.KERNEL_METHOD)) {
                method.setAccessible(true);
                return method;
            }
        }
        throw new AssertionError("no kernel method on " + kernelClass);
    }
}
