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
import java.util.Random;
import java.util.TreeMap;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.OutputNullability;

import uk.ac.manchester.tornado.tc.frontend.Frontend;
import uk.ac.manchester.tornado.tc.ir.target.TargetCaps;
import uk.ac.manchester.tornado.tc.runtime.TcCompiler;
import uk.ac.manchester.tornado.tc.runtime.TcOptions;

/**
 * How far the GPU's floating point is from the JVM's, for a {@code Math}-heavy UDF.
 *
 * <p>
 * A <strong>reimplementation</strong> of the fidelity check in Appendix A of NVIDIA's November 2026
 * review. Their {@code tvm/FpUdf.java} and {@code cudf/FpCheck.java} were never requested, so the
 * expression here is reconstructed from the report's shorthand,
 * "{@code log1p·sqrt/(1+exp(-x)) + sin·pow(x, 0.3)}", and the inputs from "1 million input values
 * in [0, 1000)". Their numbers -- 97.38% bit-identical, at most 3 ULP -- are therefore a reference
 * point, not a target: a different expression would give different percentages. What is comparable
 * is the <em>shape</em> of the answer and the reasons for it.
 * </p>
 *
 * <p>
 * The report attributes its differences to two causes: CUDA's {@code sin}, {@code pow} and
 * {@code exp} rather than the JDK's, and {@code a*b + c} contracted into an {@code fma}, which Java
 * forbids. Only the second is gone on this path -- {@code TcOptions.withDeviceFunction} compiles
 * strict, so nothing contracts. The first remains, which is why this probe also measures each
 * function on its own: it separates "the device library rounds differently" from "the expression
 * was reassociated".
 * </p>
 *
 * <p>
 * Run directly against cuDF, as NVIDIA's {@code FpCheck} was, with no Spark in the way. The output
 * column is FLOAT64 through the generalized binding -- the released one produces INT32 only, which
 * is why the original spike had to send doubles out as hi/lo INT32 halves.
 * </p>
 */
public final class FpFidelityProbe {

    private FpFidelityProbe() {
    }

    // ---- the UDF under test, and the isolated diagnostics -------------------------------------

    /** The reconstructed expression: log1p(x)*sqrt(x)/(1+exp(-x)) + sin(x)*pow(x, 0.3). */
    public static double fpUdf(double x) {
        return Math.log1p(x) * Math.sqrt(x) / (1.0 + Math.exp(-x)) + Math.sin(x) * Math.pow(x, 0.3);
    }

    public static double onlySin(double x) {
        return Math.sin(x);
    }

    public static double onlyExp(double x) {
        return Math.exp(-x);
    }

    public static double onlyPow(double x) {
        return Math.pow(x, 0.3);
    }

    public static double onlySqrt(double x) {
        return Math.sqrt(x);
    }

    public static double onlyLog1p(double x) {
        return Math.log1p(x);
    }

    /** Only arithmetic: the case where a contracted multiply-add would be the whole difference. */
    public static double onlyArithmetic(double x) {
        return x * x + x;
    }

    // ---- machinery ----------------------------------------------------------------------------

    private record Comparison(String name, long rows, long identical, long maxUlp,
            TreeMap<Long, Long> ulpHistogram, List<String> notes) {

        double identicalPercent() {
            return rows == 0 ? 0 : 100.0 * identical / rows;
        }
    }

    public static void main(String[] args) throws Exception {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 1_000_000;
        long seed = args.length > 1 ? Long.parseLong(args[1]) : 20261122L;

        String jar = new java.io.File(FpFidelityProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getPath();
        var source = Frontend.BytecodeSource.classpath(jar);
        String owner = FpFidelityProbe.class.getName().replace('.', '/');

        System.out.println("=== floating-point fidelity (reimplementation of NVIDIA Appendix A) ===");
        System.out.println("  rows=" + rows + "  seed=" + seed + "  inputs uniform in [0, 1000)");
        System.out.println("  expression: log1p(x)*sqrt(x)/(1+exp(-x)) + sin(x)*pow(x, 0.3)");
        System.out.println();

        // Does the compile-time target change the generated code at all? The plugin always asks
        // for SM 89; this machine may be SM 80. If the two sources are identical then the target
        // is a capability floor, as its javadoc claims, and not a code-generation choice.
        var sm89 = TcCompiler.compileDeviceFunction(source, owner, "fpUdf", "(D)D", TargetCaps.CUDA_SM89, TcOptions.DEFAULTS);
        var sm80 = TcCompiler.compileDeviceFunction(source, owner, "fpUdf", "(D)D",
                new TargetCaps(TargetCaps.Dialect.CUDA, 80, true, true, 16, 1024, 48 * 1024, 32, true, 0), TcOptions.DEFAULTS);
        boolean sameSource = sm89.succeeded() && sm80.succeeded() && sm89.source().equals(sm80.source());
        System.out.println("--- compile target ---");
        System.out.println("  TargetCaps.CUDA_SM89 and an SM 80 target produce "
                + (sameSource ? "byte-identical CUDA" : "DIFFERENT CUDA -- the target is not just a capability floor"));
        if (!sameSource && sm80.succeeded() && sm89.succeeded()) {
            System.out.println("  sm89 length=" + sm89.source().length() + " sm80 length=" + sm80.source().length());
        }
        System.out.println();

        double[] xs = new double[rows];
        Random rng = new Random(seed);
        for (int i = 0; i < rows; i++) {
            xs[i] = rng.nextDouble() * 1000.0;
        }

        List<Comparison> comparisons = new ArrayList<>();
        comparisons.add(compare(source, owner, "fpUdf (whole expression)", "fpUdf", xs, FpFidelityProbe::fpUdf));
        comparisons.add(compare(source, owner, "sin(x)", "onlySin", xs, FpFidelityProbe::onlySin));
        comparisons.add(compare(source, owner, "exp(-x)", "onlyExp", xs, FpFidelityProbe::onlyExp));
        comparisons.add(compare(source, owner, "pow(x, 0.3)", "onlyPow", xs, FpFidelityProbe::onlyPow));
        comparisons.add(compare(source, owner, "sqrt(x)", "onlySqrt", xs, FpFidelityProbe::onlySqrt));
        comparisons.add(compare(source, owner, "log1p(x)", "onlyLog1p", xs, FpFidelityProbe::onlyLog1p));
        comparisons.add(compare(source, owner, "x*x + x (arithmetic only)", "onlyArithmetic", xs, FpFidelityProbe::onlyArithmetic));

        System.out.println("--- results ---");
        System.out.printf("  %-28s %12s %10s %9s   %s%n", "function", "rows", "identical", "max ULP", "ULP histogram");
        for (Comparison c : comparisons) {
            System.out.printf("  %-28s %,12d %9.4f%% %9d   %s%n",
                    c.name(), c.rows(), c.identicalPercent(), c.maxUlp(), histogram(c.ulpHistogram()));
        }

        // Special values, which a percentage over random inputs would never reach.
        System.out.println();
        System.out.println("--- special values (sqrt, the one function defined across all of them) ---");
        double[] special = { 0.0, -0.0, 1.0, Double.MIN_VALUE, Double.MAX_VALUE,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN, -1.0 };
        Comparison sp = compare(source, owner, "sqrt(special)", "onlySqrt", special, FpFidelityProbe::onlySqrt);
        for (String note : sp.notes()) {
            System.out.println("  " + note);
        }

        System.out.println();
        System.out.println("--- how this relates to NVIDIA's 97.38% / 3 ULP ---");
        Comparison whole = comparisons.get(0);
        Comparison arithmetic = comparisons.get(comparisons.size() - 1);
        System.out.printf("  whole expression here : %.4f%% identical, max %d ULP%n", whole.identicalPercent(), whole.maxUlp());
        System.out.printf("  arithmetic only       : %.4f%% identical, max %d ULP%n", arithmetic.identicalPercent(), arithmetic.maxUlp());
        System.out.println("  NVIDIA reported 97.38% identical, max 3 ULP, from CUDA sin/pow/exp AND fma contraction.");
        System.out.println("  A 100% arithmetic-only row is evidence that contraction is off on this path;");
        System.out.println("  what remains is the device math library, which is what the per-function rows show.");
        System.out.println();
        System.out.println("=== FP PROBE DONE ===");
    }

    private interface Fn {
        double apply(double x);
    }

    private static Comparison compare(Frontend.BytecodeSource source, String owner, String label,
            String method, double[] xs, Fn reference) {
        var c = TcCompiler.compileDeviceFunction(source, owner, method, "(D)D", TargetCaps.CUDA_SM89, TcOptions.DEFAULTS);
        List<String> notes = new ArrayList<>();
        if (!c.succeeded()) {
            notes.add("TC refused " + method + ": " + c.diagnostics());
            return new Comparison(label, 0, 0, 0, new TreeMap<>(), notes);
        }
        String cuda = c.source().replaceAll(
                "(?<![A-Za-z0-9_])" + java.util.regex.Pattern.quote(c.entryName()) + "(?![A-Za-z0-9_])",
                "GENERIC_TRANSFORM_OP");

        long identical = 0;
        long maxUlp = 0;
        TreeMap<Long, Long> histogram = new TreeMap<>();
        try (ColumnVector in = ColumnVector.fromDoubles(xs);
                ColumnVector out = ColumnView.transform(new ColumnView[] { in }, cuda, false,
                        DType.FLOAT64, false, OutputNullability.PRESERVE);
                var host = out.copyToHost()) {
            for (int i = 0; i < xs.length; i++) {
                double want = reference.apply(xs[i]);
                double got = host.getDouble(i);
                long wantBits = Double.doubleToRawLongBits(want);
                long gotBits = Double.doubleToRawLongBits(got);
                if (wantBits == gotBits) {
                    identical++;
                    histogram.merge(0L, 1L, Long::sum);
                    continue;
                }
                if (Double.isNaN(want) && Double.isNaN(got)) {
                    // Both NaN but different payloads: the value is the same, the bits are not.
                    identical++;
                    histogram.merge(0L, 1L, Long::sum);
                    notes.add("x=" + xs[i] + ": both NaN with different bit patterns (java="
                            + Long.toHexString(wantBits) + " gpu=" + Long.toHexString(gotBits) + ")");
                    continue;
                }
                long ulp = ulpDistance(want, got);
                maxUlp = Math.max(maxUlp, ulp);
                histogram.merge(Math.min(ulp, 8L), 1L, Long::sum);
                if (notes.size() < 6) {
                    notes.add("x=" + xs[i] + ": java=" + want + " gpu=" + got + " (" + ulp + " ULP)");
                }
            }
        } catch (Exception e) {
            notes.add("transform threw: " + e);
            return new Comparison(label, 0, 0, 0, histogram, notes);
        }
        // For the special-value run the notes are the interesting output, so record every row.
        if (xs.length <= 32) {
            notes.clear();
            try (ColumnVector in = ColumnVector.fromDoubles(xs);
                    ColumnVector out = ColumnView.transform(new ColumnView[] { in }, cuda, false,
                            DType.FLOAT64, false, OutputNullability.PRESERVE);
                    var host = out.copyToHost()) {
                for (int i = 0; i < xs.length; i++) {
                    double want = reference.apply(xs[i]);
                    double got = host.getDouble(i);
                    boolean same = Double.doubleToRawLongBits(want) == Double.doubleToRawLongBits(got);
                    notes.add((same ? "same  " : "DIFF  ") + "x=" + xs[i] + "  java=" + want + "  gpu=" + got
                            + "  (bits java=" + Long.toHexString(Double.doubleToRawLongBits(want))
                            + " gpu=" + Long.toHexString(Double.doubleToRawLongBits(got)) + ")");
                }
            } catch (Exception e) {
                notes.add("transform threw: " + e);
            }
        }
        return new Comparison(label, xs.length, identical, maxUlp, histogram, notes);
    }

    /**
     * How many representable doubles lie between {@code a} and {@code b}.
     *
     * <p>
     * The bit patterns are mapped to a monotone integer ordering first, so that the distance is
     * meaningful across zero and does not depend on sign-magnitude being discontinuous there.
     * </p>
     */
    private static long ulpDistance(double a, double b) {
        if (Double.isNaN(a) || Double.isNaN(b)) {
            return Long.MAX_VALUE;
        }
        long x = ordinal(a);
        long y = ordinal(b);
        long d = x > y ? x - y : y - x;
        return d < 0 ? Long.MAX_VALUE : d;
    }

    private static long ordinal(double d) {
        long bits = Double.doubleToRawLongBits(d);
        return bits >= 0 ? bits : Long.MIN_VALUE - bits;
    }

    private static String histogram(TreeMap<Long, Long> h) {
        if (h.isEmpty()) {
            return "(none)";
        }
        StringBuilder sb = new StringBuilder();
        for (var e : h.entrySet()) {
            sb.append(e.getKey() == 8 ? ">=8" : String.valueOf(e.getKey())).append("ulp=").append(e.getValue()).append(' ');
        }
        return sb.toString().trim();
    }
}
