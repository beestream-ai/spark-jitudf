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

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * The reconstructed workloads, measured through cuDF's {@code transform}.
 *
 * <p>
 * Two of these are reconstructions of experiments described in NVIDIA's November 2026 review.
 * <strong>They are not NVIDIA's implementations.</strong> That report references its source files by
 * path and says they are available on request; they were not requested, and nothing here was derived
 * from them. What makes the reconstructions checkable is that each has an oracle established
 * independently:
 * </p>
 *
 * <ul>
 * <li><b>Collatz.</b> The report gives an aggregate, 8,600,532,084, for
 * {@code SELECT sum(collatz(id % 1e8 + 1)) FROM range(50M)}. Computed directly, that is exactly the
 * sum of steps-to-one over 1..50,000,000 with {@code collatz(1) = 0} and 64-bit intermediates. The
 * convention is therefore pinned without needing their code, and the aggregate is asserted here.</li>
 * <li><b>Murmur3 bucketing.</b> The UDF is checked against Guava's {@code murmur3_32_fixed(0)} on
 * the CPU before any GPU run, so "bit-exact" means bit-exact against the published algorithm.</li>
 * <li><b>Activity bitmap.</b> A representative reconstruction of the one pattern NVIDIA's customer
 * survey called a near-fit. There is no published number to match, so only CPU/GPU agreement is
 * asserted.</li>
 * </ul>
 *
 * <p>
 * What is comparable to NVIDIA's figures: the Collatz aggregate, and the ordering of configurations
 * on one machine. What is not: absolute times. Their runs used a Quadro RTX 6000 and Spark 3.5.7
 * with a local 26.10.0-SNAPSHOT plugin; this uses whatever GPU is present, Spark 4.0.4 and RAPIDS
 * 26.08.1. Speedups are reported against this machine's own CPU baseline and against nothing else.
 * </p>
 */
public final class CudfTransformBenchmark {

    private static int checks = 0;
    private static int failures = 0;

    /** Steps to reach one. 64-bit: 3x+1 leaves the int range well inside this input domain. */
    public static final class CollatzUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long seed) {
            long x = seed;
            int steps = 0;
            while (x != 1) {
                x = (x & 1L) == 1L ? 3 * x + 1 : x >>> 1;
                steps++;
            }
            return steps;
        }
    }

    /**
     * Murmur3_32 of a long with seed 0, then a bucket out of 1000.
     *
     * <p>
     * This exists because Spark's own {@code hash()} hard-codes seed 42, so offline assignment that
     * has to agree with an online service's bucketing cannot use it. The body is the published
     * Murmur3_32 algorithm written without any library call, which is what lets it compile: no
     * allocation, no exceptions, just arithmetic and shifts.
     * </p>
     */
    public static final class BucketUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long userXorSalt) {
            long v = userXorSalt;
            int h1 = 0;
            int k1 = (int) v;
            k1 *= 0xcc9e2d51;
            k1 = Integer.rotateLeft(k1, 15);
            k1 *= 0x1b873593;
            h1 ^= k1;
            h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
            k1 = (int) (v >>> 32);
            k1 *= 0xcc9e2d51;
            k1 = Integer.rotateLeft(k1, 15);
            k1 *= 0x1b873593;
            h1 ^= k1;
            h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
            h1 ^= 8;
            h1 ^= h1 >>> 16;
            h1 *= 0x85ebca6b;
            h1 ^= h1 >>> 13;
            h1 *= 0xc2b2ae35;
            h1 ^= h1 >>> 16;
            return Math.floorMod(h1, 1000);
        }
    }

    /** Active days in a packed bitmap: the near-fit pattern from NVIDIA's customer survey. */
    public static final class ActiveDaysUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long bitmap) {
            return Long.bitCount(bitmap);
        }
    }

    /**
     * Two input columns: a weighted blend with a loop, so RAPIDS has to decline it too.
     *
     * <p>
     * Only reachable at all because the binding now takes more than one column. The old one passed
     * exactly one, so this UDF was refused at planning time and never measured.
     * </p>
     */
    public static final class BlendUdf implements UDF2<Long, Integer, Long> {
        private static final long serialVersionUID = 1L;

        @Override
        public Long call(Long a, Integer b) {
            long acc = a;
            for (int i = 0; i < 16; i++) {
                acc = acc * 31 + b + i;
                acc ^= acc >>> 29;
            }
            return acc;
        }
    }

    /**
     * A nullable input with a default: the UDF answers for a null rather than propagating it.
     *
     * <p>
     * This is the shape that makes null-awareness necessary rather than convenient. Copying the
     * input's validity mask onto the output would make every null row null, and this UDF says
     * otherwise for every one of them.
     * </p>
     */
    public static final class NullDefaultingUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long seed) {
            if (seed == null) {
                return -1;
            }
            long x = seed;
            int steps = 0;
            while (x != 1) {
                x = (x & 1L) == 1L ? 3 * x + 1 : x >>> 1;
                steps++;
            }
            return steps;
        }
    }

    public static void main(String[] args) {
        long collatzRows = args.length > 0 ? Long.parseLong(args[0]) : 50_000_000L;
        long otherRows = args.length > 1 ? Long.parseLong(args[1]) : 20_000_000L;
        int warm = args.length > 2 ? Integer.parseInt(args[2]) : 3;

        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        SparkSession spark = IntegrationTest.session(1000);
        try {
            spark.udf().register("b_collatz", new CollatzUdf(), DataTypes.IntegerType);
            spark.udf().register("b_bucket", new BucketUdf(), DataTypes.IntegerType);
            spark.udf().register("b_active", new ActiveDaysUdf(), DataTypes.IntegerType);
            spark.udf().register("b_blend", new BlendUdf(), DataTypes.LongType);
            spark.udf().register("b_null_collatz", new NullDefaultingUdf(), DataTypes.IntegerType);

            collatz(spark, collatzRows, warm);
            bucketing(spark, otherRows, warm);
            bitmap(spark, otherRows, warm);
            multiInput(spark, otherRows, warm);
            nullable(spark, otherRows, warm);

            System.out.println();
            System.out.println("--- planning decisions ---");
            TornadoUdfProvider.decisions().forEach((n, d) -> System.out.println("  " + n + " -> " + d));

            // The driver-side cost, which the query times above do not separate out: resolving the
            // UDF, analysing it, generating the entry class and compiling it to CUDA. It is paid
            // once per distinct UDF per driver, not per batch and not per row, and the cache below
            // it means a second query using the same UDF pays none of it. NVRTC's own compilation
            // happens on the executor inside cudf and is not counted here.
            System.out.println();
            System.out.println("--- driver-side compilation (once per UDF, not per batch) ---");
            System.out.println("  " + ai.beestream.jitudf.CompilePhases.summary());
        } finally {
            spark.stop();
        }

        System.out.println();
        System.out.println("=== CHECKS: " + checks + ", FAILURES: " + failures + " ===");
        if (failures > 0) {
            System.exit(1);
        }
    }

    /**
     * {@code SELECT sum(collatz(id + 1)) FROM range(rows)}.
     *
     * <p>
     * NVIDIA's query is {@code id % 1e8 + 1} over {@code range(50M)}. Since every id is below 1e8
     * the remainder is the identity, so {@code id + 1} is the same function over the same inputs
     * and avoids a remainder, which Spark makes nullable and the analysis therefore refuses.
     * </p>
     */
    private static void collatz(SparkSession spark, long rows, int warm) {
        System.out.println();
        System.out.println("=== W-collatz: sum(collatz(id+1)) over " + rows + " rows ===");
        String query = "select sum(b_collatz(id + 1)) as total from range(0, " + rows + ", 1, 2)";

        long cpu = time(spark, query, false, warm);
        long gpu = time(spark, query, true, warm);
        long total = scalar(spark, query, true);

        if (rows == 50_000_000L) {
            // The oracle, computed independently of NVIDIA's code: sum of steps-to-one over
            // 1..50,000,000 with collatz(1)=0 and 64-bit intermediates.
            check("collatz aggregate equals the published 8,600,532,084", total == 8_600_532_084L, "got " + total);
        }
        check("collatz: CPU and GPU agree", scalar(spark, query, false) == total, "cpu != gpu");
        report("W-collatz", rows, cpu, gpu);
    }

    private static void bucketing(SparkSession spark, long rows, int warm) {
        System.out.println();
        System.out.println("=== W-bucket: custom-seed Murmur3 assignment over " + rows + " rows ===");

        // Bit-equality with the published algorithm, on the CPU, before any GPU run.
        int mismatches = 0;
        BucketUdf udf = new BucketUdf();
        for (long u = 0; u < 200_000; u++) {
            long salt = (u * 2654435761L) & 0xffff;
            if (udf.call(u ^ salt) != guavaBucket(u ^ salt)) {
                mismatches++;
            }
        }
        check("bucket UDF matches Guava murmur3_32(0) bit for bit", mismatches == 0, mismatches + " mismatches");

        // One salt's worth of assignment; the 64-salt cross join of the original is not run here.
        String query = "select sum(cast(b_bucket(id ^ 12345) < 500 as int)) as arm_a from range(0, " + rows + ", 1, 2)";
        long cpu = time(spark, query, false, warm);
        long gpu = time(spark, query, true, warm);
        check("bucket: CPU and GPU agree", scalar(spark, query, false) == scalar(spark, query, true), "cpu != gpu");
        report("W-bucket", rows, cpu, gpu);
    }

    private static void bitmap(SparkSession spark, long rows, int warm) {
        System.out.println();
        System.out.println("=== W-bitmap: active days in a packed bitmap over " + rows + " rows ===");
        String query = "select sum(b_active(id * 2654435761)) as active from range(0, " + rows + ", 1, 2)";
        long cpu = time(spark, query, false, warm);
        long gpu = time(spark, query, true, warm);
        check("bitmap: CPU and GPU agree", scalar(spark, query, false) == scalar(spark, query, true), "cpu != gpu");
        report("W-bitmap", rows, cpu, gpu);
    }

    /**
     * Two input columns, which the old binding could not pass at all.
     *
     * <p>
     * Measured against the RAPIDS row-based fallback as well as CPU Spark, because the fallback is
     * what this UDF got before: RAPIDS accepted the query, put the scan and the aggregate on the
     * GPU, and moved every batch back to the host to call the UDF row by row.
     * </p>
     */
    private static void multiInput(SparkSession spark, long rows, int warm) {
        System.out.println();
        System.out.println("=== W-blend: two input columns, 16-round mix over " + rows + " rows ===");
        // The UDF mixes 64 bits, so summing its output directly overflows Spark's own `sum` under
        // ANSI mode -- an overflow in the aggregate, not in the UDF. Masking to ten bits first
        // bounds the total without changing what is being measured, and does so identically in
        // all three configurations.
        String query = "select sum(b_blend(id * 7 + 1, cast(id & 1023 as int)) & 1023) as total "
                + "from range(0, " + rows + ", 1, 2)";
        check("blend: CPU and GPU agree", scalar(spark, query, false) == scalar(spark, query, true), "cpu != gpu");
        report3("W-blend", "b_blend", spark, query, rows, warm);
    }

    /**
     * A nullable column, run through TC's null-aware mode.
     *
     * <p>
     * One row in eight is null, and the UDF answers -1 for those rather than propagating. The
     * aggregate therefore depends on the null handling being right, not merely on the arithmetic:
     * a mask copy would give a different total, and so would skipping the null rows.
     * </p>
     */
    private static void nullable(SparkSession spark, long rows, int warm) {
        System.out.println();
        System.out.println("=== W-null-collatz: one row in eight null, over " + rows + " rows ===");
        String query = "select sum(b_null_collatz("
                + "case when (id & 7) = 0 then null else id + 1 end)) as total "
                + "from range(0, " + rows + ", 1, 2)";
        check("null collatz: CPU and GPU agree, including the null rows",
                scalar(spark, query, false) == scalar(spark, query, true), "cpu != gpu");
        report3("W-null-collatz", "b_null_collatz", spark, query, rows, warm);
    }

    /** Guava's murmur3_32 seed 0 over a long, reached reflectively so Guava is not a build dependency. */
    private static int guavaBucket(long v) {
        try {
            Class<?> hashing = Class.forName("com.google.common.hash.Hashing");
            Object fn = hashing.getMethod("murmur3_32_fixed", int.class).invoke(null, 0);
            // Through the interfaces: the implementation classes are package-private, so asking the
            // concrete class for the method gives one that cannot be invoked from here.
            Class<?> hashFunction = Class.forName("com.google.common.hash.HashFunction");
            Object code = hashFunction.getMethod("hashLong", long.class).invoke(fn, v);
            Class<?> hashCode = Class.forName("com.google.common.hash.HashCode");
            int h = (int) hashCode.getMethod("asInt").invoke(code);
            return Math.floorMod(h, 1000);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Guava is not on the classpath, so the hash cannot be checked", e);
        }
    }

    /**
     * The three configurations compared, which differ only in what evaluates the UDF.
     *
     * <ul>
     * <li>{@link #CPU}: no RAPIDS at all. The whole query is Spark's own.</li>
     * <li>{@link #RAPIDS_FALLBACK}: RAPIDS on, this integration off. The scan, the projection and
     * the aggregate run on the GPU and the UDF does not: RAPIDS has no compiler for a loop, so it
     * evaluates the UDF row by row on the CPU and moves the batch across the boundary twice per
     * projection. This is the baseline that matters, because it is what a user of RAPIDS gets
     * today for a UDF like these.</li>
     * <li>{@link #TC_CUDF}: this integration. The UDF is compiled to a CUDA device function on the
     * driver and run by cudf::transform on the column.</li>
     * </ul>
     */
    private enum Mode {
        CPU("CPU Spark"),
        RAPIDS_FALLBACK("RAPIDS, UDF on CPU"),
        TC_CUDF("TC -> cudf::transform");

        final String label;

        Mode(String label) {
            this.label = label;
        }
    }

    private static long time(SparkSession spark, String query, boolean gpu, int warm) {
        return time(spark, query, gpu ? Mode.TC_CUDF : Mode.CPU, warm);
    }

    private static long time(SparkSession spark, String query, Mode mode, int warm) {
        configure(spark, mode);
        spark.sql(query).collect();                      // warm the JIT, the plan and cudf's kernel cache
        List<Long> times = new ArrayList<>();
        for (int i = 0; i < warm; i++) {
            long t0 = System.nanoTime();
            spark.sql(query).collect();
            times.add((System.nanoTime() - t0) / 1_000_000);
        }
        times.sort(Long::compare);
        return times.get(times.size() / 2);
    }

    private static long scalar(SparkSession spark, String query, boolean gpu) {
        configure(spark, gpu ? Mode.TC_CUDF : Mode.CPU);
        Row row = spark.sql(query).collectAsList().get(0);
        return row.getLong(0);
    }

    private static void configure(SparkSession spark, Mode mode) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(mode != Mode.CPU));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(mode == Mode.TC_CUDF));
    }

    /** Whether the UDF really ran where the mode says it did, read out of the physical plan. */
    private static String where(SparkSession spark, String query, String udf, Mode mode) {
        configure(spark, mode);
        Dataset<Row> df = spark.sql(query);
        df.collect();
        String plan = df.queryExecution().executedPlan().toString();
        if (mode == Mode.CPU) {
            return plan.contains("Gpu") ? "UNEXPECTED: a GPU operator in the CPU configuration" : "all CPU";
        }
        boolean onGpu = plan.contains("gpu_" + udf);
        boolean bridged = plan.contains("GpuCpuBridge") || plan.contains("GpuRowBasedScalaUDF");
        if (mode == Mode.TC_CUDF) {
            return onGpu && !bridged ? "UDF on GPU" : "UNEXPECTED: onGpu=" + onGpu + " bridged=" + bridged;
        }
        return bridged ? "UDF row-based on CPU inside a GPU plan" : "UNEXPECTED: no row-based fallback found";
    }

    private static void report(String label, long rows, long cpuMs, long gpuMs) {
        System.out.printf("  %-12s rows=%,d  CPU=%,d ms  GPU(cudf::transform)=%,d ms  speedup=%.2fx%n",
                label, rows, cpuMs, gpuMs, cpuMs / (double) gpuMs);
    }

    /** One workload across all three configurations, with where each one actually ran. */
    private static void report3(String label, String udf, SparkSession spark, String query, long rows, int warm) {
        long cpu = time(spark, query, Mode.CPU, warm);
        long fallback = time(spark, query, Mode.RAPIDS_FALLBACK, warm);
        long ours = time(spark, query, Mode.TC_CUDF, warm);
        System.out.printf("  %-18s rows=%,d%n", label, rows);
        System.out.printf("      %-22s %6d ms            (%s)%n", Mode.CPU.label, cpu, where(spark, query, udf, Mode.CPU));
        System.out.printf("      %-22s %6d ms  %5.2fx   (%s)%n", Mode.RAPIDS_FALLBACK.label, fallback,
                cpu / (double) fallback, where(spark, query, udf, Mode.RAPIDS_FALLBACK));
        System.out.printf("      %-22s %6d ms  %5.2fx   (%s)%n", Mode.TC_CUDF.label, ours,
                cpu / (double) ours, where(spark, query, udf, Mode.TC_CUDF));
        System.out.printf("      against the RAPIDS fallback: %.2fx%n", fallback / (double) ours);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }
}
