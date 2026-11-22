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

            collatz(spark, collatzRows, warm);
            bucketing(spark, otherRows, warm);
            bitmap(spark, otherRows, warm);

            System.out.println();
            System.out.println("--- planning decisions ---");
            TornadoUdfProvider.decisions().forEach((n, d) -> System.out.println("  " + n + " -> " + d));
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

    private static long time(SparkSession spark, String query, boolean gpu, int warm) {
        configure(spark, gpu);
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
        configure(spark, gpu);
        Row row = spark.sql(query).collectAsList().get(0);
        return row.getLong(0);
    }

    private static void configure(SparkSession spark, boolean gpu) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(gpu));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(gpu));
    }

    private static void report(String label, long rows, long cpuMs, long gpuMs) {
        System.out.printf("  %-12s rows=%,d  CPU=%,d ms  GPU(cudf::transform)=%,d ms  speedup=%.2fx%n",
                label, rows, cpuMs, gpuMs, cpuMs / (double) gpuMs);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }
}
