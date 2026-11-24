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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * The A/B salt search: N users crossed with K candidate salts, and the salt whose two experiment
 * arms are best balanced wins.
 *
 * <p>
 * This is a <strong>reconstruction</strong> of the workload in Appendix B of NVIDIA's November 2026
 * review, not their implementation; their sources were never requested. Everything the report
 * specifies is followed: users with two integer pre-period metrics, broadcast salts,
 * {@code bucket = murmur3_32_seed0(user ^ salt) mod 1000}, arms split at 500, <em>exact</em>
 * {@code long} sums, the maximum Welch |t| over the two metrics ranking the salts, and the top 3 as
 * the answer.
 * </p>
 *
 * <p>
 * <strong>Reconstruction assumptions, ours and not NVIDIA's.</strong> The report gives neither the
 * user-metric generator nor the salt values, so both are fixed here and written down: salts are
 * {@code 0..K-1}, and user {@code u} has {@code m1 = (u * 2654435761) & 1023} and
 * {@code m2 = (u * 40503) & 511}. <em>Our top-ranked salts are therefore not comparable to
 * theirs</em> and no claim is made about them. What is comparable is internal: the sums are exact
 * integers, so every semantically equivalent engine must return the same aggregates and the same
 * ranking, and that is what is asserted.
 * </p>
 *
 * <h2>The point of this version</h2>
 *
 * <p>
 * NVIDIA's run needed the {@code user ^ salt} outside the UDF, because the Java binding passed one
 * input column. Their own list of what would make the workload production-worthy opens with: "A
 * {@code transform} JNI that takes several inputs. This would let the UDF take
 * {@code (user_id, salt)} directly instead of relying on the SQL {@code ^}." That binding now
 * exists, so the UDF here is an ordinary two-argument Java UDF that does the combination itself,
 * and it is compiled and run unchanged. The single-input form is kept alongside it, as the
 * baseline it was.
 * </p>
 */
public final class SaltSearch {

    private static int checks = 0;
    private static int failures = 0;

    /**
     * The workload as NVIDIA could not write it: both values straight into the UDF.
     *
     * <p>
     * Murmur3_x86_32 of the 64-bit {@code user ^ salt} with seed 0, which is Guava's
     * {@code murmur3_32_fixed(0).hashLong}. {@link CudfTransformBenchmark} asserts that bit
     * equality against Guava directly before any GPU run.
     * </p>
     */
    public static final class BucketUdf2 implements UDF2<Long, Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long userId, Long salt) {
            long v = userId ^ salt;
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

    /**
     * The one-input form NVIDIA had to use, with the XOR done in SQL.
     *
     * <p>
     * The hash is spelled out again rather than shared with {@link BucketUdf2}. It has to be: a
     * helper is copied into the generated class only when it belongs to the UDF's own class, so a
     * static method on the enclosing {@code SaltSearch} would be refused as "outside its own
     * class", and calling the other UDF would be an allocation, which is also refused. Two copies
     * of twenty lines is the honest cost of keeping both variants compilable.
     * </p>
     */
    public static final class BucketUdf1 implements UDF1<Long, Integer> {
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

    /**
     * How a run evaluates the bucket, and under what configuration.
     *
     * <p>
     * Only the first four are semantically the same function; {@link #GPU_BUILTIN_HASH} is Spark's
     * own Murmur3 with its hard-coded seed 42, so it assigns different buckets on purpose. It is
     * here because NVIDIA measured it, and it is excluded from every equality check.
     * </p>
     */
    private enum Variant {
        CPU_UDF2("CPU Spark, two-input Java UDF", false, false, true),
        RAPIDS_FALLBACK_UDF2("RAPIDS, two-input UDF row-based on CPU", true, false, true),
        TC_CUDF_UDF2("TC -> cudf::transform, two-input UDF", true, true, true),
        TC_CUDF_UDF1_XOR("TC -> cudf::transform, one-input UDF + SQL XOR", true, true, true),
        GPU_BUILTIN_HASH("RAPIDS built-in hash (seed 42, DIFFERENT HASH)", true, true, false);

        final String label;
        final boolean rapids;
        final boolean tornado;
        /** Whether this computes the same function as the others, and so must agree with them. */
        final boolean equivalent;

        Variant(String label, boolean rapids, boolean tornado, boolean equivalent) {
            this.label = label;
            this.rapids = rapids;
            this.tornado = tornado;
            this.equivalent = equivalent;
        }

        /** The SQL that produces the 0..999 bucket for this variant. */
        String bucketExpr() {
            return switch (this) {
                case CPU_UDF2, RAPIDS_FALLBACK_UDF2, TC_CUDF_UDF2 -> "salt_bucket2(user_id, salt)";
                case TC_CUDF_UDF1_XOR -> "salt_bucket1(user_id ^ salt)";
                case GPU_BUILTIN_HASH -> "pmod(hash(user_id ^ salt), 1000)";
            };
        }

        /** The UDF name RAPIDS would prefix with gpu_, or null where no UDF is involved. */
        String udfName() {
            return switch (this) {
                case CPU_UDF2, RAPIDS_FALLBACK_UDF2, TC_CUDF_UDF2 -> "salt_bucket2";
                case TC_CUDF_UDF1_XOR -> "salt_bucket1";
                case GPU_BUILTIN_HASH -> null;
            };
        }
    }

    /** One salt's exact aggregates. Every field is an exact integer, so engines must agree. */
    private record Aggregates(long salt, long bucketSum, long nA, long nB,
            long s1a, long s1b, long q1a, long q1b,
            long s2a, long s2b, long q2a, long q2b) {
    }

    private record Ranked(long salt, double maxT) {
    }

    public static void main(String[] args) {
        long users = args.length > 0 ? Long.parseLong(args[0]) : 10_000_000L;
        int salts = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        int runs = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        long smallUsers = args.length > 3 ? Long.parseLong(args[3]) : 200_000L;

        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        SparkSession spark = IntegrationTest.session(1000);
        try {
            spark.udf().register("salt_bucket2", new BucketUdf2(), DataTypes.IntegerType);
            spark.udf().register("salt_bucket1", new BucketUdf1(), DataTypes.IntegerType);

            System.out.println("=== A/B salt search ===");
            System.out.println("  master      : " + spark.sparkContext().master()
                    + "  (defaultParallelism=" + spark.sparkContext().defaultParallelism() + ")");
            System.out.println("  correctness : " + smallUsers + " users x " + salts + " salts");
            System.out.println("  benchmark   : " + users + " users x " + salts + " salts = "
                    + (users * salts) + " assignments, " + runs + " measured runs each");

            // -------- correctness first, at a size that is quick to repeat --------
            System.out.println();
            System.out.println("--- correctness (" + smallUsers + " users) ---");
            Map<Variant, List<Aggregates>> small = new LinkedHashMap<>();
            for (Variant v : Variant.values()) {
                small.put(v, aggregate(spark, smallUsers, salts, v));
                System.out.println("  " + v.label + ": " + small.get(v).size() + " salts aggregated");
            }
            compareEquivalentVariants(small);
            checkExecutionPaths(spark, smallUsers, salts);

            // -------- then the benchmark --------
            System.out.println();
            System.out.println("--- benchmark (" + users + " users x " + salts + " salts) ---");
            Map<Variant, long[]> timings = new LinkedHashMap<>();
            Map<Variant, List<Aggregates>> full = new LinkedHashMap<>();
            for (Variant v : Variant.values()) {
                long[] ms = new long[runs];
                // One unmeasured warm-up: the first run of any variant pays for plan compilation,
                // the UDF's own compilation, and cudf's JIT.
                full.put(v, aggregate(spark, users, salts, v));
                for (int i = 0; i < runs; i++) {
                    long t0 = System.nanoTime();
                    aggregate(spark, users, salts, v);
                    ms[i] = (System.nanoTime() - t0) / 1_000_000;
                }
                timings.put(v, ms);
                report(v, ms);
            }
            compareEquivalentVariants(full);
            System.out.println();
            summarise(timings);

            System.out.println();
            System.out.println("--- top 3 salts (identical for every equivalent variant) ---");
            print(rank(full.get(Variant.TC_CUDF_UDF2)));
            System.out.println("  built-in seed-42 hash, a different function, ranks:");
            print(rank(full.get(Variant.GPU_BUILTIN_HASH)));

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
     * One pass over every (user, salt) pair, returning each salt's exact aggregates.
     *
     * <p>
     * The twelve values per salt are the counts, sums and sums of squares each arm needs for a
     * Welch t on each metric, plus the sum of the buckets themselves. That last one is what makes
     * this a real comparison between variants: two engines can agree on every arm total while
     * disagreeing about which rows went to which bucket, and {@code bucketSum} notices.
     * </p>
     */
    private static List<Aggregates> aggregate(SparkSession spark, long users, int salts, Variant variant) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(variant.rapids));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(variant.tornado));

        List<Aggregates> out = new ArrayList<>();
        for (Row row : stats(spark, users, salts, variant).collectAsList()) {
            out.add(new Aggregates(row.getLong(0), row.getLong(1), row.getLong(2), row.getLong(3),
                    row.getLong(4), row.getLong(5), row.getLong(6), row.getLong(7),
                    row.getLong(8), row.getLong(9), row.getLong(10), row.getLong(11)));
        }
        out.sort((a, b) -> Long.compare(a.salt(), b.salt()));
        return out;
    }

    private static Dataset<Row> stats(SparkSession spark, long users, int salts, Variant variant) {
        // Built from &, * and + only. Spark's % and pmod are nullable, and one nullable column
        // would send the UDF down the null-aware path instead of the one being measured.
        // Partitioned by the session's own parallelism, not a constant. A fixed 8 caps the heavy
        // stage at eight tasks however many threads Spark was given, which would make a sweep over
        // masters measure nothing but the constant.
        int parts = Math.max(2, spark.sparkContext().defaultParallelism());
        Dataset<Row> userTable = spark.range(0, users, 1, parts).selectExpr(
                "id as user_id",
                "(id * 2654435761) & 1023 as m1",
                "(id * 40503) & 511 as m2");
        Dataset<Row> saltTable = spark.range(0, salts, 1, 1).selectExpr("id as salt");

        Dataset<Row> assigned = userTable.crossJoin(saltTable)
                .selectExpr("salt", "m1", "m2", variant.bucketExpr() + " as bucket");

        return assigned.selectExpr("salt", "m1", "m2", "bucket", "bucket < 500 as arm_a").groupBy("salt").agg(
                org.apache.spark.sql.functions.expr("sum(bucket) as bucket_sum"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 1 else 0 end) as n_a"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 0 else 1 end) as n_b"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then m1 else 0 end) as s1a"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 0 else m1 end) as s1b"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then m1*m1 else 0 end) as q1a"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 0 else m1*m1 end) as q1b"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then m2 else 0 end) as s2a"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 0 else m2 end) as s2b"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then m2*m2 else 0 end) as q2a"),
                org.apache.spark.sql.functions.expr("sum(case when arm_a then 0 else m2*m2 end) as q2b"));
    }

    /** Every equivalent variant must produce byte-identical aggregates. */
    private static void compareEquivalentVariants(Map<Variant, List<Aggregates>> results) {
        Variant reference = Variant.CPU_UDF2;
        List<Aggregates> expected = results.get(reference);
        for (Map.Entry<Variant, List<Aggregates>> entry : results.entrySet()) {
            if (entry.getKey() == reference || !entry.getKey().equivalent) {
                continue;
            }
            check(entry.getKey().label + ": aggregates identical to " + reference.label,
                    expected.equals(entry.getValue()), firstDifference(expected, entry.getValue()));
            check(entry.getKey().label + ": same top 3 salts",
                    topThree(rank(expected)).equals(topThree(rank(entry.getValue()))),
                    topThree(rank(expected)) + " vs " + topThree(rank(entry.getValue())));
        }
        // And the different hash really is different, or the comparison above proves nothing.
        List<Aggregates> builtin = results.get(Variant.GPU_BUILTIN_HASH);
        if (builtin != null) {
            check("the built-in seed-42 hash assigns different buckets, as expected",
                    !expected.equals(builtin), "the two hashes agreed, which would be surprising");
        }
    }

    /** That each variant took the execution path it is named for, read from the physical plan. */
    private static void checkExecutionPaths(SparkSession spark, long users, int salts) {
        System.out.println();
        System.out.println("--- execution paths ---");
        for (Variant v : Variant.values()) {
            spark.conf().set("spark.rapids.sql.enabled", String.valueOf(v.rapids));
            spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(v.tornado));
            Dataset<Row> df = stats(spark, users, salts, v);
            df.collect();
            String plan = df.queryExecution().executedPlan().toString();
            boolean bridged = plan.contains("GpuCpuBridge") || plan.contains("GpuRowBasedScalaUDF");
            switch (v) {
                case CPU_UDF2 -> check(v.label + ": no GPU operator at all", !plan.contains("Gpu"), firstLines(plan));
                case RAPIDS_FALLBACK_UDF2 -> check(v.label + ": the UDF really is row-based on the CPU",
                        bridged, "expected a bridge and found none: " + firstLines(plan));
                case TC_CUDF_UDF2, TC_CUDF_UDF1_XOR -> {
                    check(v.label + ": gpu_" + v.udfName() + " in the plan", plan.contains("gpu_" + v.udfName()), firstLines(plan));
                    check(v.label + ": no CPU bridge", !bridged, firstLines(plan));
                }
                case GPU_BUILTIN_HASH -> check(v.label + ": runs on the GPU with no UDF", plan.contains("Gpu") && !bridged, firstLines(plan));
            }
        }
        // The two-input UDF can only have run through the generalized binding: the released one
        // passes a single column.
        String decision = TornadoUdfProvider.decisions().getOrDefault("salt_bucket2", "");
        check("the two-input UDF was compiled through cudf::transform with two inputs",
                decision.contains("cudf-transform") && decision.contains("[LONG, LONG]"), "decision: " + decision);
    }

    private static List<Ranked> rank(List<Aggregates> aggregates) {
        List<Ranked> ranked = new ArrayList<>();
        for (Aggregates a : aggregates) {
            double t1 = welch(a.nA(), a.s1a(), a.q1a(), a.nB(), a.s1b(), a.q1b());
            double t2 = welch(a.nA(), a.s2a(), a.q2a(), a.nB(), a.s2b(), a.q2b());
            ranked.add(new Ranked(a.salt(), Math.max(Math.abs(t1), Math.abs(t2))));
        }
        // The best salt is the one whose arms are most alike, so the smallest |t| wins. Ties break
        // on the salt, so the order is total and two engines cannot disagree by luck.
        ranked.sort((x, y) -> {
            int c = Double.compare(x.maxT(), y.maxT());
            return c != 0 ? c : Long.compare(x.salt(), y.salt());
        });
        return ranked;
    }

    /** Welch's t from exact counts, sums and sums of squares; 0 when a variance is undefined. */
    private static double welch(long na, long sa, long qa, long nb, long sb, long qb) {
        if (na < 2 || nb < 2) {
            return 0;
        }
        double ma = sa / (double) na;
        double mb = sb / (double) nb;
        double va = (qa - na * ma * ma) / (na - 1);
        double vb = (qb - nb * mb * mb) / (nb - 1);
        double denominator = Math.sqrt(va / na + vb / nb);
        return denominator == 0 ? 0 : (ma - mb) / denominator;
    }

    private static List<Long> topThree(List<Ranked> ranked) {
        List<Long> top = new ArrayList<>();
        for (int i = 0; i < Math.min(3, ranked.size()); i++) {
            top.add(ranked.get(i).salt());
        }
        return top;
    }

    private static String firstDifference(List<Aggregates> a, List<Aggregates> b) {
        if (a.size() != b.size()) {
            return "different salt counts: " + a.size() + " vs " + b.size();
        }
        for (int i = 0; i < a.size(); i++) {
            if (!a.get(i).equals(b.get(i))) {
                return "salt " + a.get(i).salt() + ": " + a.get(i) + " vs " + b.get(i);
            }
        }
        return "no difference found";
    }

    private static void report(Variant v, long[] ms) {
        long[] sorted = ms.clone();
        java.util.Arrays.sort(sorted);
        System.out.printf("  %-48s median=%,7d ms   min=%,7d  max=%,7d  (n=%d)%n",
                v.label, sorted[sorted.length / 2], sorted[0], sorted[sorted.length - 1], ms.length);
    }

    private static void summarise(Map<Variant, long[]> timings) {
        long cpu = median(timings.get(Variant.CPU_UDF2));
        long fallback = median(timings.get(Variant.RAPIDS_FALLBACK_UDF2));
        System.out.println("--- relative to the two baselines that matter ---");
        for (Map.Entry<Variant, long[]> e : timings.entrySet()) {
            long m = median(e.getValue());
            System.out.printf("  %-48s %,7d ms   vs CPU %5.2fx   vs RAPIDS fallback %5.2fx%n",
                    e.getKey().label, m, cpu / (double) m, fallback / (double) m);
        }
    }

    private static long median(long[] ms) {
        long[] sorted = ms.clone();
        java.util.Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static void print(List<Ranked> ranked) {
        StringBuilder top = new StringBuilder();
        for (int i = 0; i < Math.min(3, ranked.size()); i++) {
            top.append(String.format("  #%d salt=%d |t|=%.6f", i + 1, ranked.get(i).salt(), ranked.get(i).maxT()));
        }
        System.out.println("   " + top);
    }

    private static String firstLines(String plan) {
        return plan.lines().limit(3).reduce("", (a, b) -> a + "\n      " + b);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }
}
