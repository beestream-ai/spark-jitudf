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
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * The A/B salt search: 10M users crossed with 64 candidate salts, 640M assignments.
 *
 * <p>
 * This is a <strong>reconstruction</strong> of the workload described in Appendix B of NVIDIA's
 * November 2026 review, not their implementation. Their sources were never requested. Everything
 * the report specifies is followed: 10M users with two integer pre-period metrics, 64 broadcast
 * salts, {@code bucket = murmur3_32(user ^ salt) mod 1000}, arms split at 500, exact {@code long}
 * sums, and the maximum Welch t-statistic over the two metrics ranking the salts.
 * </p>
 *
 * <p>
 * <strong>Reconstruction assumptions, which are ours and not NVIDIA's.</strong> The report does not
 * give the user-metric generator or the salt values, so both are fixed here and stated: salts are
 * {@code 0..63}, and user {@code u} has metrics derived from a multiplicative hash of {@code u}.
 * Because of that, <em>our top-ranked salts are not comparable to theirs</em> and no claim is made
 * about them. What is comparable is internal: the sums are exact integers, so every engine must
 * return the same ranking, and that is what is asserted.
 * </p>
 *
 * <p>
 * The UDF exists because Spark's own {@code hash()} hard-codes seed 42, so an offline assignment
 * that has to agree with an online service cannot use it. The hash here is bit-exact with Guava's
 * {@code murmur3_32_fixed(0)}, which {@link CudfTransformBenchmark} checks directly.
 * </p>
 */
public final class SaltSearch {

    private static int checks = 0;
    private static int failures = 0;

    public static void main(String[] args) {
        long users = args.length > 0 ? Long.parseLong(args[0]) : 10_000_000L;
        int salts = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        boolean runCpu = args.length <= 2 || Boolean.parseBoolean(args[2]);

        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        SparkSession spark = IntegrationTest.session(1000);
        try {
            spark.udf().register("salt_bucket", new CudfTransformBenchmark.BucketUdf(), DataTypes.IntegerType);
            System.out.println("=== salt search: " + users + " users x " + salts + " salts = " + (users * salts) + " assignments ===");

            List<Ranked> gpu = null;
            long gpuMs = 0;
            long t0 = System.nanoTime();
            gpu = search(spark, users, salts, true);
            gpuMs = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf("  GPU (cudf::transform): %,d ms%n", gpuMs);
            print("GPU", gpu);

            if (runCpu) {
                t0 = System.nanoTime();
                List<Ranked> cpu = search(spark, users, salts, false);
                long cpuMs = (System.nanoTime() - t0) / 1_000_000;
                System.out.printf("  CPU Spark            : %,d ms   (speedup %.2fx)%n", cpuMs, cpuMs / (double) gpuMs);
                print("CPU", cpu);
                // Exact integer sums, so the two must agree exactly. This is the oracle Appendix B
                // itself relies on, and the only one available without NVIDIA's generator.
                check("CPU and GPU rank the salts identically", sameRanking(cpu, gpu), cpu + " vs " + gpu);
            }

            System.out.println();
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

    private record Ranked(int salt, double maxT) {
    }

    /**
     * One pass over every (user, salt) pair.
     *
     * <p>
     * The arm is decided by the UDF; the twelve aggregates per salt are the counts, sums and sums of
     * squares each arm needs for a Welch t-statistic on each of the two metrics. They are collected
     * in one grouped aggregation so the 640M assignments are materialised once.
     * </p>
     */
    private static List<Ranked> search(SparkSession spark, long users, int salts, boolean gpu) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(gpu));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(gpu));

        // Deterministic, documented, and built without `%`, which Spark makes nullable.
        Dataset<Row> userTable = spark.range(0, users, 1, 8).selectExpr(
                "id as user_id",
                "cast(pmod(id * 2654435761, 1000) as double) as m1",
                "cast(pmod(id * 40503, 997) as double) as m2");
        Dataset<Row> saltTable = spark.range(0, salts, 1, 1).selectExpr("id as salt");

        Dataset<Row> assigned = userTable.crossJoin(saltTable)
                .selectExpr("salt", "m1", "m2", "salt_bucket(user_id ^ salt) < 500 as arm_a");

        Dataset<Row> stats = assigned.groupBy("salt").agg(
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

        List<Ranked> ranked = new ArrayList<>();
        for (Row row : stats.collectAsList()) {
            long na = row.getLong(1);
            long nb = row.getLong(2);
            double t1 = welch(na, row.getDouble(3), row.getDouble(5), nb, row.getDouble(4), row.getDouble(6));
            double t2 = welch(na, row.getDouble(7), row.getDouble(9), nb, row.getDouble(8), row.getDouble(10));
            ranked.add(new Ranked((int) row.getLong(0), Math.max(Math.abs(t1), Math.abs(t2))));
        }
        // The best salt is the one whose arms are most alike, so the smallest |t| wins.
        ranked.sort((a, b) -> Double.compare(a.maxT(), b.maxT()));
        return ranked;
    }

    /** Welch's t from counts, sums and sums of squares; 0 when a variance is not defined. */
    private static double welch(long na, double sa, double qa, long nb, double sb, double qb) {
        if (na < 2 || nb < 2) {
            return 0;
        }
        double ma = sa / na;
        double mb = sb / nb;
        double va = (qa - na * ma * ma) / (na - 1);
        double vb = (qb - nb * mb * mb) / (nb - 1);
        double denominator = Math.sqrt(va / na + vb / nb);
        return denominator == 0 ? 0 : (ma - mb) / denominator;
    }

    private static boolean sameRanking(List<Ranked> a, List<Ranked> b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (int i = 0; i < Math.min(3, a.size()); i++) {
            if (a.get(i).salt() != b.get(i).salt()) {
                return false;
            }
        }
        return true;
    }

    private static void print(String label, List<Ranked> ranked) {
        StringBuilder top = new StringBuilder();
        for (int i = 0; i < Math.min(3, ranked.size()); i++) {
            top.append(String.format("  #%d salt=%d |t|=%.4f", i + 1, ranked.get(i).salt(), ranked.get(i).maxT()));
        }
        System.out.println("    " + label + " top 3:" + top);
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + detail + "]"));
    }
}
