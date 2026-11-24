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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerExecutorAdded;
import org.apache.spark.scheduler.SparkListenerExecutorRemoved;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import ai.beestream.jitudf.TornadoUdfProvider;

/**
 * The cuDF backend with executors in their own JVMs: compiled on the driver, shipped as a string,
 * run by cuDF on executors that have no TornadoVM at all.
 *
 * <p>
 * Run with a {@code local-cluster[N,cores,memMB]} master, which starts a standalone master and
 * worker in the driver process and launches each executor as a separate JVM. Everything a real
 * deployment depends on crosses a process boundary here: the {@code RapidsUDF} is serialized, the
 * executor loads the plugin and the cuDF binding from the shipped jars, and NVRTC compiles the
 * source on the executor. It is not a multi-node cluster, and does not claim to be.
 * </p>
 */
public final class SeparateExecutorTest {

    private static int checks;
    private static int failures;

    private SeparateExecutorTest() {
    }

    public static final class BitCountUdf implements UDF1<Long, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Long x) {
            return Long.bitCount(x);
        }
    }

    /** Declined (floating point with incompatibleOps off): must stay on the CPU. */
    public static final class SinInIntUdf implements UDF1<Integer, Integer> {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Integer x) {
            return (int) (Math.sin((double) x) * 1_000_000_000.0);
        }
    }

    /**
     * Fails the first attempt of partition 0, once: by throwing, or by ending the executor JVM. It
     * calls TaskContext, so the provider declines it and it runs on the CPU; the GPU UDFs in the
     * same stage are re-run in the retried task. The marker file makes "once" hold across
     * executors, since an attempt number alone resets with a replaced executor's view.
     */
    public static final class FailOnceUdf implements UDF1<Long, Long> {
        private static final long serialVersionUID = 1L;
        private final String marker;
        private final boolean exitJvm;

        public FailOnceUdf(String marker, boolean exitJvm) {
            this.marker = marker;
            this.exitJvm = exitJvm;
        }

        @Override
        public Long call(Long x) throws Exception {
            TaskContext task = TaskContext.get();
            if (task != null && task.partitionId() == 0 && x % 1000 == 0) {
                File file = new File(marker);
                if (file.createNewFile()) {
                    if (exitJvm) {
                        Runtime.getRuntime().halt(17);
                    }
                    throw new IllegalStateException("injected failure, attempt " + task.attemptNumber());
                }
            }
            return x;
        }
    }

    public static void main(String[] args) throws Exception {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        String master = IntegrationTest.requestedMaster();
        if (!master.startsWith("local-cluster")) {
            throw new IllegalStateException("run this with BENCH_MASTER=local-cluster[2,1,4096]; got " + master);
        }
        SparkSession spark = SparkSession.builder().appName("separate-executor").master(master)
                .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
                .config("spark.rapids.sql.enabled", "true")
                .config("spark.rapids.sql.explain", "NOT_ON_GPU")
                .config("spark.rapids.memory.gpu.pool", "ASYNC")
                .config("spark.rapids.memory.gpu.allocFraction", "0.2")
                .config("spark.rapids.memory.gpu.minAllocFraction", "0.05")
                .config("spark.rapids.memory.gpu.maxAllocFraction", "0.3")
                .config("spark.rapids.memory.gpu.reserve", "536870912")
                .config("spark.sql.shuffle.partitions", "4")
                .config("spark.task.maxFailures", "4")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        AtomicInteger failedTasks = new AtomicInteger();
        AtomicInteger executorsAdded = new AtomicInteger();
        AtomicInteger executorsRemoved = new AtomicInteger();
        spark.sparkContext().addSparkListener(new SparkListener() {
            @Override
            public void onTaskEnd(SparkListenerTaskEnd end) {
                if (!end.reason().toString().equals("Success")) {
                    failedTasks.incrementAndGet();
                }
            }

            @Override
            public void onExecutorAdded(SparkListenerExecutorAdded added) {
                executorsAdded.incrementAndGet();
            }

            @Override
            public void onExecutorRemoved(SparkListenerExecutorRemoved removed) {
                executorsRemoved.incrementAndGet();
            }
        });
        File markers = Files.createTempDirectory("separate-executor-").toFile();
        try {
            spark.udf().register("bitcount", new BitCountUdf(), DataTypes.IntegerType);
            spark.udf().register("active_days", new CodexDifferentialTest.ActiveDaysListUdf(), DataTypes.IntegerType);
            spark.udf().register("sin_in_int", new SinInIntUdf(), DataTypes.IntegerType);
            spark.udf().register("fail_once", new FailOnceUdf(new File(markers, "thrown").getPath(), false), DataTypes.LongType);
            spark.udf().register("exit_once", new FailOnceUdf(new File(markers, "halted").getPath(), true), DataTypes.LongType);

            waitForExecutors(spark, 2);
            System.out.println("\n=== the executors ===");
            java.util.Set<String> original = executorFacts(spark);

            Dataset<Row> base = spark.range(0, rows, 1, 4).selectExpr("id", "id * 7 + 1 AS seed");
            String[] query = { "bitcount(seed) AS a", "active_days(slice(array(seed, seed * 3, seed * 7), 1, CAST(id & 3 AS INT))) AS b",
                    "sin_in_int(CAST(seed AS INT)) AS c" };

            System.out.println("\n=== one query, two compiled UDFs and one declined ===");
            List<Row> cpu = run(spark, base, query, false);
            List<Row> gpu = run(spark, base, query, true);
            compare("compiled on the driver, run on separate executors", cpu, gpu);
            String plan = plan(spark, base, query);
            check("bitcount runs as gpu_bitcount", plan.contains("gpu_bitcount"), plan);
            check("the LIST fold runs as gpu_active_days", plan.contains("gpu_active_days"), plan);
            check("the declined UDF is not compiled", !plan.contains("gpu_sin_in_int") && plan.contains("sin_in_int"), plan);
            check("the provider declined it for floating point",
                    TornadoUdfProvider.decisions().getOrDefault("sin_in_int", "").contains("floating point"), TornadoUdfProvider.decisions().toString());

            System.out.println("\n=== a task fails once and is retried ===");
            int before = failedTasks.get();
            List<Row> retried = run(spark, base, with(query, "fail_once(id) AS d"), true);
            List<Row> reference = run(spark, base, with(query, "id AS d"), false);
            compare("after a retried task", reference, retried);
            check("a task really failed and was retried", failedTasks.get() > before, "failed tasks: " + failedTasks.get());

            System.out.println("\n=== an executor dies and is replaced ===");
            int added = executorsAdded.get();
            int removed = executorsRemoved.get();
            List<Row> replaced = run(spark, base, with(query, "exit_once(id) AS d"), true);
            compare("after an executor was replaced", reference, replaced);
            check("an executor was lost", executorsRemoved.get() > removed, "removed: " + executorsRemoved.get());
            waitForExecutors(spark, 2);
            check("a replacement executor was launched", executorsAdded.get() > added, "added: " + executorsAdded.get());
            System.out.println("  executors after replacement:");
            // The replacement registers a moment after the loss; wait until it is running tasks.
            java.util.Set<String> now = java.util.Set.of();
            for (int attempt = 0; attempt < 30; attempt++) {
                now = executorFacts(spark, false);
                if (!original.containsAll(now)) {
                    break;
                }
                Thread.sleep(1000);
            }
            java.util.Set<String> fresh = new java.util.HashSet<>(now);
            fresh.removeAll(original);
            check("the replacement executor (" + fresh + ") ran tasks, with no TornadoVM", !fresh.isEmpty(), "executors seen: " + now + ", original " + original);
            // The replacement has to compile the shipped source itself, from nothing.
            compare("a further query on the replacement executors", cpu, run(spark, base, query, true));
        } finally {
            spark.stop();
            for (String name : new String[] { "thrown", "halted" }) {
                Files.deleteIfExists(new File(markers, name).toPath());
            }
            Files.deleteIfExists(markers.toPath());
        }
        System.out.println();
        System.out.println("=== CHECKS: " + checks + ", FAILURES: " + failures + " ===");
        if (failures > 0) {
            System.exit(1);
        }
    }

    /** What each executor JVM can and cannot load, and where from, reported back from the executors themselves. */
    private static java.util.Set<String> executorFacts(SparkSession spark) {
        return executorFacts(spark, true);
    }

    private static java.util.Set<String> executorFacts(SparkSession spark, boolean requireTwo) {
        long driverPid = ProcessHandle.current().pid();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());
        int slices = Math.max(4, spark.sparkContext().defaultParallelism() * 2);
        List<Integer> seeds = new ArrayList<>();
        for (int i = 0; i < slices; i++) {
            seeds.add(i);
        }
        List<String> facts = jsc.parallelize(seeds, slices).mapPartitions((FlatMapFunction<Iterator<Integer>, String>) ignored -> {
            String executor = org.apache.spark.SparkEnv.get().executorId();
            return List.of(executor + "|" + ProcessHandle.current().pid() + "|" + loadable("uk.ac.manchester.tornado.api.TornadoExecutionPlan") + "|"
                    + loadable("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime") + "|" + origin("ai.rapids.cudf.OutputNullability") + "|"
                    + origin("ai.beestream.jitudf.CudfTransformUdf") + "|" + sdkPresent() + "|"
                    + System.getProperty("java.library.path", "").contains("tornado")).iterator();
        }).collect().stream().distinct().toList();
        java.util.Set<String> pids = new java.util.HashSet<>();
        java.util.Set<String> ids = new java.util.HashSet<>();
        boolean tornado = false;
        boolean binding = true;
        boolean plugin = true;
        for (String fact : facts) {
            String[] f = fact.split("\\|");
            System.out.println("  executor " + f[0] + " pid " + f[1] + ": tornado-api " + f[2] + ", tornado-runtime " + f[3] + ", TORNADO_SDK " + f[6]
                    + "\n      OutputNullability from " + f[4] + "\n      CudfTransformUdf from " + f[5]);
            pids.add(f[1]);
            ids.add(f[0]);
            tornado |= f[2].equals("loadable") || f[3].equals("loadable") || !f[6].equals("none") || f[7].equals("true");
            binding &= !f[4].equals("absent");
            plugin &= !f[5].equals("absent");
        }
        if (requireTwo) {
            check("at least two executor JVMs, none of them the driver (pid " + driverPid + ")", pids.size() >= 2 && !pids.contains(String.valueOf(driverPid)), pids.toString());
        }
        check("no executor can load TornadoVM or sees a TornadoVM SDK", !tornado, facts.toString());
        check("every executor loads the generalized cuDF binding", binding, facts.toString());
        check("every executor loads the plugin's executor-side UDF class", plugin, facts.toString());
        return ids;
    }

    /** A TornadoVM SDK this JVM could use: TORNADO_SDK naming a directory that exists, or "none". */
    private static String sdkPresent() {
        String sdk = System.getenv("TORNADO_SDK");
        return sdk != null && new File(sdk, "share/java/tornado").isDirectory() ? sdk : "none";
    }

    private static String loadable(String name) {
        try {
            Class.forName(name, false, Thread.currentThread().getContextClassLoader());
            return "loadable";
        } catch (ClassNotFoundException | LinkageError e) {
            return "absent";
        }
    }

    private static String origin(String name) {
        try {
            Class<?> c = Class.forName(name, false, Thread.currentThread().getContextClassLoader());
            var source = c.getProtectionDomain().getCodeSource();
            return source == null ? "(no code source)" : String.valueOf(source.getLocation());
        } catch (ClassNotFoundException | LinkageError e) {
            return "absent";
        }
    }

    private static void waitForExecutors(SparkSession spark, int count) throws InterruptedException {
        for (int i = 0; i < 120; i++) {
            // getExecutorInfos includes the driver.
            if (spark.sparkContext().statusTracker().getExecutorInfos().length - 1 >= count) {
                return;
            }
            Thread.sleep(500);
        }
        throw new IllegalStateException("fewer than " + count + " executors registered");
    }

    private static String[] with(String[] select, String extra) {
        String[] out = java.util.Arrays.copyOf(select, select.length + 1);
        out[select.length] = extra;
        return out;
    }

    private static String[] withId(String[] select) {
        String[] out = new String[select.length + 1];
        out[0] = "id";
        System.arraycopy(select, 0, out, 1, select.length);
        return out;
    }

    private static List<Row> run(SparkSession spark, Dataset<Row> source, String[] select, boolean enabled) {
        spark.conf().set("spark.rapids.sql.enabled", String.valueOf(enabled));
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", String.valueOf(enabled));
        return source.selectExpr(withId(select)).orderBy("id").collectAsList();
    }

    private static String plan(SparkSession spark, Dataset<Row> source, String[] select) {
        spark.conf().set("spark.rapids.sql.enabled", "true");
        spark.conf().set("spark.rapids.sql.tornadovm.enabled", "true");
        Dataset<Row> df = source.selectExpr(withId(select));
        df.collect();
        return df.queryExecution().executedPlan().toString();
    }

    private static void compare(String label, List<Row> cpu, List<Row> gpu) {
        int mismatches = 0;
        for (int i = 0; i < Math.min(cpu.size(), gpu.size()); i++) {
            if (!cpu.get(i).equals(gpu.get(i)) && mismatches++ < 3) {
                System.out.println("  row " + i + ": cpu=" + cpu.get(i) + " gpu=" + gpu.get(i));
            }
        }
        check(label + ": " + cpu.size() + " rows, every value identical to CPU Spark", cpu.size() == gpu.size() && mismatches == 0, mismatches + " rows differ");
    }

    private static void check(String what, boolean ok, String detail) {
        checks++;
        if (!ok) {
            failures++;
        }
        System.out.println((ok ? "  PASS  " : "  FAIL  ") + what + (ok ? "" : "  [" + (detail.length() > 1500 ? detail.substring(0, 1500) : detail) + "]"));
    }

    @SuppressWarnings("unused")
    private static void unused() throws IOException {
    }
}
