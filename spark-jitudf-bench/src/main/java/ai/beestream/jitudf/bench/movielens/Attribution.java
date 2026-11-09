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
package ai.beestream.jitudf.bench.movielens;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * Per-run elapsed time, split into parts that do not overlap.
 *
 * <p>
 * The operator counters RAPIDS exposes are accumulated across concurrent tasks and nested inside
 * one another, so adding them up gives task-seconds rather than elapsed seconds and the same
 * nanosecond can appear in several of them. A stacked bar built that way would be wrong however
 * carefully it was drawn. This takes a different route: one wall-clock timer around the query, one
 * timer around every call into the UDF, and the remainder attributed to everything else. Those
 * three quantities are measured at disjoint boundaries, so they add up by construction.
 * </p>
 *
 * <p>
 * Whether the UDF time is <em>critical-path</em> time rather than summed concurrent work depends on
 * the scoring stage running in a single task. That is not assumed here: the task count comes from
 * the Spark event log and is checked when the breakdown is built.
 * </p>
 */
public final class Attribution {

    /** One timed execution and everything measured about it. */
    /**
     * One measured execution.
     *
     * <p>
     * {@code startWallMillis} and {@code endWallMillis} bracket the execution on the same
     * wall clock the Spark event log uses, which is what lets a row be matched to the event-log
     * execution it produced by containment rather than by position. Matching by order is wrong
     * here: the session issues schema and count queries of its own, so the executions in the log
     * outnumber the rows in this file.
     * </p>
     */
    public record Sample(String phase, int runIndex, double elapsedMillis, long rows, Map<String, Long> udf, Map<String, Long> compile, long bridgeNanos, long bridgeBatches,
            long startWallMillis, long endWallMillis, double buildMillis, long classesLoaded, long jitMillis) {
    }

    /**
     * Classes loaded so far by this JVM, and milliseconds it has spent JIT-compiling.
     *
     * <p>
     * Sampled either side of a measured run, these turn "the cold run is start-up" from an
     * assertion into a measurement: start-up means loading classes and compiling them, and both
     * are counted by the JVM itself. They cost a method call to read, so they are sampled outside
     * the timed region and cannot inflate it.
     * </p>
     */
    public static long classesLoaded() {
        return java.lang.management.ManagementFactory.getClassLoadingMXBean().getTotalLoadedClassCount();
    }

    /** Milliseconds this JVM has spent in the JIT compiler, or 0 where unsupported. */
    public static long jitMillis() {
        java.lang.management.CompilationMXBean b = java.lang.management.ManagementFactory.getCompilationMXBean();
        return b != null && b.isCompilationTimeMonitoringSupported() ? b.getTotalCompilationTime() : 0L;
    }

    /** Resets the integration's cumulative counters, if the integration is on the classpath. */
    public static void reset() {
        for (String cls : new String[] { "ai.beestream.jitudf.TornadoUdfMetrics", "ai.beestream.jitudf.CompilePhases" }) {
            try {
                Class.forName(cls).getMethod("reset").invoke(null);
            } catch (ReflectiveOperationException ignored) {
                // No integration in this configuration; nothing to reset.
            }
        }
    }

    /**
     * Driver-side compilation work for this execution.
     *
     * <p>
     * These phases run while the query is planned, which happens inside the same call the elapsed
     * timer wraps, so they are part of the measured time and have to come out of the remainder
     * rather than hide in it. They run on every planning pass, not only the first, which is why
     * they are reset per execution like everything else here.
     * </p>
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Long> compileTotals() {
        try {
            Class<?> m = Class.forName("ai.beestream.jitudf.CompilePhases");
            return (Map<String, Long>) m.getMethod("totals").invoke(null);
        } catch (ReflectiveOperationException e) {
            return Map.of();
        }
    }

    /** Cumulative per-phase nanoseconds since the last reset, empty when the integration is absent. */
    @SuppressWarnings("unchecked")
    public static Map<String, Long> udfTotals() {
        try {
            Class<?> m = Class.forName("ai.beestream.jitudf.TornadoUdfMetrics");
            return (Map<String, Long>) m.getMethod("totals").invoke(null);
        } catch (ReflectiveOperationException e) {
            return Map.of();
        }
    }

    /**
     * The CPU bridge's own time for this execution, read from the executed plan.
     *
     * <p>
     * This is the one number here that comes from a Spark counter rather than a timer of ours, and
     * it is used only for B1, where there is no instrumentation of ours to use instead. It is sound
     * for the same reason the UDF timer is: the scoring stage runs in one task, so the counter's
     * task-sum is that task's elapsed time. Returns the largest bridge counter in the plan, which
     * is the scoring projection; the other projections report zero.
     * </p>
     */
    public static long[] bridgeNanos(Dataset<Row> q) {
        long best = 0;
        long batches = 0;
        for (PlanMetrics.Node n : PlanMetrics.collect(q.queryExecution().executedPlan())) {
            if (n.metric().equals("cpuBridgeProcessingTime") && n.value() > best) {
                best = n.value();
            }
        }
        for (PlanMetrics.Node n : PlanMetrics.collect(q.queryExecution().executedPlan())) {
            if (n.metric().equals("numOutputBatches") && best > 0 && n.value() > batches && n.value() < 1000) {
                batches = n.value();
            }
        }
        return new long[] { best, batches };
    }

    /** Appends one row per execution. Columns are fixed so the plotting script can rely on them. */
    public static void write(Path file, String tag, String mode, int seeds, int pool, String appId, java.util.List<Sample> samples) {
        String[] phases = { "columnAccess", "outputAlloc", "planLookup", "bind", "inputSync", "execute", "outputSync", "wrap", "kernel", "tvmCompile", "driverInstall", "dispatch", "call", "coldCall" };
        String[] comp = { "resolve", "analyse", "generate", "verify", "define_class", "plan_build" };
        try {
            boolean fresh = !Files.exists(file);
            Files.createDirectories(file.getParent());
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
                if (fresh) {
                    StringBuilder h = new StringBuilder("tag,mode,seeds,pool,pairs,phase,run_index,elapsed_ms,rows,calls,cold_calls,bridge_ns,bridge_batches,app_id,start_wall_ms,end_wall_ms,build_ms,classes_loaded,jit_ms");
                    for (String ph : phases) {
                        h.append(',').append(ph).append("_ns");
                    }
                    for (String c : comp) {
                        h.append(',').append(c).append("_ns").append(',').append(c).append("_n");
                    }
                    w.println(h);
                }
                for (Sample s : samples) {
                    Map<String, Long> u = new LinkedHashMap<>(s.udf());
                    StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "%s,%s,%d,%d,%d,%s,%d,%.3f,%d,%d,%d,%d,%d,%s,%d,%d,%.3f,%d,%d", tag, mode, seeds, pool,
                            seeds * (pool - 1), s.phase(), s.runIndex(), s.elapsedMillis(), s.rows(), u.getOrDefault("calls", 0L), u.getOrDefault("coldCalls", 0L), s.bridgeNanos(),
                            s.bridgeBatches(), appId, s.startWallMillis(), s.endWallMillis(), s.buildMillis(), s.classesLoaded(), s.jitMillis()));
                    for (String ph : phases) {
                        line.append(',').append(u.getOrDefault(ph, 0L));
                    }
                    Map<String, Long> c = s.compile();
                    for (String k : comp) {
                        line.append(',').append(c.getOrDefault(k, 0L)).append(',').append(c.getOrDefault(k + "Count", 0L));
                    }
                    w.println(line);
                }
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException(e);
        }
    }

    private Attribution() {
    }
}
