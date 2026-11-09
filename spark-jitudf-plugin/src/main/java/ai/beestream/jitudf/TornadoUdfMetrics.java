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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-batch timings collected in the executor, so that the report's breakdown is measured rather
 * than estimated.
 *
 * <p>
 * Nothing here is sampled or extrapolated: every batch that runs through
 * {@link TornadoRapidsUdf#evaluateColumnar} adds its own figures, and the GPU work is bracketed by
 * the synchronisations that are part of the measurement rather than outside it -- timing an
 * asynchronous launch without waiting for it would measure the launch, not the kernel.
 * </p>
 */
public final class TornadoUdfMetrics {

    /**
     * How many per-batch records are kept. An executor that runs for days would otherwise hold one
     * object per batch for the life of the JVM, which is a leak dressed up as telemetry. The
     * counters below stay exact past the cap; only the per-batch detail stops accumulating, and
     * {@link #samplesDropped()} says how much of it was never kept, so no figure derived from these
     * samples can silently claim to cover batches it did not see.
     */
    private static volatile int sampleLimit = Integer.getInteger("tornado.spark.metrics.limit", 100_000);

    /** Lets a test reach the cap without recording a hundred thousand batches to get there. */
    static void sampleLimitForTesting(int limit) {
        sampleLimit = limit;
    }

    private static final List<TornadoRapidsUdf.Timings> SAMPLES = Collections.synchronizedList(new ArrayList<>());
    private static final AtomicLong BATCHES = new AtomicLong();
    private static final AtomicLong ROWS = new AtomicLong();
    private static final AtomicLong DROPPED = new AtomicLong();

    // Cumulative nanoseconds per phase, for end-to-end attribution.
    //
    // The medians above answer "what does a batch cost"; a breakdown that has to add up to elapsed
    // time needs totals, and totals taken from the sample list would silently exclude anything
    // dropped past the cap. These accumulate on every batch, sampled or not. The eight phases are
    // contiguous between t0 and t8, so they sum to exactly the time between those points; CALL is
    // measured around the whole method, so CALL minus that sum is the part of the call the phase
    // timers do not cover -- argument checks before, unbind and close after.
    private static final AtomicLong T_COLUMN = new AtomicLong();
    private static final AtomicLong T_ALLOC = new AtomicLong();
    private static final AtomicLong T_PLAN = new AtomicLong();
    private static final AtomicLong T_BIND = new AtomicLong();
    private static final AtomicLong T_INSYNC = new AtomicLong();
    private static final AtomicLong T_EXEC = new AtomicLong();
    private static final AtomicLong T_OUTSYNC = new AtomicLong();
    private static final AtomicLong T_WRAP = new AtomicLong();
    private static final AtomicLong T_KERNEL = new AtomicLong();
    // Inside execute(): TC's bytecode-to-PTX lowering, the CUDA driver's PTX-to-cubin assembly
    // and module load, and the launch dispatch. Nested inside T_EXEC, so the breakdown subtracts
    // them from it rather than adding them beside it.
    private static final AtomicLong T_TVM_COMPILE = new AtomicLong();
    private static final AtomicLong T_DRIVER_INSTALL = new AtomicLong();
    private static final AtomicLong T_DISPATCH = new AtomicLong();
    private static final AtomicLong T_CALL = new AtomicLong();
    private static final AtomicLong CALLS = new AtomicLong();
    // The same, restricted to batches that built a plan -- the cold ones.
    private static final AtomicLong T_COLD_CALL = new AtomicLong();
    private static final AtomicLong COLD_CALLS = new AtomicLong();

    /** Nanoseconds the caller spent inside {@code evaluateColumnar}, including guard and cleanup. */
    static void recordCall(long nanos) {
        T_CALL.addAndGet(nanos);
        CALLS.incrementAndGet();
    }

    /** Totals in nanoseconds, keyed by phase, for the end-to-end attribution. */
    public static java.util.Map<String, Long> totals() {
        java.util.Map<String, Long> m = new java.util.LinkedHashMap<>();
        m.put("columnAccess", T_COLUMN.get());
        m.put("outputAlloc", T_ALLOC.get());
        m.put("planLookup", T_PLAN.get());
        m.put("bind", T_BIND.get());
        m.put("inputSync", T_INSYNC.get());
        m.put("execute", T_EXEC.get());
        m.put("outputSync", T_OUTSYNC.get());
        m.put("wrap", T_WRAP.get());
        m.put("kernel", T_KERNEL.get());
        m.put("tvmCompile", T_TVM_COMPILE.get());
        m.put("driverInstall", T_DRIVER_INSTALL.get());
        m.put("dispatch", T_DISPATCH.get());
        m.put("call", T_CALL.get());
        m.put("calls", CALLS.get());
        m.put("coldCall", T_COLD_CALL.get());
        m.put("coldCalls", COLD_CALLS.get());
        return m;
    }

    /** One line of cumulative totals, in microseconds, for the benchmark to log. */
    public static String totalsLine() {
        StringBuilder sb = new StringBuilder("UDF-TOTALS");
        for (var e : totals().entrySet()) {
            if (e.getKey().equals("calls") || e.getKey().equals("coldCalls")) {
                sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
            } else {
                sb.append(' ').append(e.getKey()).append("Us=")
                        .append(String.format(java.util.Locale.ROOT, "%.1f", e.getValue() / 1000.0));
            }
        }
        return sb.toString();
    }

    private TornadoUdfMetrics() {
    }

    static void record(TornadoRapidsUdf.Timings timings) {
        BATCHES.incrementAndGet();
        ROWS.addAndGet(timings.numRows());
        T_COLUMN.addAndGet(timings.columnAccessNanos());
        T_ALLOC.addAndGet(timings.outputAllocNanos());
        T_PLAN.addAndGet(timings.planLookupNanos());
        T_BIND.addAndGet(timings.bindNanos());
        T_INSYNC.addAndGet(timings.inputSyncNanos());
        T_EXEC.addAndGet(timings.executeNanos());
        T_OUTSYNC.addAndGet(timings.outputSyncNanos());
        T_WRAP.addAndGet(timings.wrapNanos());
        T_KERNEL.addAndGet(timings.kernelNanos());
        T_TVM_COMPILE.addAndGet(Math.max(timings.tornadoCompilerNanos(), 0));
        T_DRIVER_INSTALL.addAndGet(Math.max(timings.driverInstallNanos(), 0));
        T_DISPATCH.addAndGet(Math.max(timings.dispatchNanos(), 0));
        if (timings.firstBatchForPlan()) {
            T_COLD_CALL.addAndGet(timings.totalNanos());
            COLD_CALLS.incrementAndGet();
        }
        synchronized (SAMPLES) {
            if (SAMPLES.size() < sampleLimit) {
                SAMPLES.add(timings);
            } else {
                DROPPED.incrementAndGet();
            }
        }
    }

    /** Batches that ran but whose per-batch detail was not kept, because the cap was reached. */
    public static long samplesDropped() {
        return DROPPED.get();
    }

    public static List<TornadoRapidsUdf.Timings> samples() {
        synchronized (SAMPLES) {
            return List.copyOf(SAMPLES);
        }
    }

    public static long batches() {
        return BATCHES.get();
    }

    public static long rows() {
        return ROWS.get();
    }

    /**
     * The batches that ran on a plan that already existed. The first batch on a plan pays for the
     * task graph, the TC compilation and NVRTC, which is a one-off cost of a different order; the
     * two are reported separately rather than averaged into one misleading number.
     */
    public static List<TornadoRapidsUdf.Timings> warmSamples() {
        return samples().stream().filter(t -> !t.firstBatchForPlan()).toList();
    }

    public static List<TornadoRapidsUdf.Timings> coldSamples() {
        return samples().stream().filter(TornadoRapidsUdf.Timings::firstBatchForPlan).toList();
    }

    public static void reset() {
        SAMPLES.clear();
        BATCHES.set(0);
        ROWS.set(0);
        DROPPED.set(0);
        for (AtomicLong a : new AtomicLong[] { T_COLUMN, T_ALLOC, T_PLAN, T_BIND, T_INSYNC, T_EXEC, T_OUTSYNC, T_WRAP, T_KERNEL, T_TVM_COMPILE, T_DRIVER_INSTALL, T_DISPATCH, T_CALL, CALLS, T_COLD_CALL, COLD_CALLS }) {
            a.set(0);
        }
    }

    /** A one-line summary of the medians, for logs. */
    public static String summary() {
        List<TornadoRapidsUdf.Timings> samples = samples();
        if (samples.isEmpty()) {
            // Two very different situations produce no samples, and saying the wrong one is
            // worse than saying nothing: with host instrumentation off the path may have run
            // perfectly and simply not been measured.
            return TornadoRapidsUdf.TIMING
                    ? "no batches ran through the TornadoVM UDF path"
                    : "host instrumentation is off (-Dtornado.spark.metrics=true to collect); "
                            + "this says nothing about whether batches ran";
        }
        List<TornadoRapidsUdf.Timings> warm = warmSamples();
        List<TornadoRapidsUdf.Timings> use = warm.isEmpty() ? samples : warm;
        return "batches=" + batches() + " sampledBatches=" + samples.size() + " droppedSamples=" + samplesDropped() //
                + " warmBatches=" + warm.size() + " rows=" + rows() //
                + " medianColumnAccessUs=" + medianMicros(use, TornadoRapidsUdf.Timings::columnAccessNanos) //
                + " medianOutputAllocUs=" + medianMicros(use, TornadoRapidsUdf.Timings::outputAllocNanos) //
                + " medianPlanLookupUs=" + medianMicros(use, TornadoRapidsUdf.Timings::planLookupNanos) //
                + " medianBindUs=" + medianMicros(use, TornadoRapidsUdf.Timings::bindNanos) //
                + " medianInputSyncUs=" + medianMicros(use, TornadoRapidsUdf.Timings::inputSyncNanos) //
                + " medianExecuteUs=" + medianMicros(use, TornadoRapidsUdf.Timings::executeNanos) //
                + " medianOutputSyncUs=" + medianMicros(use, TornadoRapidsUdf.Timings::outputSyncNanos) //
                + " medianWrapUs=" + medianMicros(use, TornadoRapidsUdf.Timings::wrapNanos)
                // The plan cache is keyed on the batch's shape, so a workload whose batches vary in
                // row count builds plans instead of reusing them. Without these counters a
                // batch-size regression is indistinguishable from the kernel simply being slower.
                + " classesDefined=" + TornadoKernelCache.classesDefined() //
                + " plansBuilt=" + TornadoKernelCache.plansBuilt() //
                + " plansReused=" + TornadoKernelCache.plansReused()
                // The first batch through a plan pays NVRTC compilation and the CUDA module load
                // inside execute(); separating it from the warm median is what keeps "compilation
                // cost" from being conflated with "first run was slow".
                + " coldExecuteUs=" + medianMicros(coldSamples(), TornadoRapidsUdf.Timings::executeNanos) //
                + " coldBatches=" + coldSamples().size();
    }

    public static double medianMicros(List<TornadoRapidsUdf.Timings> samples, java.util.function.ToLongFunction<TornadoRapidsUdf.Timings> phase) {
        long[] values = samples.stream().mapToLong(phase).sorted().toArray();
        if (values.length == 0) {
            return 0;
        }
        long median = values.length % 2 == 1 ? values[values.length / 2] : (values[values.length / 2 - 1] + values[values.length / 2]) / 2;
        return Math.round(median / 1000.0 * 100) / 100.0;
    }
}
