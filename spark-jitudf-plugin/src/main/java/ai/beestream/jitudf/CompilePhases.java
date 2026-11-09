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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Where the one-off cost of turning a UDF into a running kernel actually goes.
 *
 * <p>
 * The first run of a query is slower than the ones after it, and it is tempting to label the whole
 * of that excess "compilation overhead". That label is wrong, and this class exists so the claim
 * does not have to be made on faith. A first run also pays for JIT warm-up in Spark itself, a cold
 * page cache, broadcast construction, GPU context creation and allocator growth -- none of which
 * this integration causes, and all of which a plain RAPIDS first run pays too.
 * </p>
 *
 * <p>
 * What this measures is only the part that belongs to the integration, split into the phases that
 * actually happen, each timed where it occurs:
 * </p>
 * <ul>
 * <li><b>resolve</b> &mdash; recovering the UDF's implementation class from Spark's registration.</li>
 * <li><b>analyse</b> &mdash; deciding whether the bytecode is eligible, and with what signature.</li>
 * <li><b>generate</b> &mdash; emitting the parallel kernel class.</li>
 * <li><b>verify</b> &mdash; loading and checking the generated class on the driver.</li>
 * <li><b>defineClass</b> &mdash; defining it in the executor's generated-kernel classloader.</li>
 * <li><b>planBuild</b> &mdash; constructing the TornadoVM task graph and execution plan.</li>
 * </ul>
 *
 * <p>
 * TornadoVM compiles lazily, so NVRTC compilation and CUDA module load are <em>not</em> in
 * {@code planBuild}: they happen inside the first {@code execute()} of a plan, and are visible as
 * the difference between the first batch's execution time and the warm median, which
 * {@link TornadoUdfMetrics} already records separately. Nothing here is added to anything there.
 * </p>
 *
 * <p>
 * These are wall-clock nanoseconds accumulated per phase, across every UDF compiled in this JVM.
 * They are a decomposition of the integration's own one-off work and <b>not</b> a decomposition of
 * the cold-run excess; the two are reported side by side rather than one being presented as the
 * other.
 * </p>
 */
public final class CompilePhases {

    /** The phases, in the order they happen. */
    public enum Phase {
        RESOLVE, ANALYSE, GENERATE, VERIFY, DEFINE_CLASS, PLAN_BUILD
    }

    private static final Map<Phase, AtomicLong> NANOS = new LinkedHashMap<>();
    private static final Map<Phase, AtomicLong> COUNTS = new LinkedHashMap<>();

    static {
        for (Phase p : Phase.values()) {
            NANOS.put(p, new AtomicLong());
            COUNTS.put(p, new AtomicLong());
        }
    }

    private CompilePhases() {
    }

    /** Times {@code body} into {@code phase} and returns its value. */
    /**
     * Driver-side phase timing, gated by the same flag as the executor-side phase timers.
     *
     * <p>
     * These run during query planning, so with timing off the planner does the same work with no
     * clock reads and no counter updates. The flag is read from {@link TornadoRapidsUdf#TIMING} so
     * that one property controls host instrumentation on both sides and the two can never be
     * half-enabled.
     * </p>
     */
    public static <T> T time(Phase phase, java.util.function.Supplier<T> body) {
        if (!TornadoRapidsUdf.TIMING && !IntervalLog.ENABLED) {
            return body.get();
        }
        long t0 = System.nanoTime();
        try {
            return body.get();
        } finally {
            long t1 = System.nanoTime();
            if (TornadoRapidsUdf.TIMING) {
                NANOS.get(phase).addAndGet(t1 - t0);
                COUNTS.get(phase).incrementAndGet();
            }
            IntervalLog.record("phase:" + phase.name().toLowerCase(java.util.Locale.ROOT), t0, t1);
        }
    }

    /** Records an already-measured interval, for call sites that cannot be wrapped in a lambda. */
    public static void add(Phase phase, long nanos) {
        NANOS.get(phase).addAndGet(nanos);
        COUNTS.get(phase).incrementAndGet();
    }

    public static long nanos(Phase phase) {
        return NANOS.get(phase).get();
    }

    public static long count(Phase phase) {
        return COUNTS.get(phase).get();
    }

    public static void reset() {
        for (Phase p : Phase.values()) {
            NANOS.get(p).set(0);
            COUNTS.get(p).set(0);
        }
    }

    /** Nanoseconds and counts per phase, for the end-to-end attribution. */
    public static java.util.Map<String, Long> totals() {
        java.util.Map<String, Long> m = new LinkedHashMap<>();
        for (Phase p : Phase.values()) {
            String k = p.name().toLowerCase(java.util.Locale.ROOT);
            m.put(k, nanos(p));
            m.put(k + "Count", count(p));
        }
        return m;
    }

    /** One line per phase, in a form the benchmark can parse out of the driver log. */
    public static String summary() {
        StringBuilder sb = new StringBuilder("COMPILE-PHASES");
        long total = 0;
        for (Phase p : Phase.values()) {
            long n = nanos(p);
            total += n;
            sb.append(' ').append(p.name().toLowerCase(java.util.Locale.ROOT))
                    .append("Us=").append(String.format(java.util.Locale.ROOT, "%.1f", n / 1000.0))
                    .append(" n").append(p.name().charAt(0)).append('=').append(count(p));
        }
        sb.append(String.format(java.util.Locale.ROOT, " totalUs=%.1f", total / 1000.0));
        return sb.toString();
    }
}
