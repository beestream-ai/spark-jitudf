/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Keeps the work that must not happen per batch from happening per batch.
 *
 * <p>
 * Two things are cached, at two different scopes. The generated kernel <em>class</em> is defined
 * once per executor JVM and shared: it is immutable bytecode, and defining it again would make
 * TornadoVM compile a second, identical kernel under a different name. The execution <em>plan</em>
 * is per thread, because a device view may be bound by one plan at a time and Spark runs several
 * tasks at once in the same JVM; sharing one plan across tasks would be a race over the bindings,
 * not a saving.
 * </p>
 *
 * <p>
 * Within a thread, plans are kept per row count, since the row count is fixed in the task graph. A
 * query usually has one batch size and one smaller remainder, so the map stays tiny; it is bounded
 * anyway, so a workload with many distinct batch sizes evicts rather than growing without end.
 * </p>
 */
public final class TornadoKernelCache {

    /** How many row counts one thread keeps plans for before evicting the least recently used. */
    private static final int MAX_PLANS_PER_UDF = 8;

    private static final Map<String, Class<?>> KERNEL_CLASSES = new ConcurrentHashMap<>();
    private static final GeneratedKernelClassLoader LOADER = new GeneratedKernelClassLoader(TornadoKernelCache.class.getClassLoader());

    /** How many distinct UDFs one thread keeps plans for before evicting the least recently used. */
    private static final int MAX_UDFS_PER_THREAD = 32;

    /**
     * A plan is identified by the row count and the list-child capacity its views were built for,
     * and by the GPU it was pinned to: a plan compiled and queued for one device cannot be handed
     * another device's pointers, so a thread that changes CUDA device gets a different plan rather
     * than silently reusing one built elsewhere.
     */
    private record PlanShape(int numRows, long listValueCapacity, int cudaOrdinal) {
    }

    /**
     * One entry per UDF this thread has run, each holding that UDF's plans by shape.
     *
     * <p>
     * Both levels are bounded. A plan owns a task graph, compiled native code and device views, so
     * an executor thread that meets an unusual number of distinct UDFs or batch sizes would
     * otherwise accumulate device resources for the life of the JVM. Eviction closes what it drops.
     * </p>
     */
    private static final ThreadLocal<Map<String, Map<PlanShape, KernelExecution>>> PLANS = ThreadLocal.withInitial(() -> new LinkedHashMap<>(8, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Map<PlanShape, KernelExecution>> eldest) {
            if (size() > MAX_UDFS_PER_THREAD) {
                eldest.getValue().values().forEach(KernelExecution::close);
                return true;
            }
            return false;
        }
    });

    private static final java.util.concurrent.atomic.AtomicLong CLASSES_DEFINED = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong PLANS_BUILT = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong PLANS_REUSED = new java.util.concurrent.atomic.AtomicLong();

    private TornadoKernelCache() {
    }

    /** The kernel class for {@code descriptor}, defined on first use in this JVM. */
    public static Class<?> kernelClass(KernelDescriptor descriptor) {
        return KERNEL_CLASSES.computeIfAbsent(descriptor.cacheKey(), key -> {
            CLASSES_DEFINED.incrementAndGet();
            return CompilePhases.time(CompilePhases.Phase.DEFINE_CLASS,
                    () -> LOADER.define(descriptor.generatedClassName(), descriptor.generatedClassFile()));
        });
    }

    /** The classloader the generated classes live in; TornadoVM reads their bytecode through it. */
    public static GeneratedKernelClassLoader loader() {
        return LOADER;
    }

    /**
     * This thread's plan for {@code descriptor} at {@code numRows}, built on first use.
     *
     * <p>
     * {@code listElements} is how many list elements this batch's child columns hold; it is rounded
     * up to a power of two so that batches of slightly different list lengths share one plan rather
     * than each building their own. The rounding only sizes the views' capacity, not the compiled
     * kernel: the kernel reads the length from the offsets.
     * </p>
     */
    static KernelExecution execution(KernelDescriptor descriptor, int numRows, long listElements) {
        PlanShape shape = new PlanShape(numRows, roundUpToPowerOfTwo(listElements), DeviceAffinity.currentCudaOrdinal());
        Map<PlanShape, KernelExecution> byShape = PLANS.get().computeIfAbsent(descriptor.cacheKey(), key -> new LinkedHashMap<>(8, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<PlanShape, KernelExecution> eldest) {
                if (size() > MAX_PLANS_PER_UDF) {
                    eldest.getValue().close();
                    return true;
                }
                return false;
            }
        });
        KernelExecution existing = byShape.get(shape);
        if (existing != null) {
            PLANS_REUSED.incrementAndGet();
            return existing;
        }
        // Note what this does and does not include: TornadoVM compiles lazily, so this times the
        // task-graph and execution-plan construction only. NVRTC compilation and the CUDA module
        // load happen inside the first execute() of the plan, and show up in the first batch's
        // execution time rather than here.
        KernelExecution built = CompilePhases.time(CompilePhases.Phase.PLAN_BUILD,
                () -> new KernelExecution(descriptor, kernelClass(descriptor), numRows, shape.listValueCapacity(), shape.cudaOrdinal()));
        PLANS_BUILT.incrementAndGet();
        byShape.put(shape, built);
        return built;
    }

    private static long roundUpToPowerOfTwo(long value) {
        if (value <= 1) {
            return 1;
        }
        return Long.highestOneBit(value - 1) << 1;
    }

    /** Counters for the benchmark breakdown: classes defined, plans built, plans reused. */
    public static long classesDefined() {
        return CLASSES_DEFINED.get();
    }

    public static long plansBuilt() {
        return PLANS_BUILT.get();
    }

    public static long plansReused() {
        return PLANS_REUSED.get();
    }
}
