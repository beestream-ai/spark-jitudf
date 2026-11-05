/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.function.Supplier;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

/**
 * Runs TornadoVM calls with a context class loader that can see the TornadoVM modules.
 *
 * <p>
 * TornadoVM finds its backends with {@code ServiceLoader.load(TornadoBackendProvider.class)}, the
 * one-argument form, which looks in the <em>thread context</em> class loader. In a program launched
 * by TornadoVM's own {@code tornado} script that is the application loader and the CUDA backend is
 * found. In a Spark task it is not: Spark sets the context loader to the one holding the job's
 * jars, which knows nothing about the module path, so no backend is found and the task graph
 * resolves its device to null. The failure surfaces much later, as a {@code NullPointerException}
 * deep inside scheduling, which is why it is worth naming here.
 * </p>
 *
 * <p>
 * The loader used is the one that defined TornadoVM's own API, so service lookup sees the boot
 * layer's modules; Spark's loader is restored immediately afterwards, since Spark relies on it for
 * everything else in the task.
 * </p>
 */
public final class TornadoContext {

    private static final ClassLoader TORNADO_LOADER = TornadoExecutionPlan.class.getClassLoader();

    private TornadoContext() {
    }

    public static <T> T call(Supplier<T> action) {
        Thread current = Thread.currentThread();
        ClassLoader previous = current.getContextClassLoader();
        current.setContextClassLoader(TORNADO_LOADER);
        try {
            return action.get();
        } finally {
            current.setContextClassLoader(previous);
        }
    }

    public static void run(Runnable action) {
        call(() -> {
            action.run();
            return null;
        });
    }
}
