/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Records the start and end of each timed span so that their disjointness can be checked instead
 * of argued.
 *
 * <p>
 * The elapsed-time attribution adds three classes of span together: every UDF call, every
 * planner-side compiler phase, and the remainder. Adding them is only valid if a call never
 * overlaps another call and a compiler phase never runs inside a call. Both are true by
 * construction in the reviewed code — but "by construction" is exactly the kind of claim that
 * turned out to be wrong about host dispatch and device execution, so here the intervals are
 * written out and tested against each other by {@code validate-intervals.py}.
 * </p>
 *
 * <p>
 * Enabled by {@code -Dtornado.spark.intervals=<file>}, off otherwise, and never on during a timed
 * measurement: writing a line per span is itself observation. The thread id is recorded because
 * two spans on different threads overlapping is a different fact from two spans on one thread
 * overlapping, and only the first is possible here. {@link System#nanoTime} is a single monotonic
 * source across threads on this platform, which is what makes the comparison meaningful.
 * </p>
 */
public final class IntervalLog {

    private static final String FILE = System.getProperty("tornado.spark.intervals", "");
    /** Whether interval recording is on at all; folded away by the JIT when it is not. */
    public static final boolean ENABLED = !FILE.isEmpty();

    private static final StringBuilder PENDING = new StringBuilder(1 << 16);

    private IntervalLog() {
    }

    /** Records one span: what it was, which thread ran it, and the two instants that bound it. */
    public static void record(String kind, long startNanos, long endNanos) {
        if (!ENABLED) {
            return;
        }
        Thread t = Thread.currentThread();
        synchronized (PENDING) {
            PENDING.append(kind).append(',').append(t.threadId()).append(',') //
                    .append(startNanos).append(',').append(endNanos).append('\n');
            if (PENDING.length() > 1 << 15) {
                flush();
            }
        }
    }

    /** Writes whatever has accumulated; called on exit and whenever the buffer grows. */
    public static void flush() {
        if (!ENABLED) {
            return;
        }
        synchronized (PENDING) {
            if (PENDING.length() == 0) {
                return;
            }
            try {
                Path p = Path.of(FILE);
                if (p.getParent() != null) {
                    Files.createDirectories(p.getParent());
                }
                Files.writeString(p, PENDING, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                PENDING.setLength(0);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static {
        if (ENABLED) {
            Runtime.getRuntime().addShutdownHook(new Thread(IntervalLog::flush));
        }
    }
}
