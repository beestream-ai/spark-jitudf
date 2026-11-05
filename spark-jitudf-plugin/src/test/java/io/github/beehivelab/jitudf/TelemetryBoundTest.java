/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * ADV-10, stated using only the API that existed before the fix, so that it can be run against the
 * unfixed code. The permanent test uses the new test seam; this one cannot, because that seam is
 * part of the fix.
 */
class TelemetryBoundTest {

    private static final int CAP = 100_000;

    @Test
    void perBatchSamplesDoNotGrowWithoutBound() {
        TornadoUdfMetrics.reset();
        try {
            for (int i = 0; i < CAP + 5_000; i++) {
                TornadoUdfMetrics.record(new TornadoRapidsUdf.Timings(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 64, false, 0, 0, 0, 0, 0));
            }
            assertEquals(CAP + 5_000, TornadoUdfMetrics.batches(), "every batch must still be counted");
            assertTrue(TornadoUdfMetrics.samples().size() <= CAP,
                    "per-batch records must stop accumulating at the cap, but the list held " + TornadoUdfMetrics.samples().size());
        } finally {
            TornadoUdfMetrics.reset();
        }
    }
}
