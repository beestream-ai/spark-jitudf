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
