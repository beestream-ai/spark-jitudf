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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

/** The host-side ranking of the salt search, including the degenerate arms it used to rank first. */
class SaltSearchWelchTest {

    /** n, sum and sum of squares of a sample. */
    private static long[] stats(long... values) {
        long s = 0;
        long q = 0;
        for (long v : values) {
            s += v;
            q += v * v;
        }
        return new long[] { values.length, s, q };
    }

    private static double t(long[] a, long[] b) {
        return SaltSearch.welch(a[0], a[1], a[2], b[0], b[1], b[2]);
    }

    @Test
    void matchesTheTextbookValueOnOrdinaryArms() {
        // a = {1,2,3,4}: mean 2.5, var 5/3; b = {2,4,6,8}: mean 5, var 20/3.
        double expected = (2.5 - 5) / Math.sqrt((5.0 / 3) / 4 + (20.0 / 3) / 4);
        assertEquals(expected, t(stats(1, 2, 3, 4), stats(2, 4, 6, 8)), 1e-12);
    }

    @Test
    void constantArmsWithDifferentMeansAreMaximallyUnbalanced() {
        assertEquals(Double.NEGATIVE_INFINITY, t(stats(3, 3, 3), stats(5, 5, 5)));
        assertEquals(Double.POSITIVE_INFINITY, t(stats(5, 5, 5), stats(3, 3, 3)));
    }

    @Test
    void constantArmsWithEqualMeansAreBalanced() {
        assertEquals(0.0, t(stats(4, 4), stats(4, 4, 4)));
    }

    @Test
    void anArmTooSmallForAVarianceIsNeverRankedBalanced() {
        assertEquals(Double.POSITIVE_INFINITY, t(stats(7), stats(1, 2, 3)));
        assertEquals(Double.POSITIVE_INFINITY, t(stats(), stats(1, 2, 3)));
    }

    @Test
    void aDegenerateSaltRanksLastNotFirst() {
        // Salt 1: ordinary, small difference. Salt 2: constant arms with different means, which the
        // old routine scored 0 and ranked as the best-balanced salt of all.
        long[] a1 = stats(10, 11, 12, 13);
        long[] b1 = stats(10, 12, 11, 14);
        SaltSearch.Aggregates ordinary = new SaltSearch.Aggregates(1, 0, a1[0], b1[0], a1[1], b1[1], a1[2], b1[2], a1[1], b1[1], a1[2], b1[2]);
        long[] a2 = stats(3, 3, 3, 3);
        long[] b2 = stats(9, 9, 9, 9);
        SaltSearch.Aggregates degenerate = new SaltSearch.Aggregates(2, 0, a2[0], b2[0], a2[1], b2[1], a2[2], b2[2], a2[1], b2[1], a2[2], b2[2]);
        List<SaltSearch.Ranked> ranked = SaltSearch.rank(List.of(degenerate, ordinary));
        assertEquals(1L, ranked.get(0).salt());
        assertEquals(2L, ranked.get(1).salt());
        assertTrue(Double.isInfinite(ranked.get(1).maxT()));
    }
}
