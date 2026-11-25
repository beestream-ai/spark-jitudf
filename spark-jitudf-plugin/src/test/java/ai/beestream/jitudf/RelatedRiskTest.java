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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.Serializable;

import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

/**
 * Related risks from the review (its section E) checked for a further wrong-code path. Neither is
 * one; these pin the behaviour that makes them safe.
 */
class RelatedRiskTest {
    public static final class ClinitConstant implements UDF1<Integer, Integer> {
        static final int K = Integer.parseInt(System.getProperty("probe.k", "7"));

        @Override
        public Integer call(Integer x) {
            return x + K;
        }
    }

    @Test
    void erasedInvocationUnboxesNullAndClinitStateIsRefused() {
        TornadoUdfProvider.resetDecisions();
        scala.Function1<Long, Long> scalaShape = (scala.Function1<Long, Long> & Serializable) CoverageCorpus.ScalaShapes::plusOne;
        ProviderHarness.Outcome n = ProviderHarness.cudf(scalaShape, new DataType[] { DataTypes.LongType }, DataTypes.LongType, true, "scalaNullable");
        // This provider-only fixture calls the erased Java SAM. The Catalyst primitive
        // encoder's external null short circuit is exercised through SQL in CodexDifferentialTest.
        assertThrows(NullPointerException.class, () -> scalaShape.apply(null));
        assertTrue(n.accepted(), n::decision);
        assertTrue(n.cuda().contains("if (arg0_valid == 0LL) return 3"), n::cuda);
        ProviderHarness.Outcome c = ProviderHarness.cudf(new ClinitConstant(), new DataType[] { DataTypes.IntegerType }, DataTypes.IntegerType, false, "clinit");
        // A static final computed in <clinit> is not a compile-time constant; the analyzer refuses it before TC.
        assertFalse(c.accepted(), c::decision);
        assertTrue(c.decision().contains("reads static state K"), c::decision);
    }
}
