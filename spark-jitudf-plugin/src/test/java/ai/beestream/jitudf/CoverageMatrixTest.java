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

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * The GPU-eligibility matrix over {@link CoverageCorpus}: every intended-safe pattern must stay
 * accepted and every counterexample must be refused.
 *
 * <p>
 * Set {@code -Dcoverage.report=only} to print and write the matrix without asserting, which is how
 * the pre-change baseline was recorded. The matrix is written to
 * {@code target/coverage-matrix.tsv} either way.
 * </p>
 */
class CoverageMatrixTest {

    @Test
    void safePatternsStayEligibleAndCounterexamplesAreRefused() throws Exception {
        boolean reportOnly = "only".equals(System.getProperty("coverage.report"));
        TornadoUdfProvider.resetDecisions();
        List<String> rows = new ArrayList<>();
        List<String> violations = new ArrayList<>();
        rows.add("id\tintent\tresult\tpattern\tdecision");
        for (CoverageCorpus.Case c : CoverageCorpus.all()) {
            ProviderHarness.Outcome o = ProviderHarness.cudf(c.function(), c.inputs(), c.output(), c.nullable(), c.id(), c.incompatibleOps());
            String result = o.accepted() ? "GPU" : "CPU";
            rows.add(c.id() + "\t" + c.intent() + "\t" + result + "\t" + c.pattern() + "\t" + o.decision());
            if (c.intent() == CoverageCorpus.Intent.SAFE && !o.accepted()) {
                violations.add(c.id() + " (" + c.pattern() + ") lost GPU eligibility: " + o.decision());
            }
            if (c.intent() == CoverageCorpus.Intent.UNSAFE && o.accepted()) {
                violations.add(c.id() + " (" + c.pattern() + ") is still accepted: " + o.decision());
            }
        }
        Path out = Path.of("target", "coverage-matrix.tsv");
        Files.createDirectories(out.getParent());
        Files.write(out, rows);
        rows.forEach(r -> System.out.println("COVERAGE " + r));
        if (!reportOnly) {
            assertTrue(violations.isEmpty(), () -> String.join("\n", violations));
        }
    }
}
