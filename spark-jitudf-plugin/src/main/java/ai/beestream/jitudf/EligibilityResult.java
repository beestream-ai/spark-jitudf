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

import java.util.List;

/** Whether a UDF can be lowered to a kernel, and if not, exactly what stopped it. */
public final class EligibilityResult {

    private final boolean eligible;
    private final String reason;
    private final List<InputSpec> inputs;
    private final PrimitiveKind outputKind;
    private final ResolvedUdf udf;

    private EligibilityResult(boolean eligible, String reason, List<InputSpec> inputs, PrimitiveKind outputKind, ResolvedUdf udf) {
        this.eligible = eligible;
        this.reason = reason;
        this.inputs = inputs;
        this.outputKind = outputKind;
        this.udf = udf;
    }

    public static EligibilityResult eligible(ResolvedUdf udf, List<InputSpec> inputs, PrimitiveKind outputKind) {
        return new EligibilityResult(true, "", inputs, outputKind, udf);
    }

    public static EligibilityResult ineligible(String reason) {
        return new EligibilityResult(false, reason, List.of(), null, null);
    }

    public boolean isEligible() {
        return eligible;
    }

    public String reason() {
        return reason;
    }

    public List<InputSpec> inputs() {
        return inputs;
    }

    public PrimitiveKind outputKind() {
        return outputKind;
    }

    public ResolvedUdf udf() {
        return udf;
    }

    @Override
    public String toString() {
        return eligible ? "TORNADOVM_ELIGIBLE(" + inputs + " -> " + outputKind + ")" : "TORNADOVM_INELIGIBLE(" + reason + ")";
    }
}
