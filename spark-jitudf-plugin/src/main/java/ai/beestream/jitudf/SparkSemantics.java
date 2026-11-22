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

import org.apache.spark.sql.internal.SQLConf;

/**
 * The Spark and RAPIDS settings that decide whether a compiled UDF would still mean what the CPU
 * means.
 *
 * <p>
 * Both answers here are reasons to <em>refuse</em>, and that is deliberate. A GPU kernel that
 * computes a different number from the CPU is worse than no kernel at all, because the query still
 * returns rows and nobody looks. So each setting this integration has not implemented support for
 * is read while planning, and a UDF it affects is left on the path it already had.
 * </p>
 *
 * <p>
 * The configuration is read through {@link SQLConf#get()} rather than passed in, so that the
 * signature RAPIDS calls reflectively does not have to change. On the driver, during planning, that
 * is the session's own conf.
 * </p>
 */
public final class SparkSemantics {

    /** RAPIDS's own switch for operations whose results differ from Spark's CPU implementation. */
    public static final String INCOMPATIBLE_OPS = "spark.rapids.sql.incompatibleOps.enabled";

    private SparkSemantics() {
    }

    /**
     * Why this UDF must not be compiled under the current configuration, or null.
     *
     * @param outputKind
     *         what the UDF returns; floating point is the case that needs the opt-in.
     */
    public static String refusalFor(PrimitiveKind outputKind, java.util.List<InputSpec> inputs) {
        // ANSI mode is deliberately not consulted here, and it took a failing test to see why.
        //
        // Spark 4.0 turns `spark.sql.ansi.enabled` on by default, so a gate on it would refuse every
        // UDF on a current Spark. It would also be refusing the wrong thing: ANSI governs the
        // semantics of Spark SQL *expressions* -- casts, arithmetic written in SQL -- and a Java UDF
        // is opaque to it. Inside the UDF, Java's own rules apply whether ANSI is on or off: an int
        // overflow wraps, and division by zero throws ArithmeticException. Spark does not
        // reinterpret the body, so compiling it does not change what ANSI means for the query.
        //
        // The divergence that does exist is Java's, not ANSI's, and it is handled where the bytecode
        // is visible: see the integer-division check in TornadoUdfProvider.
        if (usesFloatingPoint(outputKind, inputs) && !incompatibleOpsEnabled()) {
            // Not a precision complaint: TornadoVM's CUDA backend contracts a*b+c into fma, which
            // Java forbids, so results can differ from the CPU in the last ulp. More accurate is
            // still different, and RAPIDS gates differences behind this flag rather than judging
            // which direction they go in.
            return "this UDF uses floating point, whose GPU result can differ from Spark's in the last ulp (fma contraction); set " + INCOMPATIBLE_OPS + "=true to allow it";
        }
        return null;
    }

    static boolean usesFloatingPoint(PrimitiveKind outputKind, java.util.List<InputSpec> inputs) {
        if (outputKind == PrimitiveKind.FLOAT || outputKind == PrimitiveKind.DOUBLE) {
            return true;
        }
        for (InputSpec input : inputs) {
            if (input.kind() == PrimitiveKind.FLOAT || input.kind() == PrimitiveKind.DOUBLE) {
                return true;
            }
        }
        return false;
    }

    static boolean ansiEnabled() {
        try {
            return SQLConf.get().ansiEnabled();
        } catch (Throwable t) {
            // No active session (a unit test, or a call off the planning thread). Reading the
            // configuration is not the point of failure this should surface, so assume the
            // permissive default and let the rest of the analysis decide.
            return false;
        }
    }

    static boolean incompatibleOpsEnabled() {
        try {
            return Boolean.parseBoolean(SQLConf.get().getConfString(INCOMPATIBLE_OPS, "false"));
        } catch (Throwable t) {
            return false;
        }
    }
}
