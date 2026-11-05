/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * What the RAPIDS {@code udf-compiler} found when it declined a UDF: the reasons it ran into and
 * the message that goes with them.
 *
 * <p>
 * This is the typed result that the RAPIDS fork records at the rejection site, instead of logging
 * an exception at debug level and returning the original {@code ScalaUDF} with nothing to show for
 * it. It is deliberately a summary rather than a transcript: the dispatcher uses it to decide
 * whether to look further, and the looking is done by {@link UdfEligibilityAnalyzer} over the
 * bytecode, not by re-reading this.
 * </p>
 */
public final class UdfRejection {

    private final String udfName;
    private final Set<RejectionReason> reasons;
    private final String diagnostics;

    public UdfRejection(String udfName, Set<RejectionReason> reasons, String diagnostics) {
        this.udfName = udfName == null ? "<unknown>" : udfName;
        this.reasons = Collections.unmodifiableSet(new LinkedHashSet<>(reasons));
        this.diagnostics = diagnostics == null ? "" : diagnostics;
    }

    public String udfName() {
        return udfName;
    }

    public Set<RejectionReason> reasons() {
        return reasons;
    }

    public String diagnostics() {
        return diagnostics;
    }

    /**
     * Whether these reasons are the kind a TornadoVM kernel can plausibly handle, which is worth an
     * eligibility analysis. Loops, counted increments, primitive arrays and unsupported maths are:
     * they are ordinary numeric code that has no Catalyst expression, not code doing something a
     * kernel could not do. A wrapper that could not even be unwrapped is not.
     */
    public boolean isWorthAnalysing() {
        if (reasons.contains(RejectionReason.WRAPPER_EXTRACTION) || reasons.contains(RejectionReason.EXCEPTION_HANDLING)) {
            return false;
        }
        return reasons.contains(RejectionReason.LOOP) //
                || reasons.contains(RejectionReason.IINC) //
                || reasons.contains(RejectionReason.PRIMITIVE_ARRAY) //
                || reasons.contains(RejectionReason.UNSUPPORTED_MATH) //
                || reasons.contains(RejectionReason.UNSUPPORTED_INVOKE) //
                || reasons.contains(RejectionReason.UNSUPPORTED_BYTECODE) //
                || reasons.contains(RejectionReason.OTHER);
    }

    @Override
    public String toString() {
        return "UdfRejection[" + udfName + " " + reasons + (diagnostics.isEmpty() ? "" : ": " + diagnostics) + "]";
    }
}
