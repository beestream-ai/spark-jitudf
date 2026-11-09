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

/**
 * Why the RAPIDS {@code udf-compiler} would not turn a JVM UDF into Catalyst expressions.
 *
 * <p>
 * RAPIDS raises one exception and stops, so a reason is cheap triage, not a verdict: it says
 * roughly what kind of construct got in the way, which is enough to decide whether it is worth
 * handing the UDF to {@link UdfEligibilityAnalyzer} for a proper look at the whole reachable
 * method graph. Several of these -- loops, counted increments, primitive arrays -- are exactly what
 * a TornadoVM kernel does well, which is the point of classifying them at all.
 * </p>
 */
public enum RejectionReason {

    /** A loop of any kind; RAPIDS builds expression trees, which have no way to express one. */
    LOOP,

    /** An {@code iinc}, which in practice means a counted loop or an accumulating index. */
    IINC,

    /** A primitive array load or store ({@code faload}, {@code dastore}, ...). */
    PRIMITIVE_ARRAY,

    /** A {@code java.lang.Math} (or similar) call RAPIDS has no Catalyst equivalent for. */
    UNSUPPORTED_MATH,

    /** A method call RAPIDS cannot inline or map. */
    UNSUPPORTED_INVOKE,

    /** A field read or write on an object. */
    OBJECT_ACCESS,

    /** {@code athrow}, a {@code try}/{@code catch}, or anything else with exception semantics. */
    EXCEPTION_HANDLING,

    /** The user's UDF could not be found behind Spark's wrapper at all. */
    WRAPPER_EXTRACTION,

    /** A bytecode RAPIDS does not implement, not covered by the categories above. */
    UNSUPPORTED_BYTECODE,

    /** Something else, including a failure whose cause could not be classified. */
    OTHER
}
