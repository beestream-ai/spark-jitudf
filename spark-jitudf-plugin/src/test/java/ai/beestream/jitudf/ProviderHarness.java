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

import java.util.function.Supplier;

import org.apache.spark.sql.types.DataType;

/**
 * Drives {@link TornadoUdfProvider} the way the RAPIDS bridge does, with the cudf-transform backend
 * selected and, optionally, RAPIDS's incompatible-ops switch set, and reports what it decided.
 *
 * <p>
 * The provider reads both settings from the JVM: the backend from a system property, and the
 * floating-point switch from {@code SQLConf.get()}, which without an active session is a fresh
 * default conf on this thread. Each call restores both afterwards, so tests do not leak state into
 * each other.
 * </p>
 */
final class ProviderHarness {

    private ProviderHarness() {
    }

    /** What the provider returned for one UDF, and why. */
    record Outcome(Object udf, String decision) {
        boolean accepted() {
            return udf != null;
        }

        /** The CUDA shipped to the executor, for either executor-side shape. */
        String cuda() {
            if (udf instanceof CudfTransformUdf t) {
                return t.cudaSource();
            }
            if (udf instanceof CudfListReductionUdf l) {
                return l.cudaSource();
            }
            return null;
        }

        @Override
        public String toString() {
            return (accepted() ? "ACCEPTED " : "REFUSED  ") + decision;
        }
    }

    static Outcome cudf(Object function, DataType[] inputs, DataType output, boolean nullable, String name) {
        return cudf(function, inputs, output, nullable, name, false);
    }

    static Outcome cudf(Object function, DataType[] inputs, DataType output, boolean nullable, String name, boolean incompatibleOps) {
        return withSettings(incompatibleOps, () -> {
            Object udf = TornadoUdfProvider.tryCompile(function, inputs, output, nullable, true, name, new String[] { "LOOP" }, "probe");
            return new Outcome(udf, TornadoUdfProvider.decisions().get(name));
        });
    }

    static <T> T withSettings(boolean incompatibleOps, Supplier<T> body) {
        String previous = System.getProperty("tornado.spark.backend");
        System.setProperty("tornado.spark.backend", TornadoUdfProvider.BACKEND_CUDF_TRANSFORM);
        org.apache.spark.sql.internal.SQLConf conf = org.apache.spark.sql.internal.SQLConf.get();
        String previousOps = conf.getConfString(SparkSemantics.INCOMPATIBLE_OPS, null);
        conf.setConfString(SparkSemantics.INCOMPATIBLE_OPS, Boolean.toString(incompatibleOps));
        try {
            return body.get();
        } finally {
            if (previousOps == null) {
                conf.unsetConf(SparkSemantics.INCOMPATIBLE_OPS);
            } else {
                conf.setConfString(SparkSemantics.INCOMPATIBLE_OPS, previousOps);
            }
            if (previous == null) {
                System.clearProperty("tornado.spark.backend");
            } else {
                System.setProperty("tornado.spark.backend", previous);
            }
        }
    }
}
