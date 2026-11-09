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
package ai.beestream.jitudf.bench.udfs;

import org.apache.spark.sql.api.java.UDF2;

/** A loop that calls a static helper in the same class, to exercise the reachable-method walk. */
public class HelperLoopUdf implements UDF2<Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    private static double step(double acc, double a, double b, int i) {
        return acc * 0.5 + a * b + i * 0.125;
    }

    @Override
    public Double call(Double a, Double b) {
        double acc = 0.0;
        for (int i = 0; i < 32; i++) {
            acc = step(acc, a, b, i);
        }
        return acc;
    }
}
