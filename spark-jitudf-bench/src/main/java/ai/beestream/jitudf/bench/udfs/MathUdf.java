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

import org.apache.spark.sql.api.java.UDF3;

/**
 * Fixed-width numeric arithmetic over three columns, with no loop: this is the shape that isolates
 * the per-batch cost of the integration from the cost of the computation.
 */
public class MathUdf implements UDF3<Double, Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    @Override
    public Double call(Double a, Double b, Double c) {
        double scaled = a * 1.5 + b * 0.25 - c;
        double norm = Math.sqrt(a * a + b * b + c * c);
        return Math.cbrt(scaled) + norm * 0.5;
    }
}
