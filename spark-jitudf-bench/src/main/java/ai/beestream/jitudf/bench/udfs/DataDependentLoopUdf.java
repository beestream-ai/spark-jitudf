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

/**
 * A loop whose trip count depends on the data, which is the case an expression tree cannot express
 * even in principle. The bound is clamped so one row cannot run away with the whole warp.
 */
public class DataDependentLoopUdf implements UDF2<Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    @Override
    public Double call(Double a, Double b) {
        int steps = (int) (a % 97.0);
        if (steps < 1) {
            steps = 1;
        }
        if (steps > 96) {
            steps = 96;
        }
        double acc = b;
        for (int i = 0; i < steps; i++) {
            acc = acc * 0.97 + a * 0.013 + i * 0.001;
        }
        return acc;
    }
}
