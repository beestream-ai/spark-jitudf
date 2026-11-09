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
 * A counted loop over two numeric columns. RAPIDS's udf-compiler builds Catalyst expression trees,
 * which have no way to express a loop, so it declines this one and the query would otherwise fall
 * back to the CPU.
 */
public class LoopUdf implements UDF2<Double, Double, Double> {

    private static final long serialVersionUID = 1L;

    private final int iterations;

    public LoopUdf(int iterations) {
        this.iterations = iterations;
    }

    @Override
    public Double call(Double a, Double b) {
        double acc = 0.0;
        double x = a;
        double y = b;
        for (int i = 0; i < 64; i++) {
            acc = acc * 0.5 + x * y;
            x = x * 1.000001 + 0.5;
            y = y * 0.999999 - 0.25;
        }
        return acc;
    }
}
