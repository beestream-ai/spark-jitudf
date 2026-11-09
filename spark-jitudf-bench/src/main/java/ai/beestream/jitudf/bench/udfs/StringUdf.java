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

import org.apache.spark.sql.api.java.UDF1;

/**
 * A UDF neither compiler can take: RAPIDS has no expression for it and a GPU kernel has no strings.
 * It is here to show the refusal, which has to leave the query on the CPU and still be correct.
 */
public class StringUdf implements UDF1<Double, String> {

    private static final long serialVersionUID = 1L;

    @Override
    public String call(Double a) {
        return a > 0 ? "positive" : "non-positive";
    }
}
