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
 * A boolean result: eligible for the cuDF {@code transform} backend, refused by the runtime one.
 *
 * <p>
 * The cuDF backend writes a BOOL8 column and the device function's output pointer is a C++
 * {@code bool}. The TornadoVM runtime backend binds the result to a native array instead, and
 * TornadoVM has no {@code BooleanArray} -- so rather than write booleans into a {@code ByteArray}
 * and hope the widths line up, the provider declines the UDF and says which backend would take it.
 * </p>
 */
public class BooleanUdf implements UDF1<Double, Boolean> {

    private static final long serialVersionUID = 1L;

    @Override
    public Boolean call(Double value) {
        return value > 100.0;
    }
}
