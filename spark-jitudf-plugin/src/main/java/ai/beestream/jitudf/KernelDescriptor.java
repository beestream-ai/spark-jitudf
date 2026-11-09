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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Everything the executor needs to build and run the kernel for one UDF, and nothing that only
 * exists on the driver.
 *
 * <p>
 * It is {@link Serializable} on purpose: the driver decides eligibility and generates the kernel
 * class, the executor compiles and runs it, and what crosses between them is this description plus
 * the generated class bytes -- never a device pointer, a live task graph or an execution plan. The
 * driver needs no GPU to produce it.
 * </p>
 */
public final class KernelDescriptor implements Serializable {

    private static final long serialVersionUID = 1L;

    /** What one kernel parameter stands for. */
    public enum Role {
        /** A fixed-width column of input, read as a device view of the cuDF buffer. */
        INPUT_COLUMN,
        /** A LIST column's offsets: {@code numRows + 1} int32 values. */
        INPUT_LIST_OFFSETS,
        /** A LIST column's flattened child values. */
        INPUT_LIST_VALUES,
        /** The output column, written into an RMM allocation of ours. */
        OUTPUT_COLUMN,
        /** The number of rows in this batch, passed as an ordinary {@code int}. */
        NUM_ROWS
    }

    public static final class Argument implements Serializable {
        private static final long serialVersionUID = 1L;

        private final Role role;
        private final PrimitiveKind kind;
        private final int columnIndex;

        public Argument(Role role, PrimitiveKind kind, int columnIndex) {
            this.role = role;
            this.kind = kind;
            this.columnIndex = columnIndex;
        }

        public Role role() {
            return role;
        }

        public PrimitiveKind kind() {
            return kind;
        }

        /** Which of {@code evaluateColumnar}'s arguments this reads, or -1 when it reads none. */
        public int columnIndex() {
            return columnIndex;
        }

        @Override
        public String toString() {
            return role + (kind == null ? "" : "(" + kind + ")") + (columnIndex >= 0 ? "@" + columnIndex : "");
        }
    }

    private final String generatedClassName;
    private final String kernelMethodName;
    private final byte[] generatedClassFile;
    private final List<Argument> arguments;
    private final List<InputSpec> inputs;
    private final PrimitiveKind outputKind;
    private final String udfName;
    private final String cacheKey;

    public KernelDescriptor(String generatedClassName, String kernelMethodName, byte[] generatedClassFile, List<Argument> arguments, List<InputSpec> inputs, PrimitiveKind outputKind,
            String udfName, String cacheKey) {
        this.generatedClassName = generatedClassName;
        this.kernelMethodName = kernelMethodName;
        this.generatedClassFile = generatedClassFile.clone();
        this.arguments = Collections.unmodifiableList(new ArrayList<>(arguments));
        this.inputs = Collections.unmodifiableList(new ArrayList<>(inputs));
        this.outputKind = outputKind;
        this.udfName = udfName;
        this.cacheKey = cacheKey;
    }

    public String generatedClassName() {
        return generatedClassName;
    }

    public String kernelMethodName() {
        return kernelMethodName;
    }

    public byte[] generatedClassFile() {
        return generatedClassFile.clone();
    }

    public List<Argument> arguments() {
        return arguments;
    }

    public List<InputSpec> inputs() {
        return inputs;
    }

    public PrimitiveKind outputKind() {
        return outputKind;
    }

    public String udfName() {
        return udfName;
    }

    /**
     * Identifies the compiled kernel: the UDF's method and the schema it was generated for. The
     * device and the layout of the arguments are not in it, because TornadoVM keys its own
     * compilation cache on those; this only has to stop two different UDFs sharing a plan.
     */
    public String cacheKey() {
        return cacheKey;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelDescriptor descriptor && descriptor.cacheKey.equals(cacheKey);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(cacheKey);
    }

    @Override
    public String toString() {
        return "KernelDescriptor[" + udfName + " " + inputs + " -> " + outputKind + ", class " + generatedClassName + "]";
    }
}
