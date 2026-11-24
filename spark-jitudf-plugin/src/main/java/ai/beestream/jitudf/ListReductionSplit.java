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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Splits a UDF that sums over a list into a per-element function and a reduction.
 *
 * <p>
 * This exists because of a gap that looked permanent and is not. {@code cudf::transform} refuses any
 * input that is not fixed-width or a string, so a {@code LIST} column cannot be handed to a per-row
 * UDF -- which is why a segmented transform is NVIDIA's deferred opportunity #5, and why this
 * architecture appeared unable to run the one customer pattern their survey called a near-fit.
 * </p>
 *
 * <p>
 * But a reduction over a list does not need a per-row kernel at all. A list column is offsets plus a
 * flattened child, and the child <em>is</em> a fixed-width column, so the element function can be
 * applied to the whole child in one ordinary {@code transform}, and the per-row answer recovered by
 * a segmented reduction over the same offsets. Both halves already exist in the shipped cuDF Java
 * API. Nothing upstream has to change.
 * </p>
 *
 * <p>
 * What this class does is find the two halves in the user's bytecode. It matches one shape, the one
 * the pattern is actually written in:
 * </p>
 *
 * <pre>{@code
 * acc = 0;
 * for (int i = 0; i < arr.length; i++) { acc += g(arr[i]); }
 * return acc;
 * }</pre>
 *
 * <p>
 * and returns {@code g} as a method over one element. Anything else is declined, and the UDF is
 * refused as it was before: this widens the supported set, it does not guess.
 * </p>
 */
public final class ListReductionSplit {

    private ListReductionSplit() {
    }

    /** The element function and the reduction that combines its results. */
    public record Split(MethodNode elementMethod, PrimitiveKind elementKind, PrimitiveKind resultKind) {
    }

    /**
     * Tries to split {@code body}, the lowered {@code udfBody(values, start, end)}.
     *
     * @param body
     *         the lowered body, whose shape {@link AsmUdfLowering} fixes.
     * @param elementKind
     *         the list's element type.
     * @param resultKind
     *         what the UDF returns, which is also the reduction's type.
     */
    public static Optional<Split> split(MethodNode body, PrimitiveKind elementKind, PrimitiveKind resultKind) {
        // Only the single-list, no-scalar signature: (values, start, end).
        if (!body.desc.startsWith("(L" + elementKind.arrayInternalName() + ";II)")) {
            return Optional.empty();
        }
        List<AbstractInsnNode> code = new ArrayList<>();
        for (AbstractInsnNode insn : body.instructions) {
            if (insn.getOpcode() >= 0) {
                code.add(insn);
            }
        }

        // The element read, as AsmUdfLowering emits it: the values array, the row's start offset and
        // the loop index added together, then a get. The two ILOADs may be in either order, so this
        // requires only that one of them is the start parameter.
        int readEnd = -1;
        for (int i = 0; i + 4 < code.size(); i++) {
            if (code.get(i).getOpcode() == Opcodes.ALOAD && ((VarInsnNode) code.get(i)).var == 0
                    && code.get(i + 1).getOpcode() == Opcodes.ILOAD
                    && code.get(i + 2).getOpcode() == Opcodes.ILOAD
                    && (((VarInsnNode) code.get(i + 1)).var == 1 || ((VarInsnNode) code.get(i + 2)).var == 1)
                    && code.get(i + 3).getOpcode() == Opcodes.IADD
                    && code.get(i + 4) instanceof MethodInsnNode call && call.name.equals("get")
                    && call.owner.equals(elementKind.arrayInternalName())) {
                if (readEnd >= 0) {
                    // Two element reads: not a simple map over the list.
                    return Optional.empty();
                }
                readEnd = i + 4;
            }
        }
        if (readEnd < 0) {
            return Optional.empty();
        }

        // The accumulate: the reduction's add, immediately followed by a store of the accumulator.
        int addOpcode = addOpcodeFor(resultKind);
        int accumulate = -1;
        for (int i = readEnd + 1; i + 1 < code.size(); i++) {
            if (code.get(i).getOpcode() == addOpcode && code.get(i + 1).getOpcode() == storeOpcodeFor(resultKind)) {
                accumulate = i;
                break;
            }
        }
        if (accumulate < 0) {
            return Optional.empty();
        }
        int accumulator = ((VarInsnNode) code.get(accumulate + 1)).var;

        // The accumulator must be loaded immediately before the element read, so that the loop body
        // is exactly `acc = acc + g(element)` and nothing else touches the running total.
        int beforeRead = readEnd - 4;
        if (beforeRead <= 0 || code.get(beforeRead - 1).getOpcode() != loadOpcodeFor(resultKind)
                || ((VarInsnNode) code.get(beforeRead - 1)).var != accumulator) {
            return Optional.empty();
        }
        // ... and nowhere else, or the result depends on more than a sum of the elements.
        for (int i = 0; i < code.size(); i++) {
            AbstractInsnNode insn = code.get(i);
            boolean touches = (insn instanceof VarInsnNode var && var.var == accumulator)
                    || (insn instanceof IincInsnNode inc && inc.var == accumulator);
            if (!touches) {
                continue;
            }
            boolean expected = i == beforeRead - 1 || i == accumulate + 1 || isInitialisation(code, i, accumulator, resultKind) || isReturnLoad(code, i);
            if (!expected) {
                return Optional.empty();
            }
        }

        // The element function is what sits between the read and the accumulate.
        MethodNode element = new MethodNode(Opcodes.ASM9, Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, "udfElement",
                "(" + elementKind.descriptor() + ")" + resultKind.descriptor(), null, null);
        InsnList out = new InsnList();
        out.add(new VarInsnNode(loadOpcodeFor(elementKind), 0));
        for (int i = readEnd + 1; i < accumulate; i++) {
            AbstractInsnNode copy = code.get(i).clone(java.util.Map.of());
            if (copy instanceof VarInsnNode || copy instanceof IincInsnNode) {
                // The element expression must not reach any local: it sees one value.
                return Optional.empty();
            }
            out.add(copy);
        }
        out.add(new InsnNode(returnOpcodeFor(resultKind)));
        element.instructions = out;
        element.maxStack = body.maxStack + 4;
        element.maxLocals = elementKind.slots() + 4;
        return Optional.of(new Split(element, elementKind, resultKind));
    }

    /** Whether instruction {@code i} is the accumulator's initialisation to the sum identity, zero. */
    private static boolean isInitialisation(List<AbstractInsnNode> code, int i, int accumulator, PrimitiveKind kind) {
        if (code.get(i).getOpcode() != storeOpcodeFor(kind) || ((VarInsnNode) code.get(i)).var != accumulator || i == 0) {
            return false;
        }
        int before = code.get(i - 1).getOpcode();
        return before == Opcodes.ICONST_0 || before == Opcodes.LCONST_0 || before == Opcodes.FCONST_0 || before == Opcodes.DCONST_0;
    }

    /** Whether instruction {@code i} is the final load of the accumulator, feeding the return. */
    private static boolean isReturnLoad(List<AbstractInsnNode> code, int i) {
        return i + 1 < code.size() && isReturn(code.get(i + 1).getOpcode());
    }

    private static boolean isReturn(int opcode) {
        return opcode == Opcodes.IRETURN || opcode == Opcodes.LRETURN || opcode == Opcodes.FRETURN || opcode == Opcodes.DRETURN;
    }

    private static int addOpcodeFor(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.IADD;
            case LONG -> Opcodes.LADD;
            case FLOAT -> Opcodes.FADD;
            case DOUBLE -> Opcodes.DADD;
            // A fold that sums booleans has no meaning; the split refuses such a UDF earlier.
            case BOOL -> throw new IllegalStateException("a list reduction cannot sum booleans");
        };
    }

    private static int loadOpcodeFor(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.ILOAD;
            case LONG -> Opcodes.LLOAD;
            case FLOAT -> Opcodes.FLOAD;
            case DOUBLE -> Opcodes.DLOAD;
            case BOOL -> throw new IllegalStateException("a list reduction cannot sum booleans");
        };
    }

    private static int storeOpcodeFor(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.ISTORE;
            case LONG -> Opcodes.LSTORE;
            case FLOAT -> Opcodes.FSTORE;
            case DOUBLE -> Opcodes.DSTORE;
            case BOOL -> throw new IllegalStateException("a list reduction cannot sum booleans");
        };
    }

    private static int returnOpcodeFor(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.IRETURN;
            case LONG -> Opcodes.LRETURN;
            case FLOAT -> Opcodes.FRETURN;
            case DOUBLE -> Opcodes.DRETURN;
            case BOOL -> throw new IllegalStateException("a list reduction cannot sum booleans");
        };
    }
}
