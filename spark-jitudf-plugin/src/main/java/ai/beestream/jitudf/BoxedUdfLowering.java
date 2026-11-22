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

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FrameNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Makes the user's UDF method static without changing anything else about it.
 *
 * <p>
 * This is the counterpart of {@link AsmUdfLowering} for a nullable column, and the difference is
 * the point. That lowering erases the boxing: it turns {@code Integer call(Long v)} into
 * {@code int udfBody(long v)}, which is the only shape a kernel over non-null values needs, and in
 * doing so it throws away the one thing a nullable column cares about -- that {@code v} can be
 * {@code null} and that the method may return {@code null}. Here the boxes stay, because they are
 * the nulls: TC's null-aware mode reads {@code Long} as a SQL value that may be absent and compiles
 * the method into {@code __device__ int f(cuda::std::optional<int> *out, cuda::std::optional<long
 * long> v_opt)}, which is what {@code cudf::transform} calls with {@code null_aware::YES}.
 * </p>
 *
 * <p>
 * So there is exactly one thing to do: the method is an instance method, and a device function has
 * no receiver. Slot 0 goes and every other slot moves down one. Widths do not change, because
 * nothing else about the frame changes, so the shift is uniform -- a {@code long} local in slots
 * 5 and 6 lands in 4 and 5 and is still two slots wide.
 * </p>
 *
 * <p>
 * Reaching slot 0 at all would mean the method still reads a field of its instance.
 * {@link ConstantFieldFolding} runs before any of this and replaces a final field's read with its
 * value, and {@link UdfEligibilityAnalyzer} refuses what is left, so a {@code this} here is a
 * disagreement with one of those rather than something to compile around: it fails loudly.
 * </p>
 */
final class BoxedUdfLowering {

    private BoxedUdfLowering() {
    }

    /**
     * {@code source} as a static method called {@code newName}, with the same descriptor.
     *
     * @param isStatic
     *         whether {@code source} is already static, in which case there is no receiver to drop
     *         and the slots do not move.
     */
    static MethodNode lower(MethodNode source, String newName, boolean isStatic) {
        int shift = isStatic ? 0 : 1;

        MethodNode lowered = new MethodNode(Opcodes.ASM9, Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, newName, source.desc, null, null);
        InsnList out = new InsnList();
        for (AbstractInsnNode insn : source.instructions) {
            if (insn.getOpcode() < 0) {
                // Labels and line numbers are kept; frames are recomputed when the class is written,
                // because the frame's local list is one entry shorter than the original's.
                if (!(insn instanceof FrameNode)) {
                    out.add(insn);
                }
                continue;
            }
            if (insn instanceof VarInsnNode variable) {
                out.add(new VarInsnNode(variable.getOpcode(), shifted(variable.var, shift)));
            } else if (insn instanceof IincInsnNode increment) {
                out.add(new IincInsnNode(shifted(increment.var, shift), increment.incr));
            } else {
                out.add(insn);
            }
        }
        lowered.instructions = out;
        lowered.maxStack = source.maxStack;
        lowered.maxLocals = Math.max(0, source.maxLocals - shift);
        return lowered;
    }

    /** The descriptor TC is asked to compile: the user's own, boxes and all. */
    static String descriptorOf(MethodNode source) {
        return source.desc;
    }

    /** The number of parameters the device function will take, which is one cudf input each. */
    static int parameterCount(MethodNode source) {
        return Type.getArgumentTypes(source.desc).length;
    }

    private static int shifted(int slot, int shift) {
        int moved = slot - shift;
        if (moved < 0) {
            throw new IllegalStateException(
                    "the UDF still reaches its receiver in local slot " + slot + "; constant folding and the eligibility analysis should have removed every use of it");
        }
        return moved;
    }
}
