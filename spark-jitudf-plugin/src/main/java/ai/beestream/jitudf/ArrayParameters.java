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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.tree.analysis.Analyzer;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.SourceInterpreter;
import org.objectweb.asm.tree.analysis.SourceValue;

/**
 * Works out which array reads in a method are reads of one of its {@code LIST} parameters, and
 * refuses any other use of those parameters.
 *
 * <p>
 * The question matters because a {@code LIST} parameter does not survive as an array. It becomes
 * two device views -- the offsets and the flattened child -- and {@code arr.length} and
 * {@code arr[i]} become arithmetic on those. So only uses that have an answer in that layout can be
 * allowed, and the way to know whether a given {@code faload} is reading parameter 0 or something
 * else is a dataflow question, not a question about the shape of the nearby bytecode.
 * </p>
 *
 * <p>
 * It is answered with ASM's {@link SourceInterpreter}, which tracks for every stack value the set of
 * instructions that could have produced it. An access qualifies when the reference it consumes has
 * exactly one source and that source is an {@code aload} of a known {@code LIST} parameter slot.
 * Anything else about those parameters -- storing one into a local, passing one to a helper,
 * returning one, comparing one to null -- makes the whole method ineligible, because the value being
 * moved around has no representation on the device.
 * </p>
 *
 * <p>
 * Two spellings of the same thing qualify, because Spark uses the second. A UDF declared over a
 * {@code double[]} reads it with {@code arraylength} and {@code daload}. A UDF declared over a
 * {@code scala.collection.Seq}, which is what Spark actually hands a Java UDF for an
 * {@code ArrayType} column, reads it with {@code Seq.length()} and {@code Seq.apply(int)} followed
 * by a cast and an unboxing. Both are reads of one element of one row's list, and both lower to the
 * same indexing of the flattened child. For the second, the cast that follows {@code apply} is
 * checked against the column's element type, so a UDF that casts to {@code Integer} over a
 * {@code double} column is refused rather than compiled into a kernel that reinterprets the bits.
 * </p>
 */
final class ArrayParameters {

    private static final ArrayParameters NONE = new ArrayParameters(Map.of());

    /** Local-variable slot of each list parameter, mapped to its element kind. */
    private final Map<Integer, PrimitiveKind> listSlots;

    /** Array accesses established to read a list parameter; filled in by {@link #check}. */
    private final Set<AbstractInsnNode> qualified = Collections.newSetFromMap(new java.util.IdentityHashMap<>());

    /** For each qualified access, the slot it reads. */
    private final Map<AbstractInsnNode, Integer> slotOf = new java.util.IdentityHashMap<>();

    private ArrayParameters(Map<Integer, PrimitiveKind> listSlots) {
        this.listSlots = listSlots;
    }

    static ArrayParameters none() {
        return NONE;
    }

    /** The list parameters of {@code method}, given what each of the UDF's arguments is. */
    static ArrayParameters of(MethodNode method, List<InputSpec> inputs) {
        Map<Integer, PrimitiveKind> slots = new HashMap<>();
        int slot = (method.access & Opcodes.ACC_STATIC) != 0 ? 0 : 1;
        Type[] parameters = Type.getArgumentTypes(method.desc);
        for (int i = 0; i < parameters.length && i < inputs.size(); i++) {
            if (inputs.get(i).isList()) {
                slots.put(slot, inputs.get(i).kind());
            }
            slot += parameters[i].getSize();
        }
        return slots.isEmpty() ? NONE : new ArrayParameters(slots);
    }

    Map<Integer, PrimitiveKind> listSlots() {
        return listSlots;
    }

    /**
     * Runs the analysis over {@code method} and returns the first problem, or {@code null}. After a
     * {@code null} return, {@link #readsAListParameter} answers for every array access in it.
     */
    /** Built lazily, once per {@link #check}, because it computes a dominator tree. */
    private ListBoundsProof bounds;

    private ListBoundsProof bounds(MethodNode method, Frame<SourceValue>[] frames) {
        if (bounds == null) {
            bounds = new ListBoundsProof(method, frames);
        }
        return bounds;
    }

    String check(MethodNode method) {
        bounds = null;
        if (listSlots.isEmpty()) {
            return null;
        }
        Frame<SourceValue>[] frames;
        try {
            frames = new Analyzer<>(new SourceInterpreter()).analyze("udf", method);
        } catch (AnalyzerException e) {
            return "the UDF's bytecode could not be analysed for how it uses its array arguments: " + e.getMessage();
        }
        AbstractInsnNode[] instructions = method.instructions.toArray();
        for (int i = 0; i < instructions.length; i++) {
            AbstractInsnNode insn = instructions[i];
            Frame<SourceValue> frame = frames[i];
            if (frame == null) {
                continue; // unreachable
            }
            int op = insn.getOpcode();
            if (op == Opcodes.ARRAYLENGTH) {
                Integer slot = singleListSource(frame, 0);
                if (slot == null) {
                    return "the UDF takes the length of an array that is not one of its LIST arguments";
                }
                qualified.add(insn);
                slotOf.put(insn, slot);
            } else if (op == Opcodes.IALOAD || op == Opcodes.LALOAD || op == Opcodes.FALOAD || op == Opcodes.DALOAD) {
                // Stack is [arrayref, index]: the reference is one below the top.
                Integer slot = singleListSource(frame, 1);
                if (slot == null) {
                    return "the UDF indexes an array that is not one of its LIST arguments";
                }
                if (listSlots.get(slot).arrayLoadOpcode() != op) {
                    return "the UDF reads its LIST argument with the wrong element type";
                }
                String unproven = bounds(method, frames).whyUnproven(insn, slot, 0);
                if (unproven != null) {
                    return "this index into a LIST argument is not provably inside its row: " + unproven;
                }
                qualified.add(insn);
                slotOf.put(insn, slot);
            } else if (op == Opcodes.INVOKEINTERFACE && insn instanceof MethodInsnNode call && isSeqAccessor(call)) {
                boolean isApply = call.name.equals("apply");
                Integer slot = singleListSource(frame, isApply ? 1 : 0);
                if (slot == null) {
                    return "the UDF calls " + call.name + " on a sequence that is not one of its LIST arguments";
                }
                if (isApply) {
                    String elementProblem = checkUnboxing(insn, listSlots.get(slot));
                    if (elementProblem != null) {
                        return elementProblem;
                    }
                    String unproven = bounds(method, frames).whyUnproven(insn, slot, 0);
                    if (unproven != null) {
                        return "this index into a LIST argument is not provably inside its row: " + unproven;
                    }
                }
                qualified.add(insn);
                slotOf.put(insn, slot);
            } else if (insn instanceof VarInsnNode variable && listSlots.containsKey(variable.var) && variable.getOpcode() == Opcodes.ASTORE) {
                return "the UDF stores a LIST argument into another local, which has no meaning once it is offsets and values";
            }
        }
        String misuse = checkNoOtherUse(method, frames, instructions);
        return misuse;
    }

    /**
     * Every {@code aload} of a list parameter must be consumed by one of the accesses above and by
     * nothing else: passed to a method, returned, or compared, the reference would have to exist,
     * and it does not.
     */
    private String checkNoOtherUse(MethodNode method, Frame<SourceValue>[] frames, AbstractInsnNode[] instructions) {
        Set<AbstractInsnNode> consumedByAccess = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        for (int i = 0; i < instructions.length; i++) {
            AbstractInsnNode insn = instructions[i];
            if (!qualified.contains(insn)) {
                continue;
            }
            Frame<SourceValue> frame = frames[i];
            int depth = receiverDepth(insn);
            consumedByAccess.addAll(frame.getStack(frame.getStackSize() - 1 - depth).insns);
        }
        List<AbstractInsnNode> loads = new ArrayList<>();
        for (AbstractInsnNode insn : instructions) {
            if (insn instanceof VarInsnNode variable && variable.getOpcode() == Opcodes.ALOAD && listSlots.containsKey(variable.var)) {
                loads.add(insn);
            }
        }
        for (AbstractInsnNode load : loads) {
            if (!consumedByAccess.contains(load)) {
                return "the UDF uses a LIST argument as a value rather than indexing it, which has no device representation";
            }
        }
        return null;
    }

    /** How far below the top of the stack the sequence or array reference sits for this access. */
    private static int receiverDepth(AbstractInsnNode insn) {
        if (insn.getOpcode() == Opcodes.ARRAYLENGTH) {
            return 0;
        }
        if (insn instanceof MethodInsnNode call) {
            return call.name.equals("apply") ? 1 : 0;
        }
        return 1; // xALOAD: [arrayref, index]
    }

    /** {@code length()}, {@code size()} or {@code apply(int)} on a Scala collection. */
    private static boolean isSeqAccessor(MethodInsnNode call) {
        if (!call.owner.startsWith("scala/collection/")) {
            return false;
        }
        return (call.name.equals("length") || call.name.equals("size")) && call.desc.equals("()I") //
                || call.name.equals("apply") && call.desc.equals("(I)Ljava/lang/Object;");
    }

    /**
     * {@code apply} returns a boxed element, which the UDF then casts and unboxes. Those two
     * instructions say which type the UDF believes the elements are, so they are checked against
     * the column's: a mismatch would otherwise become a kernel reading the wrong width.
     */
    private String checkUnboxing(AbstractInsnNode apply, PrimitiveKind element) {
        AbstractInsnNode cast = nextReal(apply);
        AbstractInsnNode unbox = cast == null ? null : nextReal(cast);
        String box = element.boxedClass().getName().replace('.', '/');
        if (!(cast instanceof org.objectweb.asm.tree.TypeInsnNode typeInsn) || typeInsn.getOpcode() != Opcodes.CHECKCAST || !typeInsn.desc.equals(box)) {
            return "the UDF does not cast the element it reads from its LIST argument to " + element.boxedClass().getSimpleName();
        }
        if (!(unbox instanceof MethodInsnNode unboxCall) || !unboxCall.owner.equals(box) || !unboxCall.name.endsWith("Value") || !unboxCall.desc.equals("()" + element.descriptor())) {
            return "the UDF does not unbox the element it reads from its LIST argument to " + element.primitiveClass().getSimpleName();
        }
        return null;
    }

    private static AbstractInsnNode nextReal(AbstractInsnNode insn) {
        AbstractInsnNode next = insn.getNext();
        while (next != null && next.getOpcode() < 0) {
            next = next.getNext();
        }
        return next;
    }

    /** The list-parameter slot behind the stack value {@code depth} below the top, or null. */
    private Integer singleListSource(Frame<SourceValue> frame, int depth) {
        int index = frame.getStackSize() - 1 - depth;
        if (index < 0) {
            return null;
        }
        SourceValue value = frame.getStack(index);
        if (value.insns.size() != 1) {
            return null;
        }
        AbstractInsnNode source = value.insns.iterator().next();
        if (source instanceof VarInsnNode variable && variable.getOpcode() == Opcodes.ALOAD && listSlots.containsKey(variable.var)) {
            return variable.var;
        }
        return null;
    }

    boolean readsAListParameter(AbstractInsnNode insn) {
        return qualified.contains(insn);
    }

    /** Which list-parameter slot {@code insn} reads; only meaningful for a qualified access. */
    int slotFor(AbstractInsnNode insn) {
        Integer slot = slotOf.get(insn);
        if (slot == null) {
            throw new IllegalStateException("this instruction was not established to read a LIST argument");
        }
        return slot;
    }
}
