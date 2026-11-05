/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.LookupSwitchInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TableSwitchInsnNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.SourceValue;

/**
 * Proves that every index a UDF uses on a {@code LIST} argument is inside that row, or refuses the
 * UDF.
 *
 * <p>
 * This is the difference between a kernel that reads a row and a kernel that reads whatever happens
 * to be next to it. A cuDF {@code LIST} column is offsets plus one flattened child shared by every
 * row, so {@code a.apply(i)} becomes {@code values.get(start + i)}. On the JVM an {@code i} past
 * the end of that row raises {@code ArrayIndexOutOfBoundsException}; in the flattened child it is
 * simply the next row's data, and far enough past the end it is memory outside the allocation. A
 * device kernel has no exception to raise, so the bound cannot be checked there and turned into the
 * failure Spark expects. It has to be established before the query is planned, or the UDF has to
 * keep its CPU plan.
 * </p>
 *
 * <p>
 * What counts as established here is deliberately narrow, because the cost of being wrong is
 * reading another row's data and the cost of being too strict is a CPU plan. An index qualifies
 * only when all of this holds:
 * </p>
 *
 * <ul>
 * <li>it is a single local variable, loaded directly, not an expression;</li>
 * <li>that local is compared against a bound belonging to the same list, by an {@code if_icmpge}
 *     that jumps away when the index has reached the bound -- which is what javac emits for
 *     {@code for (int i = 0; i &lt; a.length(); i++)};</li>
 * <li>the access is <em>dominated</em> by the in-bounds side of that comparison, so no path reaches
 *     it without the comparison having held;</li>
 * <li>the local starts at a non-negative constant and is only ever changed by a positive
 *     {@code iinc} that the access itself dominates, so its value at the access is the value the
 *     comparison accepted, and never negative.</li>
 * </ul>
 *
 * <p>
 * A bound "belongs to the same list" when it is that list's own length, or {@code Math.min} of
 * something that is, or a local holding one of those and assigned exactly once. The second case is
 * what makes the published cosine-similarity shape work: two lists, a shared
 * {@code Math.min(a.length(), b.length())} bound, and one loop indexing both.
 * </p>
 */
final class ListBoundsProof {

    private final MethodNode method;
    private final AbstractInsnNode[] instructions;
    private final Frame<SourceValue>[] frames;
    private final Map<AbstractInsnNode, Integer> indexOf = new java.util.IdentityHashMap<>();
    private final List<Set<Integer>> successors;
    private final int[] immediateDominator;

    ListBoundsProof(MethodNode method, Frame<SourceValue>[] frames) {
        this.method = method;
        this.frames = frames;
        this.instructions = method.instructions.toArray();
        for (int i = 0; i < instructions.length; i++) {
            indexOf.put(instructions[i], i);
        }
        this.successors = buildSuccessors();
        this.immediateDominator = computeDominators();
    }

    /**
     * Why {@code access} cannot be shown to stay inside row {@code listSlot}, or null when it can.
     *
     * @param access
     *         the indexing instruction ({@code xaload} or {@code Seq.apply}).
     * @param indexDepth
     *         how far below the top of the stack the index sits at {@code access}.
     */
    String whyUnproven(AbstractInsnNode access, int listSlot, int indexDepth) {
        Integer at = indexOf.get(access);
        if (at == null || frames[at] == null) {
            return "the indexing instruction could not be located in the method's control flow";
        }
        Frame<SourceValue> frame = frames[at];
        SourceValue indexValue = frame.getStack(frame.getStackSize() - 1 - indexDepth);
        if (indexValue.insns.size() != 1) {
            return "the index is computed on more than one path, so it cannot be tied to this row's length";
        }
        AbstractInsnNode indexSource = indexValue.insns.iterator().next();
        if (!(indexSource instanceof VarInsnNode load) || load.getOpcode() != Opcodes.ILOAD) {
            return "the index is an expression rather than a loop variable, and this analysis only proves a loop variable in range";
        }
        int indexSlot = load.var;

        for (int g = 0; g < instructions.length; g++) {
            if (!(instructions[g] instanceof JumpInsnNode guard) || guard.getOpcode() != Opcodes.IF_ICMPGE) {
                continue;
            }
            if (frames[g] == null || !comparesSlotAgainstLengthOf(g, indexSlot, listSlot)) {
                continue;
            }
            // The in-bounds side of `if (i >= bound) goto exit` is the fall-through.
            int inBounds = g + 1;
            if (inBounds >= instructions.length || !dominates(inBounds, at)) {
                continue;
            }
            String counterProblem = checkCounterDiscipline(indexSlot, at);
            if (counterProblem != null) {
                return counterProblem;
            }
            return null;
        }
        return "no check was found proving this index stays inside the row; an index past the end of a row "
                + "would read the next row's data instead of raising ArrayIndexOutOfBoundsException";
    }

    /** Whether the comparison at {@code g} is {@code iload indexSlot} against a bound of {@code listSlot}. */
    private boolean comparesSlotAgainstLengthOf(int g, int indexSlot, int listSlot) {
        Frame<SourceValue> frame = frames[g];
        if (frame.getStackSize() < 2) {
            return false;
        }
        SourceValue left = frame.getStack(frame.getStackSize() - 2);
        SourceValue right = frame.getStack(frame.getStackSize() - 1);
        if (left.insns.size() != 1) {
            return false;
        }
        AbstractInsnNode leftSource = left.insns.iterator().next();
        if (!(leftSource instanceof VarInsnNode load) || load.getOpcode() != Opcodes.ILOAD || load.var != indexSlot) {
            return false;
        }
        return isBoundOf(right, listSlot, new HashSet<>(), 0);
    }

    /** Whether {@code value} is this list's own length, a {@code Math.min} containing one, or a local holding one. */
    private boolean isBoundOf(SourceValue value, int listSlot, Set<AbstractInsnNode> seen, int depth) {
        if (depth > 6 || value.insns.size() != 1) {
            return false;
        }
        AbstractInsnNode source = value.insns.iterator().next();
        if (!seen.add(source)) {
            return false;
        }
        Integer at = indexOf.get(source);
        if (at == null || frames[at] == null) {
            return false;
        }
        int op = source.getOpcode();
        if (op == Opcodes.ARRAYLENGTH) {
            return readsList(at, 0, listSlot);
        }
        if (op == Opcodes.INVOKEINTERFACE && source instanceof MethodInsnNode call
                && (call.name.equals("length") || call.name.equals("size")) && call.desc.equals("()I")) {
            return readsList(at, 0, listSlot);
        }
        if (op == Opcodes.INVOKESTATIC && source instanceof MethodInsnNode call
                && (call.owner.equals("java/lang/Math") || call.owner.equals("java/lang/StrictMath"))
                && call.name.equals("min") && call.desc.equals("(II)I")) {
            // min(x, y) <= x and <= y, so either operand being this list's length is enough.
            Frame<SourceValue> frame = frames[at];
            return isBoundOf(frame.getStack(frame.getStackSize() - 2), listSlot, seen, depth + 1)
                    || isBoundOf(frame.getStack(frame.getStackSize() - 1), listSlot, seen, depth + 1);
        }
        if (op == Opcodes.ILOAD && source instanceof VarInsnNode load) {
            // A local holding a bound, provided it is assigned exactly once in the method.
            List<AbstractInsnNode> writes = writesTo(load.var);
            if (writes.size() != 1) {
                return false;
            }
            AbstractInsnNode write = writes.get(0);
            if (write.getOpcode() != Opcodes.ISTORE) {
                return false;
            }
            Integer writeAt = indexOf.get(write);
            if (writeAt == null || frames[writeAt] == null) {
                return false;
            }
            Frame<SourceValue> frame = frames[writeAt];
            return isBoundOf(frame.getStack(frame.getStackSize() - 1), listSlot, seen, depth + 1);
        }
        return false;
    }

    /** Whether the value {@code depth} below the top at {@code at} came from loading {@code listSlot}. */
    private boolean readsList(int at, int depth, int listSlot) {
        Frame<SourceValue> frame = frames[at];
        int position = frame.getStackSize() - 1 - depth;
        if (position < 0) {
            return false;
        }
        SourceValue value = frame.getStack(position);
        if (value.insns.size() != 1) {
            return false;
        }
        AbstractInsnNode source = value.insns.iterator().next();
        return source instanceof VarInsnNode load && load.getOpcode() == Opcodes.ALOAD && load.var == listSlot;
    }

    /**
     * The loop variable must start non-negative and only ever move up, after the access. Otherwise
     * the value the comparison accepted is not the value the access uses.
     */
    private String checkCounterDiscipline(int indexSlot, int accessAt) {
        for (AbstractInsnNode write : writesTo(indexSlot)) {
            Integer writeAt = indexOf.get(write);
            if (writeAt == null) {
                continue;
            }
            if (write instanceof IincInsnNode increment) {
                if (increment.incr <= 0) {
                    return "the loop variable can decrease, so a checked index does not stay checked";
                }
                if (!dominates(accessAt, writeAt)) {
                    return "the loop variable is advanced on a path that reaches the access, so the checked value is not the one used";
                }
                continue;
            }
            // A plain store: only the initialisation, and only of a non-negative constant.
            Frame<SourceValue> frame = frames[writeAt];
            if (frame == null) {
                continue;
            }
            SourceValue stored = frame.getStack(frame.getStackSize() - 1);
            if (stored.insns.size() != 1) {
                return "the loop variable is assigned a value this analysis cannot bound below zero";
            }
            AbstractInsnNode source = stored.insns.iterator().next();
            int op = source.getOpcode();
            boolean nonNegativeConstant = (op >= Opcodes.ICONST_0 && op <= Opcodes.ICONST_5)
                    || (op == Opcodes.BIPUSH && ((org.objectweb.asm.tree.IntInsnNode) source).operand >= 0)
                    || (op == Opcodes.SIPUSH && ((org.objectweb.asm.tree.IntInsnNode) source).operand >= 0);
            if (!nonNegativeConstant) {
                return "the loop variable is assigned a value this analysis cannot bound below zero; a negative index "
                        + "would read before the start of the row";
            }
        }
        return null;
    }

    private List<AbstractInsnNode> writesTo(int slot) {
        List<AbstractInsnNode> writes = new ArrayList<>();
        for (AbstractInsnNode insn : instructions) {
            if (insn instanceof VarInsnNode var && var.var == slot && var.getOpcode() == Opcodes.ISTORE) {
                writes.add(insn);
            } else if (insn instanceof IincInsnNode increment && increment.var == slot) {
                writes.add(insn);
            }
        }
        return writes;
    }

    // ---------------------------------------------------------------- control flow

    private List<Set<Integer>> buildSuccessors() {
        List<Set<Integer>> out = new ArrayList<>();
        for (int i = 0; i < instructions.length; i++) {
            out.add(new HashSet<>());
        }
        for (int i = 0; i < instructions.length; i++) {
            AbstractInsnNode insn = instructions[i];
            int op = insn.getOpcode();
            boolean fallsThrough = true;
            if (insn instanceof JumpInsnNode jump) {
                out.get(i).add(indexOf.get(jump.label));
                fallsThrough = op != Opcodes.GOTO;
            } else if (insn instanceof TableSwitchInsnNode table) {
                out.get(i).add(indexOf.get(table.dflt));
                for (LabelNode label : table.labels) {
                    out.get(i).add(indexOf.get(label));
                }
                fallsThrough = false;
            } else if (insn instanceof LookupSwitchInsnNode lookup) {
                out.get(i).add(indexOf.get(lookup.dflt));
                for (LabelNode label : lookup.labels) {
                    out.get(i).add(indexOf.get(label));
                }
                fallsThrough = false;
            } else if (op >= Opcodes.IRETURN && op <= Opcodes.RETURN || op == Opcodes.ATHROW) {
                fallsThrough = false;
            }
            if (fallsThrough && i + 1 < instructions.length) {
                out.get(i).add(i + 1);
            }
        }
        return out;
    }

    /** Iterative dominator computation; methods here are small enough that simplicity wins. */
    private int[] computeDominators() {
        int n = instructions.length;
        int[] idom = new int[n];
        Arrays.fill(idom, -1);
        if (n == 0) {
            return idom;
        }
        List<Set<Integer>> predecessors = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            predecessors.add(new HashSet<>());
        }
        for (int i = 0; i < n; i++) {
            for (int s : successors.get(i)) {
                predecessors.get(s).add(i);
            }
        }
        List<Integer> order = reversePostorder();
        int[] position = new int[n];
        Arrays.fill(position, -1);
        for (int i = 0; i < order.size(); i++) {
            position[order.get(i)] = i;
        }
        idom[0] = 0;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int node : order) {
                if (node == 0) {
                    continue;
                }
                int newIdom = -1;
                for (int predecessor : predecessors.get(node)) {
                    if (idom[predecessor] == -1 || position[predecessor] < 0) {
                        continue;
                    }
                    newIdom = newIdom == -1 ? predecessor : intersect(predecessor, newIdom, idom, position);
                }
                if (newIdom != -1 && idom[node] != newIdom) {
                    idom[node] = newIdom;
                    changed = true;
                }
            }
        }
        return idom;
    }

    private List<Integer> reversePostorder() {
        List<Integer> postorder = new ArrayList<>();
        boolean[] visited = new boolean[instructions.length];
        java.util.Deque<int[]> stack = new java.util.ArrayDeque<>();
        stack.push(new int[] { 0, 0 });
        visited[0] = true;
        while (!stack.isEmpty()) {
            int[] top = stack.peek();
            List<Integer> next = new ArrayList<>(successors.get(top[0]));
            if (top[1] < next.size()) {
                int child = next.get(top[1]++);
                if (!visited[child]) {
                    visited[child] = true;
                    stack.push(new int[] { child, 0 });
                }
            } else {
                postorder.add(stack.pop()[0]);
            }
        }
        java.util.Collections.reverse(postorder);
        return postorder;
    }

    private static int intersect(int a, int b, int[] idom, int[] position) {
        while (a != b) {
            while (position[a] > position[b]) {
                if (idom[a] == a || idom[a] == -1) {
                    return b;
                }
                a = idom[a];
            }
            while (position[b] > position[a]) {
                if (idom[b] == b || idom[b] == -1) {
                    return a;
                }
                b = idom[b];
            }
        }
        return a;
    }

    /** Whether every path from the method's entry to {@code target} passes through {@code source}. */
    private boolean dominates(int source, int target) {
        if (source == target) {
            return true;
        }
        int current = target;
        int guard = 0;
        while (current != 0 && guard++ < instructions.length + 2) {
            int next = immediateDominator[current];
            if (next == -1 || next == current) {
                return false;
            }
            if (next == source) {
                return true;
            }
            current = next;
        }
        return source == 0;
    }
}
