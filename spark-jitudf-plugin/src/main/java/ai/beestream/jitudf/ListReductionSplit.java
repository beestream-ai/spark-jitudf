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
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.LookupSwitchInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TableSwitchInsnNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Splits a UDF that sums over a list into a per-element function and a segmented reduction -- but
 * only after proving that the two compute the same thing.
 *
 * <p>
 * {@code cudf::transform} refuses a {@code LIST} input, but a reduction over a list does not need a
 * per-row kernel: the element function can be applied to the whole flattened child in one ordinary
 * {@code transform}, and the per-row answer recovered by a segmented sum over the same offsets
 * ({@link CudfListReductionUdf}).
 * </p>
 *
 * <p>
 * <strong>What has to be proved.</strong> That executor applies the element function to
 * <em>every</em> element of <em>every</em> row and adds all the results, starting from zero. So the
 * Java loop must do exactly that: visit each index of the row once, in any order, unconditionally
 * add one value that depends on that element alone, and return the total. Finding "an element read
 * and an add" is not enough -- a loop with {@code i += 2} has both and skips half the elements; so
 * does one with {@code if (x < 0) continue;} or an early {@code return}. Safe indexing
 * ({@link ListBoundsProof}) is a different and weaker property: a loop can index safely and still
 * visit the wrong elements.
 * </p>
 *
 * <p>
 * The recognizer therefore matches the complete method, on the lowered body
 * {@code udfBody(values, start, end)}, against the canonical fold:
 * </p>
 *
 * <pre>{@code
 * acc = 0;                       // the sum's identity, and nothing else before the loop
 * i = 0;                         // ... apart from n = length, if the bound is kept in a local
 * H: if (i >= length) goto X;    // the only exit, with the row's own length as the bound
 *    acc = acc + E(arr[i]);      // exactly one update, unconditional, of the accumulator only
 *    i += 1;                     // stride one, the only other write to i
 *    goto H;                     // the only backward jump
 * X: return acc;                 // nothing after the loop
 * }</pre>
 *
 * <p>
 * Inside {@code E}: any number of reads of {@code arr[i]} (all of the same element), constants,
 * arithmetic, calls the analysis admitted, forward branches that stay inside {@code E}, and
 * temporaries that live only inside {@code E}. No other use of {@code i}, the accumulator or the
 * list, no return, no jump out of {@code E} and none into it. Then {@code sum_i E(arr[i])} is the
 * Java result for every row, and it is exactly what the executor computes: integer addition wraps
 * in both and is associative and commutative modulo 2^n, so the order of the segmented sum does not
 * matter. Floating-point addition is not associative, which is why a floating-point fold is
 * additionally gated on RAPIDS's own setting for order-dependent float aggregation (see
 * {@link TornadoUdfProvider}).
 * </p>
 *
 * <p>
 * Every other loop is declined with the specific reason, and the UDF keeps its CPU plan.
 * </p>
 */
public final class ListReductionSplit {

    private ListReductionSplit() {
    }

    /** The element function and the reduction that combines its results. */
    public record Split(MethodNode elementMethod, PrimitiveKind elementKind, PrimitiveKind resultKind) {
    }

    /** Why the most recent {@link #split} on this thread declined, for the refusal message. */
    private static final ThreadLocal<String> LAST_REASON = new ThreadLocal<>();

    public static String lastReason() {
        String reason = LAST_REASON.get();
        return reason == null ? "no reason recorded" : reason;
    }

    private static Optional<Split> decline(String reason) {
        LAST_REASON.set(reason);
        return Optional.empty();
    }

    private static final int VALUES = 0;
    private static final int START = 1;
    private static final int END = 2;

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
        LAST_REASON.remove();
        if (resultKind == PrimitiveKind.BOOL) {
            return decline("a boolean result is not a sum");
        }
        if (!body.desc.equals("(L" + elementKind.arrayInternalName() + ";II)" + resultKind.descriptor())) {
            return decline("only a UDF over a single list argument and nothing else can become a segmented reduction");
        }
        List<AbstractInsnNode> all = new ArrayList<>();
        for (AbstractInsnNode insn : body.instructions) {
            if (insn.getOpcode() >= 0 || insn instanceof LabelNode) {
                all.add(insn);
            }
        }
        Map<AbstractInsnNode, Integer> at = new IdentityHashMap<>();
        for (int i = 0; i < all.size(); i++) {
            at.put(all.get(i), i);
        }

        // ---- the one loop: exactly one backward jump, a GOTO to the header.
        int backJump = -1;
        for (int i = 0; i < all.size(); i++) {
            for (LabelNode target : targetsOf(all.get(i))) {
                if (at.get(target) <= i) {
                    if (backJump >= 0 || all.get(i).getOpcode() != Opcodes.GOTO) {
                        return decline("the UDF does not have exactly one loop with its back edge at the end of the body");
                    }
                    backJump = i;
                }
            }
        }
        if (backJump < 0) {
            return decline("the UDF has no loop over the list");
        }
        int header = at.get(((JumpInsnNode) all.get(backJump)).label);

        // ---- the header: iload i; <length>; if_icmpge X
        List<Integer> headerCode = realFrom(all, header, 8);
        if (headerCode.size() < 3 || !(all.get(headerCode.get(0)) instanceof VarInsnNode indexLoad) || indexLoad.getOpcode() != Opcodes.ILOAD) {
            return decline("the loop does not start by comparing an index variable with a bound");
        }
        int index = indexLoad.var;
        int afterBound = matchLength(all, headerCode, 1);
        Integer lengthLocal = null;
        if (afterBound < 0 && all.get(headerCode.get(1)) instanceof VarInsnNode boundLoad && boundLoad.getOpcode() == Opcodes.ILOAD && boundLoad.var != index) {
            lengthLocal = boundLoad.var;
            afterBound = 2;
        }
        if (afterBound < 0) {
            return decline("the loop's bound is not the list's own length, so it may not visit every element");
        }
        int exitTest = headerCode.get(afterBound);
        if (!(all.get(exitTest) instanceof JumpInsnNode exit) || exit.getOpcode() != Opcodes.IF_ICMPGE) {
            return decline("the loop's exit test is not `index >= length`");
        }
        int exitLabel = at.get(exit.label);
        if (exitLabel <= backJump) {
            return decline("the loop's exit does not leave the loop");
        }

        // ---- the increment: iinc i 1 immediately before the back edge, and i's only other write is i = 0.
        int increment = previousReal(all, backJump);
        if (increment < 0 || !(all.get(increment) instanceof IincInsnNode step) || step.var != index) {
            return decline("the index is not advanced at the end of each iteration");
        }
        if (step.incr != 1) {
            return decline("the index advances by " + step.incr + ", so the loop does not visit every element exactly once; the segmented reduction would");
        }

        // ---- the body: everything between the exit test and the increment.
        int bodyStart = exitTest + 1;
        int bodyEnd = increment; // exclusive
        List<Integer> bodyCode = realBetween(all, bodyStart, bodyEnd);
        int load = loadOpcode(resultKind);
        int store = storeOpcode(resultKind);
        int add = addOpcode(resultKind);
        if (bodyCode.size() < 3) {
            return decline("the loop body is not a single accumulation");
        }
        AbstractInsnNode lastInsn = all.get(bodyCode.get(bodyCode.size() - 1));
        AbstractInsnNode addInsn = all.get(bodyCode.get(bodyCode.size() - 2));
        if (lastInsn.getOpcode() != store || addInsn.getOpcode() != add) {
            return decline("the loop body does not end with `acc = acc + value`, the only update a segmented sum represents");
        }
        int accumulator = ((VarInsnNode) lastInsn).var;
        int addAt = bodyCode.get(bodyCode.size() - 2);
        // The accumulator is loaded exactly once in the body, and that load is one operand of the
        // final add -- established by dataflow, not by position, so temporaries computed before
        // `acc +=` are fine and `acc * x + y` is not.
        int accLoad = -1;
        for (int k : bodyCode) {
            if (all.get(k) instanceof VarInsnNode var && var.var == accumulator && k != bodyCode.get(bodyCode.size() - 1)) {
                if (accLoad >= 0 || var.getOpcode() != load) {
                    return decline("the accumulator is read or written more than once in the loop body, so the result depends on more than a sum");
                }
                accLoad = k;
            }
        }
        if (accLoad < 0) {
            return decline("the loop body does not add to the running total");
        }
        String operand = accumulatorIsAnAddOperand(body, all.get(addAt), all.get(accLoad));
        if (operand != null) {
            return decline(operand);
        }
        int expressionFrom = bodyStart;
        int expressionTo = addAt; // exclusive; the accumulator load inside it is skipped below
        // ---- jumps: nothing leaves the expression, and nothing enters it except by falling in.
        for (int i = 0; i < all.size(); i++) {
            boolean inside = i >= expressionFrom && i < expressionTo;
            for (LabelNode target : targetsOf(all.get(i))) {
                int t = at.get(target);
                boolean targetInside = t >= expressionFrom && t < expressionTo;
                if (inside && !targetInside) {
                    return decline("the loop body can skip the accumulation or leave the loop (a `continue`, a `break` or a conditional update), "
                            + "so not every element contributes; the segmented reduction would add them all");
                }
                if (!inside && targetInside) {
                    return decline("control enters the element computation from outside it");
                }
                if (inside && t <= i) {
                    return decline("the element computation contains a loop of its own");
                }
            }
            if (inside && isExit(all.get(i))) {
                return decline("the loop body can return early, so the result is not the sum over every element");
            }
        }

        // ---- the prologue: acc = 0, i = 0, optionally n = length; nothing else.
        String prologueProblem = checkPrologue(all, header, accumulator, index, lengthLocal, resultKind);
        if (prologueProblem != null) {
            return decline(prologueProblem);
        }
        // ---- the epilogue: return acc, and nothing else after the loop.
        List<Integer> epilogue = realBetween(all, exitLabel, all.size());
        if (epilogue.size() != 2 || all.get(epilogue.get(0)).getOpcode() != load || ((VarInsnNode) all.get(epilogue.get(0))).var != accumulator
                || all.get(epilogue.get(1)).getOpcode() != returnOpcode(resultKind)) {
            return decline("the method does more after the loop than return the accumulator");
        }
        // ---- the header is reached only by falling into it and from the back edge.
        for (int i = 0; i < all.size(); i++) {
            if (i != backJump && targetsOf(all.get(i)).stream().anyMatch(l -> at.get(l) == header)) {
                return decline("the loop header is entered from somewhere other than the start and the back edge");
            }
        }

        // ---- every use of i, acc, n and the list's slots must be one accounted for above.
        Set<Integer> accounted = new HashSet<>();
        accounted.add(headerCode.get(0));
        accounted.add(increment);
        accounted.add(bodyCode.get(bodyCode.size() - 1));
        accounted.add(accLoad);
        accounted.add(epilogue.get(0));
        for (int k = 1; k < afterBound; k++) {
            accounted.add(headerCode.get(k));
        }

        // Build the element function while checking the expression.
        MethodNode element = new MethodNode(Opcodes.ASM9, Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, "udfElement",
                "(" + elementKind.descriptor() + ")" + resultKind.descriptor(), null, null);
        InsnList out = new InsnList();
        Map<LabelNode, LabelNode> labels = new HashMap<>();
        for (int i = expressionFrom; i < expressionTo; i++) {
            if (all.get(i) instanceof LabelNode label) {
                labels.put(label, new LabelNode());
            }
        }
        boolean hasJumps = labels.values().size() > 0 && containsJump(all, expressionFrom, expressionTo);
        Map<Integer, Integer> temporaries = new HashMap<>();
        int nextTemporary = elementKind.slots();
        int i = expressionFrom;
        while (i < expressionTo) {
            if (i == accLoad) {
                i++;
                continue;
            }
            int readEnd = matchElementRead(all, i, expressionTo, index, elementKind);
            if (readEnd > 0) {
                out.add(new VarInsnNode(loadOpcode(elementKind), 0));
                for (int k = i; k < readEnd; k++) {
                    accounted.add(k);
                }
                i = readEnd;
                continue;
            }
            AbstractInsnNode insn = all.get(i);
            if (insn instanceof VarInsnNode var) {
                if (var.var == index || var.var == accumulator || var.var <= END || (lengthLocal != null && var.var == lengthLocal)) {
                    return decline(var.var == index ? "the element computation uses the loop index itself, so it depends on position and not only on the element"
                            : var.var == accumulator ? "the element computation reads the running total, so the result depends on the order of accumulation"
                                    : "the element computation reaches the list other than through its current element");
                }
                Integer mapped = temporaries.get(var.var);
                if (mapped == null) {
                    if (hasJumps || !isStore(var.getOpcode())) {
                        return decline("the element computation uses a local that is not set from the current element before it is read");
                    }
                    if (usedOutside(all, var.var, expressionFrom, expressionTo)) {
                        return decline("the element computation shares a local with the rest of the method, so iterations are not independent");
                    }
                    mapped = nextTemporary;
                    temporaries.put(var.var, mapped);
                    nextTemporary += isWide(var.getOpcode()) ? 2 : 1;
                }
                out.add(new VarInsnNode(var.getOpcode(), mapped));
                accounted.add(i);
                i++;
                continue;
            }
            if (insn instanceof IincInsnNode) {
                return decline("the element computation increments a local, which carries state between iterations");
            }
            out.add(insn.clone(labels));
            accounted.add(i);
            i++;
        }
        // Anything left that touches the special slots is an unexplained use.
        for (int k = 0; k < all.size(); k++) {
            if (accounted.contains(k)) {
                continue;
            }
            int slot = slotOf(all.get(k));
            if (slot < 0) {
                continue;
            }
            boolean special = slot == index || slot == accumulator || slot <= END || (lengthLocal != null && slot == lengthLocal);
            boolean prologueInit = k < header;
            if (special && !prologueInit) {
                return decline("the loop index, the accumulator or the list is used outside the canonical fold");
            }
        }

        out.add(new InsnNode(returnOpcode(resultKind)));
        element.instructions = out;
        element.maxStack = body.maxStack + 4;
        element.maxLocals = nextTemporary + 4;
        return Optional.of(new Split(element, elementKind, resultKind));
    }

    /**
     * Why {@code load} is not exactly one operand of {@code add}, or null when it is. With the
     * accumulator loaded once and consumed directly by the add, the update is {@code acc + E} (or
     * {@code E + acc}) where {@code E} does not involve the accumulator at all.
     */
    private static String accumulatorIsAnAddOperand(MethodNode body, AbstractInsnNode add, AbstractInsnNode load) {
        org.objectweb.asm.tree.analysis.Frame<org.objectweb.asm.tree.analysis.SourceValue>[] frames;
        try {
            frames = new org.objectweb.asm.tree.analysis.Analyzer<>(new org.objectweb.asm.tree.analysis.SourceInterpreter()).analyze("ai/beestream/jitudf/Lowered", body);
        } catch (org.objectweb.asm.tree.analysis.AnalyzerException e) {
            return "the loop body's dataflow could not be analysed";
        }
        int index = body.instructions.indexOf(add);
        var frame = frames[index];
        if (frame == null || frame.getStackSize() < 2) {
            return "the accumulation is not reachable";
        }
        var left = frame.getStack(frame.getStackSize() - 2).insns;
        var right = frame.getStack(frame.getStackSize() - 1).insns;
        boolean isLeft = left.size() == 1 && left.contains(load);
        boolean isRight = right.size() == 1 && right.contains(load);
        if (isLeft == isRight) {
            return "the running total is not simply added to: it takes part in the element computation, so the result depends on the order of accumulation";
        }
        return null;
    }

    /** acc = 0; i = 0; [n = length;] in any order, and nothing else before the header. */
    private static String checkPrologue(List<AbstractInsnNode> all, int header, int accumulator, int index, Integer lengthLocal, PrimitiveKind kind) {
        List<Integer> code = realBetween(all, 0, header);
        boolean accSet = false;
        boolean indexSet = false;
        boolean lengthSet = lengthLocal == null;
        int i = 0;
        while (i < code.size()) {
            AbstractInsnNode insn = all.get(code.get(i));
            if (isZero(insn, kind) && i + 1 < code.size() && all.get(code.get(i + 1)) instanceof VarInsnNode s && s.getOpcode() == storeOpcode(kind) && s.var == accumulator && !accSet) {
                accSet = true;
                i += 2;
                continue;
            }
            if (insn.getOpcode() == Opcodes.ICONST_0 && i + 1 < code.size() && all.get(code.get(i + 1)) instanceof VarInsnNode s && s.getOpcode() == Opcodes.ISTORE && s.var == index
                    && !indexSet) {
                indexSet = true;
                i += 2;
                continue;
            }
            if (!lengthSet) {
                int after = matchLength(all, code, i);
                if (after > 0 && after < code.size() && all.get(code.get(after)) instanceof VarInsnNode s && s.getOpcode() == Opcodes.ISTORE && s.var == lengthLocal) {
                    lengthSet = true;
                    i = after + 1;
                    continue;
                }
            }
            if (insn.getOpcode() == Opcodes.GOTO) {
                return "the loop is entered by a jump, not by falling into its header";
            }
            if (i + 1 < code.size() && all.get(code.get(i + 1)) instanceof VarInsnNode s && s.getOpcode() == storeOpcode(kind) && s.var == accumulator) {
                return "the accumulator does not start at zero, the identity the segmented sum starts from";
            }
            return "the method does something before the loop other than set the accumulator to zero and the index to zero";
        }
        if (!accSet) {
            return "the accumulator does not start at zero, the identity the segmented sum starts from";
        }
        if (!indexSet) {
            return "the index does not start at zero, so the loop does not visit the first element";
        }
        if (!lengthSet) {
            return "the loop's bound is a local that is not exactly the list's length";
        }
        return null;
    }

    /** Whether {@code insn} pushes zero of {@code kind}: the sum's identity. */
    private static boolean isZero(AbstractInsnNode insn, PrimitiveKind kind) {
        return switch (kind) {
            case INT -> insn.getOpcode() == Opcodes.ICONST_0;
            case LONG -> insn.getOpcode() == Opcodes.LCONST_0;
            case FLOAT -> insn.getOpcode() == Opcodes.FCONST_0;
            case DOUBLE -> insn.getOpcode() == Opcodes.DCONST_0;
            case BOOL -> false;
        };
    }

    /**
     * Matches the lowered length of the list at position {@code from} of {@code code} (indices into
     * {@code all}); returns the position after it, or -1. {@link AsmUdfLowering} lowers both
     * {@code arr.length} and {@code seq.length()} to {@code aload 0; pop; iload end; iload start; isub}.
     */
    private static int matchLength(List<AbstractInsnNode> all, List<Integer> code, int from) {
        if (from + 5 > code.size()) {
            return -1;
        }
        AbstractInsnNode a = all.get(code.get(from));
        AbstractInsnNode b = all.get(code.get(from + 1));
        AbstractInsnNode c = all.get(code.get(from + 2));
        AbstractInsnNode d = all.get(code.get(from + 3));
        AbstractInsnNode e = all.get(code.get(from + 4));
        boolean match = a instanceof VarInsnNode va && va.getOpcode() == Opcodes.ALOAD && va.var == VALUES
                && b.getOpcode() == Opcodes.POP
                && c instanceof VarInsnNode vc && vc.getOpcode() == Opcodes.ILOAD && vc.var == END
                && d instanceof VarInsnNode vd && vd.getOpcode() == Opcodes.ILOAD && vd.var == START
                && e.getOpcode() == Opcodes.ISUB;
        return match ? from + 5 : -1;
    }

    /**
     * Matches a read of the current element at {@code i}; returns the index after it, or -1. Lowered
     * from {@code arr[i]} and {@code seq.apply(i)} alike to
     * {@code aload 0; iload i; iload start; iadd; invokevirtual Array.get}.
     */
    private static int matchElementRead(List<AbstractInsnNode> all, int i, int limit, int index, PrimitiveKind elementKind) {
        List<Integer> code = realBetween(all, i, limit);
        if (code.size() < 5 || code.get(0) != i) {
            return -1;
        }
        AbstractInsnNode a = all.get(code.get(0));
        AbstractInsnNode b = all.get(code.get(1));
        AbstractInsnNode c = all.get(code.get(2));
        AbstractInsnNode d = all.get(code.get(3));
        AbstractInsnNode e = all.get(code.get(4));
        boolean match = a instanceof VarInsnNode va && va.getOpcode() == Opcodes.ALOAD && va.var == VALUES
                && b instanceof VarInsnNode vb && vb.getOpcode() == Opcodes.ILOAD && vb.var == index
                && c instanceof VarInsnNode vc && vc.getOpcode() == Opcodes.ILOAD && vc.var == START
                && d.getOpcode() == Opcodes.IADD
                && e instanceof MethodInsnNode get && get.name.equals("get") && get.owner.equals(elementKind.arrayInternalName());
        return match ? code.get(4) + 1 : -1;
    }

    private static List<LabelNode> targetsOf(AbstractInsnNode insn) {
        List<LabelNode> targets = new ArrayList<>();
        if (insn instanceof JumpInsnNode jump) {
            targets.add(jump.label);
        } else if (insn instanceof TableSwitchInsnNode table) {
            targets.add(table.dflt);
            targets.addAll(table.labels);
        } else if (insn instanceof LookupSwitchInsnNode lookup) {
            targets.add(lookup.dflt);
            targets.addAll(lookup.labels);
        }
        return targets;
    }

    private static boolean containsJump(List<AbstractInsnNode> all, int from, int to) {
        for (int i = from; i < to; i++) {
            if (!targetsOf(all.get(i)).isEmpty()) {
                return true;
            }
        }
        return false;
    }

    private static boolean isExit(AbstractInsnNode insn) {
        int op = insn.getOpcode();
        return (op >= Opcodes.IRETURN && op <= Opcodes.RETURN) || op == Opcodes.ATHROW;
    }

    private static List<Integer> realFrom(List<AbstractInsnNode> all, int from, int count) {
        List<Integer> out = new ArrayList<>();
        for (int i = from; i < all.size() && out.size() < count; i++) {
            if (all.get(i).getOpcode() >= 0) {
                out.add(i);
            }
        }
        return out;
    }

    private static List<Integer> realBetween(List<AbstractInsnNode> all, int from, int to) {
        List<Integer> out = new ArrayList<>();
        for (int i = from; i < to && i < all.size(); i++) {
            if (all.get(i).getOpcode() >= 0) {
                out.add(i);
            }
        }
        return out;
    }

    private static int previousReal(List<AbstractInsnNode> all, int i) {
        for (int j = i - 1; j >= 0; j--) {
            if (all.get(j).getOpcode() >= 0) {
                return j;
            }
        }
        return -1;
    }

    private static boolean usedOutside(List<AbstractInsnNode> all, int slot, int from, int to) {
        for (int i = 0; i < all.size(); i++) {
            if ((i < from || i >= to) && slotOf(all.get(i)) == slot) {
                return true;
            }
        }
        return false;
    }

    private static int slotOf(AbstractInsnNode insn) {
        if (insn instanceof VarInsnNode var) {
            return var.var;
        }
        if (insn instanceof IincInsnNode inc) {
            return inc.var;
        }
        return -1;
    }

    private static boolean isStore(int opcode) {
        return opcode >= Opcodes.ISTORE && opcode <= Opcodes.ASTORE;
    }

    private static boolean isWide(int opcode) {
        return opcode == Opcodes.LSTORE || opcode == Opcodes.DSTORE || opcode == Opcodes.LLOAD || opcode == Opcodes.DLOAD;
    }

    private static int addOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.IADD;
            case LONG -> Opcodes.LADD;
            case FLOAT -> Opcodes.FADD;
            case DOUBLE -> Opcodes.DADD;
            case BOOL -> throw new IllegalStateException("a list reduction cannot sum booleans");
        };
    }

    private static int loadOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT, BOOL -> Opcodes.ILOAD;
            case LONG -> Opcodes.LLOAD;
            case FLOAT -> Opcodes.FLOAD;
            case DOUBLE -> Opcodes.DLOAD;
        };
    }

    private static int storeOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT, BOOL -> Opcodes.ISTORE;
            case LONG -> Opcodes.LSTORE;
            case FLOAT -> Opcodes.FSTORE;
            case DOUBLE -> Opcodes.DSTORE;
        };
    }

    private static int returnOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT, BOOL -> Opcodes.IRETURN;
            case LONG -> Opcodes.LRETURN;
            case FLOAT -> Opcodes.FRETURN;
            case DOUBLE -> Opcodes.DRETURN;
        };
    }
}
