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

import java.util.List;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.IntInsnNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.analysis.Analyzer;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.Interpreter;
import org.objectweb.asm.tree.analysis.Value;

/**
 * What is known about every integer value at every instruction of a method: an exact constant,
 * "non-zero", or nothing.
 *
 * <p>
 * This is what decides whether a division can raise. It used to be decided by looking at the one
 * instruction in front of the division, which is not a fact about the divisor: in
 * {@code Math.floorMod(x, x == 7 ? y : 3)} the instruction textually in front of the call pushes
 * {@code 3}, but it is the end of only one of the two paths that reach the call -- the other
 * arrives with {@code y}, which can be zero. The rule that matters is the one a data-flow analysis
 * enforces by construction: a divisor is known non-zero only if it is non-zero on <em>every</em>
 * path that reaches the division.
 * </p>
 *
 * <p>
 * The analysis is ASM's standard forward fixpoint over the method's control-flow graph with a
 * three-level lattice per value -- {@code CONST(c)} below {@code NONZERO} below {@code UNKNOWN} --
 * and a join that keeps only what holds on both sides: equal constants stay that constant, two
 * non-zero facts stay non-zero, anything else is unknown. Locals are tracked through stores and
 * loads, so a divisor assigned on different paths is joined at the point the paths meet, and a
 * loop's back edge is joined into its header until nothing changes. Values that come from outside
 * -- arguments, calls, fields, array elements -- are unknown. Arithmetic folds only where the
 * result is exact (two constants); it does not try to reason about ranges.
 * </p>
 *
 * <p>
 * Unknown is not a refusal. A backend that can report an arithmetic error compiles the division
 * with a zero test that raises, so precision here only decides whether the test is needed, never
 * whether the UDF reaches the GPU.
 * </p>
 */
final class DivisorFacts {

    /** One abstract value. */
    static final class Fact implements Value {
        static final int CONST = 0;
        static final int NONZERO = 1;
        static final int UNKNOWN = 2;

        final int size;
        final int level;
        final long constant;

        private Fact(int size, int level, long constant) {
            this.size = size;
            this.level = level;
            this.constant = constant;
        }

        static Fact constant(int size, long value) {
            return new Fact(size, CONST, value);
        }

        static Fact unknown(int size) {
            return new Fact(size, UNKNOWN, 0);
        }

        static Fact nonZero(int size) {
            return new Fact(size, NONZERO, 0);
        }

        @Override
        public int getSize() {
            return size;
        }

        boolean isConstant() {
            return level == CONST;
        }

        /** Whether this value is non-zero on every path. */
        boolean provablyNonZero() {
            return level == NONZERO || level == CONST && constant != 0;
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof Fact f && f.size == size && f.level == level && f.constant == constant;
        }

        @Override
        public int hashCode() {
            return (size * 31 + level) * 31 + Long.hashCode(constant);
        }

        @Override
        public String toString() {
            return level == CONST ? "const " + constant : level == NONZERO ? "non-zero" : "unknown";
        }
    }

    private final Frame<Fact>[] frames;

    private DivisorFacts(Frame<Fact>[] frames) {
        this.frames = frames;
    }

    /** The facts for {@code method}, or null if the method could not be analysed. */
    static DivisorFacts of(String owner, MethodNode method) {
        try {
            return new DivisorFacts(new Analyzer<>(new FactInterpreter()).analyze(owner, method));
        } catch (AnalyzerException | RuntimeException e) {
            return null;
        }
    }

    /**
     * The fact about the value {@code depthFromTop} slots below the top of the operand stack just
     * before {@code insn} executes (0 is the top), or unknown if the instruction is unreachable.
     */
    Fact operand(MethodNode method, AbstractInsnNode insn, int depthFromTop) {
        Frame<Fact> frame = frames[method.instructions.indexOf(insn)];
        if (frame == null) {
            return Fact.unknown(1);
        }
        return frame.getStack(frame.getStackSize() - 1 - depthFromTop);
    }

    /** Whether {@code insn} is ever reached. Unreachable code has no frame. */
    boolean reachable(MethodNode method, AbstractInsnNode insn) {
        return frames[method.instructions.indexOf(insn)] != null;
    }

    private static final class FactInterpreter extends Interpreter<Fact> {

        FactInterpreter() {
            super(Opcodes.ASM9);
        }

        @Override
        public Fact newValue(Type type) {
            if (type == Type.VOID_TYPE) {
                return null;
            }
            return Fact.unknown(type == null ? 1 : type.getSize());
        }

        @Override
        public Fact newOperation(AbstractInsnNode insn) {
            int op = insn.getOpcode();
            if (op >= Opcodes.ICONST_M1 && op <= Opcodes.ICONST_5) {
                return Fact.constant(1, op - Opcodes.ICONST_0);
            }
            switch (op) {
                case Opcodes.LCONST_0:
                    return Fact.constant(2, 0);
                case Opcodes.LCONST_1:
                    return Fact.constant(2, 1);
                case Opcodes.BIPUSH, Opcodes.SIPUSH:
                    return Fact.constant(1, ((IntInsnNode) insn).operand);
                case Opcodes.LDC: {
                    Object cst = ((LdcInsnNode) insn).cst;
                    if (cst instanceof Integer i) {
                        return Fact.constant(1, i);
                    }
                    if (cst instanceof Long l) {
                        return Fact.constant(2, l);
                    }
                    return Fact.unknown(cst instanceof Double ? 2 : 1);
                }
                case Opcodes.DCONST_0, Opcodes.DCONST_1, Opcodes.GETSTATIC:
                    if (op == Opcodes.GETSTATIC) {
                        return Fact.unknown(Type.getType(((org.objectweb.asm.tree.FieldInsnNode) insn).desc).getSize());
                    }
                    return Fact.unknown(2);
                default:
                    return Fact.unknown(1);
            }
        }

        @Override
        public Fact copyOperation(AbstractInsnNode insn, Fact value) {
            return value;
        }

        @Override
        public Fact unaryOperation(AbstractInsnNode insn, Fact value) {
            switch (insn.getOpcode()) {
                case Opcodes.IINC:
                    return value.isConstant() ? Fact.constant(1, (int) (value.constant + ((IincInsnNode) insn).incr)) : Fact.unknown(1);
                case Opcodes.INEG:
                    // Negation is a bijection that fixes zero, so it keeps "non-zero".
                    return value.isConstant() ? Fact.constant(1, (int) -value.constant) : value.level == Fact.NONZERO ? Fact.nonZero(1) : Fact.unknown(1);
                case Opcodes.LNEG:
                    return value.isConstant() ? Fact.constant(2, -value.constant) : value.level == Fact.NONZERO ? Fact.nonZero(2) : Fact.unknown(2);
                case Opcodes.I2L:
                    // Widening keeps the value, and so keeps "non-zero".
                    return value.isConstant() ? Fact.constant(2, value.constant) : value.level == Fact.NONZERO ? Fact.nonZero(2) : Fact.unknown(2);
                case Opcodes.L2I:
                    // Narrowing can turn a non-zero long into zero; only an exact constant survives.
                    return value.isConstant() ? Fact.constant(1, (int) value.constant) : Fact.unknown(1);
                case Opcodes.CHECKCAST:
                    return value;
                case Opcodes.I2D, Opcodes.F2D, Opcodes.F2L, Opcodes.D2L, Opcodes.L2D, Opcodes.DNEG:
                    return Fact.unknown(2);
                case Opcodes.GETFIELD:
                    return Fact.unknown(Type.getType(((org.objectweb.asm.tree.FieldInsnNode) insn).desc).getSize());
                default:
                    return Fact.unknown(1);
            }
        }

        @Override
        public Fact binaryOperation(AbstractInsnNode insn, Fact a, Fact b) {
            int op = insn.getOpcode();
            boolean both = a.isConstant() && b.isConstant();
            switch (op) {
                case Opcodes.IADD:
                    return both ? Fact.constant(1, (int) (a.constant + b.constant)) : Fact.unknown(1);
                case Opcodes.ISUB:
                    return both ? Fact.constant(1, (int) (a.constant - b.constant)) : Fact.unknown(1);
                case Opcodes.IMUL:
                    return both ? Fact.constant(1, (int) (a.constant * b.constant)) : Fact.unknown(1);
                case Opcodes.IOR:
                    // A bit set in either operand stays set.
                    if (a.provablyNonZero() || b.provablyNonZero()) {
                        return both ? Fact.constant(1, (int) (a.constant | b.constant)) : Fact.nonZero(1);
                    }
                    return Fact.unknown(1);
                case Opcodes.LADD:
                    return both ? Fact.constant(2, a.constant + b.constant) : Fact.unknown(2);
                case Opcodes.LSUB:
                    return both ? Fact.constant(2, a.constant - b.constant) : Fact.unknown(2);
                case Opcodes.LMUL:
                    return both ? Fact.constant(2, a.constant * b.constant) : Fact.unknown(2);
                case Opcodes.LOR:
                    if (a.provablyNonZero() || b.provablyNonZero()) {
                        return both ? Fact.constant(2, a.constant | b.constant) : Fact.nonZero(2);
                    }
                    return Fact.unknown(2);
                case Opcodes.LALOAD, Opcodes.DALOAD, Opcodes.LDIV, Opcodes.LREM, Opcodes.LAND, Opcodes.LXOR, Opcodes.LSHL, Opcodes.LSHR, Opcodes.LUSHR,
                        Opcodes.DADD, Opcodes.DSUB, Opcodes.DMUL, Opcodes.DDIV, Opcodes.DREM:
                    return Fact.unknown(2);
                default:
                    return Fact.unknown(1);
            }
        }

        @Override
        public Fact ternaryOperation(AbstractInsnNode insn, Fact a, Fact b, Fact c) {
            return Fact.unknown(1);
        }

        @Override
        public Fact naryOperation(AbstractInsnNode insn, List<? extends Fact> values) {
            Type returned;
            if (insn instanceof org.objectweb.asm.tree.MethodInsnNode call) {
                returned = Type.getReturnType(call.desc);
            } else if (insn instanceof org.objectweb.asm.tree.InvokeDynamicInsnNode indy) {
                returned = Type.getReturnType(indy.desc);
            } else {
                return Fact.unknown(1); // MULTIANEWARRAY
            }
            if (returned == Type.VOID_TYPE) {
                return null;
            }
            // Boxing and unboxing are the identity on the number: keep what is known about it.
            if (insn instanceof org.objectweb.asm.tree.MethodInsnNode call && values.size() == 1
                    && BoxConversions.isKnown(call) && BoxConversions.replacementFor(call) == BoxConversions.IDENTITY) {
                Fact in = values.get(0);
                return in.isConstant() ? Fact.constant(returned.getSize(), in.constant) : in.level == Fact.NONZERO ? Fact.nonZero(returned.getSize()) : Fact.unknown(returned.getSize());
            }
            return Fact.unknown(returned.getSize());
        }

        @Override
        public void returnOperation(AbstractInsnNode insn, Fact value, Fact expected) {
        }

        @Override
        public Fact merge(Fact a, Fact b) {
            if (a.equals(b)) {
                return a;
            }
            int size = Math.max(a.size, b.size);
            if (a.provablyNonZero() && b.provablyNonZero()) {
                return Fact.nonZero(size);
            }
            return Fact.unknown(size);
        }
    }
}
