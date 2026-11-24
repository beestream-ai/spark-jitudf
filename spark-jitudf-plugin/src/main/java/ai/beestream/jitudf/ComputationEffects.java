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

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;

/**
 * What a computation can do that changes how it must be compiled, found in the bytecode that will
 * actually be compiled -- including the JDK bodies TC inlines -- rather than in the UDF's signature.
 *
 * <p>
 * Two decisions depend on this, and both used to look in the wrong place:
 * </p>
 *
 * <ul>
 * <li><strong>Arithmetic errors.</strong> A device function reports Java's division by zero only
 * when it is compiled with error reporting on. The analysis turned that on when it saw
 * {@code idiv}/{@code irem} in the UDF itself. But {@code Math.floorDiv(x, y)} contains its
 * {@code idiv} in {@code java.base}, and TC inlines that body: the division reached the GPU with no
 * error channel, and a zero divisor produced a number where Java throws. So the bodies of the JDK
 * methods the analysis admits are read here, transitively, and a division found in any of them is
 * treated exactly like one written in the UDF.</li>
 * <li><strong>Floating point.</strong> RAPIDS gates results that can differ from the CPU behind
 * {@code spark.rapids.sql.incompatibleOps.enabled}. The gate looked at the input and output types,
 * so {@code (int) (Math.sin(x) * 1e9)} -- integers in, integers out, a device {@code sin} in the
 * middle -- went straight past it. Here any floating-point operation anywhere in the reachable code
 * counts: arithmetic, comparisons, conversions, constants, a call taking or returning a float or
 * double, and the same inside inlined JDK bodies.</li>
 * </ul>
 *
 * <p>
 * An integer UDF with no floating point anywhere stays eligible with the switch off, exactly as
 * before; only the ones that really compute in floating point now need it.
 * </p>
 */
final class ComputationEffects {

    /** The effects of one method or call: whether it can divide by zero, and whether it uses FP. */
    record Effects(boolean mayDivideByZero, boolean usesFloatingPoint) {
        static final Effects NONE = new Effects(false, false);

        Effects or(Effects other) {
            return new Effects(mayDivideByZero || other.mayDivideByZero, usesFloatingPoint || other.usesFloatingPoint);
        }
    }

    private static final int MAX_JDK_METHODS = 24;

    /** JDK summaries never change within a JVM, so each is computed once. */
    private static final Map<String, Effects> JDK_SUMMARIES = new ConcurrentHashMap<>();

    private ComputationEffects() {
    }

    /** Whether {@code insn}, on its own, is a floating-point operation. */
    static boolean isFloatingPoint(AbstractInsnNode insn) {
        int op = insn.getOpcode();
        switch (op) {
            case Opcodes.FCONST_0, Opcodes.FCONST_1, Opcodes.FCONST_2, Opcodes.DCONST_0, Opcodes.DCONST_1,
                    Opcodes.FLOAD, Opcodes.DLOAD, Opcodes.FSTORE, Opcodes.DSTORE,
                    Opcodes.FADD, Opcodes.DADD, Opcodes.FSUB, Opcodes.DSUB, Opcodes.FMUL, Opcodes.DMUL,
                    Opcodes.FDIV, Opcodes.DDIV, Opcodes.FREM, Opcodes.DREM, Opcodes.FNEG, Opcodes.DNEG,
                    Opcodes.FCMPL, Opcodes.FCMPG, Opcodes.DCMPL, Opcodes.DCMPG,
                    Opcodes.I2F, Opcodes.I2D, Opcodes.L2F, Opcodes.L2D, Opcodes.F2I, Opcodes.F2L, Opcodes.F2D,
                    Opcodes.D2I, Opcodes.D2L, Opcodes.D2F, Opcodes.FRETURN, Opcodes.DRETURN,
                    Opcodes.FALOAD, Opcodes.DALOAD, Opcodes.FASTORE, Opcodes.DASTORE -> {
                return true;
            }
            default -> {
            }
        }
        if (insn instanceof LdcInsnNode ldc) {
            return ldc.cst instanceof Float || ldc.cst instanceof Double;
        }
        if (insn instanceof MethodInsnNode call) {
            return mentionsFloatingPoint(call.desc) || isFloatingBox(call.owner);
        }
        if (insn instanceof FieldInsnNode field) {
            return mentionsFloatingPoint(field.desc);
        }
        if (insn instanceof TypeInsnNode type) {
            return isFloatingBox(type.desc);
        }
        return false;
    }

    private static boolean mentionsFloatingPoint(String descriptor) {
        Type type = Type.getType(descriptor);
        if (type.getSort() == Type.METHOD) {
            for (Type argument : type.getArgumentTypes()) {
                if (isFloatingType(argument)) {
                    return true;
                }
            }
            return isFloatingType(type.getReturnType());
        }
        return isFloatingType(type);
    }

    private static boolean isFloatingType(Type type) {
        return switch (type.getSort()) {
            case Type.FLOAT, Type.DOUBLE -> true;
            case Type.OBJECT -> isFloatingBox(type.getInternalName());
            case Type.ARRAY -> isFloatingType(type.getElementType());
            default -> false;
        };
    }

    private static boolean isFloatingBox(String internalName) {
        String name = internalName.startsWith("L") && internalName.endsWith(";") ? internalName.substring(1, internalName.length() - 1) : internalName;
        return name.equals("java/lang/Float") || name.equals("java/lang/Double");
    }

    /** Whether {@code insn} is an integer division or remainder. */
    static boolean isIntegerDivision(AbstractInsnNode insn) {
        int op = insn.getOpcode();
        return op == Opcodes.IDIV || op == Opcodes.IREM || op == Opcodes.LDIV || op == Opcodes.LREM;
    }

    /**
     * The effects of calling the JDK method {@code call}, from its own body and everything it calls
     * in {@code java.base}, transitively. A method whose bytecode cannot be read is assumed to do
     * both, which is the safe direction: error reporting on, and the floating-point gate applied.
     */
    static Effects ofJdkCall(MethodInsnNode call) {
        if (isDeviceIntrinsic(call)) {
            // TC replaces these with a CUDA built-in and never compiles the JDK body (Math.sqrt's
            // body is FdLibm, which divides on integer bits, but sqrt() on the device cannot raise
            // anything). The call's effect is the built-in's: floating point iff its signature is,
            // and no arithmetic exception -- as in Java, where none of them throws.
            return new Effects(false, mentionsFloatingPoint(call.desc));
        }
        String key = call.owner + "." + call.name + call.desc;
        return JDK_SUMMARIES.computeIfAbsent(key, k -> summarise(call.owner, call.name, call.desc));
    }

    /**
     * The {@code Math}/{@code StrictMath} functions TC maps to device built-ins. {@code floorDiv} and
     * {@code floorMod} are deliberately absent: TC compiles those from the JDK's own body, which is
     * exactly where their division is.
     */
    private static final Set<String> INTRINSIC_MATH = Set.of("sqrt", "exp", "log", "log10", "sin", "cos", "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "atan2",
            "cbrt", "abs", "floor", "ceil", "rint", "pow", "min", "max", "fma", "signum", "hypot", "toRadians", "toDegrees");

    private static boolean isDeviceIntrinsic(MethodInsnNode call) {
        return (call.owner.equals("java/lang/Math") || call.owner.equals("java/lang/StrictMath")) && INTRINSIC_MATH.contains(call.name);
    }

    private static Effects summarise(String owner, String name, String desc) {
        Effects effects = new Effects(false, mentionsFloatingPoint(desc));
        Deque<String[]> pending = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        pending.add(new String[] { owner, name, desc });
        seen.add(owner + "." + name + desc);
        Map<String, ClassNode> classes = new java.util.HashMap<>();
        while (!pending.isEmpty()) {
            if (seen.size() > MAX_JDK_METHODS) {
                return new Effects(true, true);
            }
            String[] target = pending.poll();
            ClassNode node = classes.computeIfAbsent(target[0], ComputationEffects::readJdkClass);
            MethodNode method = node == null ? null : find(node, target[1], target[2]);
            if (method == null || method.instructions.size() == 0) {
                // Native, intrinsic-only, or unreadable: nothing is known, so assume the worst.
                return new Effects(true, true);
            }
            for (AbstractInsnNode insn : method.instructions) {
                if (isIntegerDivision(insn)) {
                    effects = effects.or(new Effects(true, false));
                }
                if (isFloatingPoint(insn)) {
                    effects = effects.or(new Effects(false, true));
                }
                if (insn instanceof MethodInsnNode inner && inner.owner.startsWith("java/") && seen.add(inner.owner + "." + inner.name + inner.desc)) {
                    pending.add(new String[] { inner.owner, inner.name, inner.desc });
                }
            }
        }
        return effects;
    }

    private static MethodNode find(ClassNode node, String name, String desc) {
        for (MethodNode method : node.methods) {
            if (method.name.equals(name) && method.desc.equals(desc)) {
                return method;
            }
        }
        return null;
    }

    private static ClassNode readJdkClass(String internalName) {
        try (InputStream in = Object.class.getResourceAsStream("/" + internalName + ".class")) {
            if (in == null) {
                return null;
            }
            ClassNode node = new ClassNode();
            new ClassReader(in.readAllBytes()).accept(node, ClassReader.SKIP_FRAMES | ClassReader.SKIP_DEBUG);
            return node;
        } catch (IOException e) {
            return null;
        }
    }
}
