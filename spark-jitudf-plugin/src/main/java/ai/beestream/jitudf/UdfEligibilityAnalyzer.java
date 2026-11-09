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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.Set;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;

/**
 * Decides whether a UDF can be turned into a TornadoVM kernel, by reading the bytecode of every
 * method it can reach.
 *
 * <p>
 * The decision has to be conservative, and it has to be made here, at planning time. Once the plan
 * says the UDF runs on the GPU, there is no way back: a failure inside
 * {@code RapidsUDF.evaluateColumnar} on an executor fails the query, it does not quietly fall back
 * to the CPU. So anything this analysis has not positively established is a refusal, and the query
 * takes the path it would have taken without this integration at all.
 * </p>
 *
 * <p>
 * What is allowed is a closed set: arithmetic and comparisons on {@code int}, {@code long},
 * {@code float} and {@code double}, branches and loops over them, the {@code java.lang.Math}
 * functions TornadoVM has device equivalents for, boxing and unboxing of exactly those four types
 * (which is all the Java UDF interfaces' signatures amount to), and calls to other methods of the
 * same class that pass this same analysis. Everything else -- allocation, fields, strings,
 * exceptions, {@code invokedynamic}, recursion, monitors, anything referring to a type not listed
 * -- is refused by name, so adding support for something later is a deliberate act and not an
 * accident.
 * </p>
 *
 * <p>
 * One array shape is allowed, for Spark's {@code ArrayType} over a supported element type: a
 * parameter of the entry method, used only as {@code arr.length} and {@code arr[i]}. That is
 * exactly what a cuDF {@code LIST} column can serve without materialising anything -- the length is
 * the difference of two offsets and the element is an index into the flattened child. An array that
 * is stored into a local, passed to a helper, written to, compared, or reached in any other way is
 * refused, because none of those have an answer in that layout. Where the array came from is
 * established with a dataflow analysis, not by the shape of the surrounding instructions.
 * </p>
 */
public final class UdfEligibilityAnalyzer {

    /** How deep the reachable-method walk goes before giving up rather than looping. */
    private static final int MAX_METHODS = 32;

    /** {@code java.lang.Math} functions TornadoVM lowers to device built-ins. */
    private static final Set<String> SUPPORTED_MATH = Set.of( //
            "sqrt", "exp", "log", "log10", "sin", "cos", "tan", "tanh", "sinh", "cosh", //
            "asin", "acos", "atan", "atan2", "cbrt", "abs", "floor", "ceil", "rint", //
            "pow", "min", "max", "fma", "signum", "hypot", "toRadians", "toDegrees");

    private UdfEligibilityAnalyzer() {
    }

    /**
     * Analyses {@code udf} against the schema Spark will call it with.
     *
     * @param udf
     *         the user's method, as {@link UdfResolver} found it.
     * @param inputTypes
     *         the Spark types of the UDF's arguments, in order.
     * @param outputType
     *         the Spark type the UDF returns.
     * @param inputsNullable
     *         whether any input column may contain nulls. Nulls are not supported yet, and a
     *         nullable input is refused rather than guessed at.
     * @param deterministic
     *         whether Spark considers the UDF deterministic. A non-deterministic UDF is refused:
     *         running it per row on a GPU would change when and how often it is evaluated.
     */
    public static EligibilityResult analyse(ResolvedUdf udf, List<DataType> inputTypes, DataType outputType, boolean inputsNullable, boolean deterministic) {
        if (udf == null) {
            return EligibilityResult.ineligible("the user's UDF could not be found behind Spark's wrapper");
        }
        if (!deterministic) {
            return EligibilityResult.ineligible("the UDF is not deterministic");
        }
        if (inputsNullable) {
            return EligibilityResult.ineligible("an input column is nullable, and null handling is not implemented");
        }

        PrimitiveKind outputKind = PrimitiveKind.ofSparkType(outputType);
        if (outputKind == null) {
            return EligibilityResult.ineligible("the return type " + outputType.simpleString() + " is not a fixed-width numeric type");
        }
        List<InputSpec> inputs = new ArrayList<>();
        for (DataType inputType : inputTypes) {
            if (inputType instanceof ArrayType arrayType) {
                PrimitiveKind element = PrimitiveKind.ofSparkType(arrayType.elementType());
                if (element == null) {
                    return EligibilityResult.ineligible("the array element type " + arrayType.elementType().simpleString() + " is not a fixed-width numeric type");
                }
                if (arrayType.containsNull()) {
                    return EligibilityResult.ineligible("the array column may contain null elements, and null handling is not implemented");
                }
                inputs.add(InputSpec.list(element));
                continue;
            }
            PrimitiveKind kind = PrimitiveKind.ofSparkType(inputType);
            if (kind == null) {
                return EligibilityResult.ineligible("the argument type " + inputType.simpleString() + " is not a fixed-width numeric type");
            }
            inputs.add(InputSpec.scalar(kind));
        }

        // The method's own signature has to agree with the schema, boxed or not. A mismatch means
        // the resolved method is not the one Spark calls.
        Class<?>[] parameterTypes = udf.method().getParameterTypes();
        if (parameterTypes.length != inputs.size()) {
            return EligibilityResult.ineligible("the resolved method takes " + parameterTypes.length + " arguments but the UDF is called with " + inputs.size());
        }
        for (int i = 0; i < parameterTypes.length; i++) {
            InputSpec spec = inputs.get(i);
            if (spec.isList()) {
                // Either a primitive array, whose component type must agree with the column, or a
                // Scala sequence, which is what Spark hands a Java UDF for an ArrayType column and
                // whose element type is erased -- so the column's element type is taken as the
                // truth and the casts in the bytecode are checked against it (ArrayParameters).
                PrimitiveKind component = PrimitiveKind.ofArrayComponent(parameterTypes[i]);
                boolean isSequence = isScalaSequence(parameterTypes[i]);
                if (component != spec.kind() && !isSequence) {
                    return EligibilityResult.ineligible("argument " + i + " is declared " + parameterTypes[i].getSimpleName() + " but the column is " + inputTypes.get(i).simpleString()
                            + "; an array column needs a matching primitive array or a scala.collection.Seq");
                }
                continue;
            }
            if (PrimitiveKind.ofJavaType(parameterTypes[i]) != spec.kind()) {
                return EligibilityResult.ineligible("argument " + i + " is declared " + parameterTypes[i].getSimpleName() + " but the column is " + inputTypes.get(i).simpleString());
            }
        }
        if (PrimitiveKind.ofJavaType(udf.method().getReturnType()) != outputKind) {
            return EligibilityResult.ineligible("the method returns " + udf.method().getReturnType().getSimpleName() + " but the UDF is declared " + outputType.simpleString());
        }

        String problem = walkReachableMethods(udf, inputs);
        if (problem != null) {
            return EligibilityResult.ineligible(problem);
        }
        return EligibilityResult.eligible(udf, inputs, outputKind);
    }

    /** Walks every method the UDF can reach inside its own class; returns the first problem or null. */
    private static String walkReachableMethods(ResolvedUdf udf, List<InputSpec> inputs) {
        ClassNode classNode = new ClassNode();
        new ClassReader(udf.classFile()).accept(classNode, ClassReader.SKIP_FRAMES);
        String owner = classNode.name;

        Deque<String> pending = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        String entry = udf.method().getName() + Type.getMethodDescriptor(udf.method());
        pending.add(entry);
        seen.add(entry);

        // Who calls whom, so that a cycle anywhere in the graph is found -- not only a method that
        // calls itself, and not only the entry method. A helper that recurses, or two helpers that
        // call each other, is just as impossible to compile into a kernel.
        Map<String, Set<String>> callGraph = new HashMap<>();

        while (!pending.isEmpty()) {
            if (seen.size() > MAX_METHODS) {
                return "the UDF reaches more than " + MAX_METHODS + " methods, which is more than this analysis will follow";
            }
            String key = pending.poll();
            MethodNode method = find(classNode, key);
            if (method == null) {
                return "the method " + key + " could not be read back from " + owner;
            }
            if (method.tryCatchBlocks != null && !method.tryCatchBlocks.isEmpty()) {
                return "the UDF catches exceptions, and exception semantics are not supported on the device";
            }
            boolean isEntry = key.equals(entry);
            if (!isEntry) {
                String signatureProblem = checkHelperSignature(method, key);
                if (signatureProblem != null) {
                    return signatureProblem;
                }
            }
            Set<Integer> referenceSlots = isEntry ? parameterSlots(method, inputs) : Set.of();
            String problem = checkInstructions(method, classNode, pending, seen, key, callGraph, referenceSlots,
                    isEntry ? ArrayParameters.of(method, inputs) : ArrayParameters.none());
            if (problem != null) {
                return problem;
            }
        }
        String cycle = findCycle(callGraph);
        if (cycle != null) {
            return "the UDF's call graph contains a cycle (" + cycle + "); recursion cannot be compiled into a kernel";
        }
        return null;
    }

    /**
     * A helper is copied into the generated class unchanged, so its signature has to already be
     * one the kernel can speak: static, and primitive in and out. A helper taking or returning a
     * boxed value leaves the lowered call site disagreeing with the body it calls, which is invalid
     * bytecode rather than a wrong number.
     */
    private static String checkHelperSignature(MethodNode method, String key) {
        if ((method.access & Opcodes.ACC_STATIC) == 0) {
            return "the UDF calls its own instance method " + method.name + "; only static helpers can be lowered";
        }
        for (Type argument : Type.getArgumentTypes(method.desc)) {
            if (!isSupportedPrimitive(argument)) {
                return "the helper " + key + " takes " + argument.getClassName() + ", and only int, long, float and double are carried into a kernel";
            }
        }
        Type returnType = Type.getReturnType(method.desc);
        if (!isSupportedPrimitive(returnType)) {
            return "the helper " + key + " returns " + returnType.getClassName() + ", and only int, long, float and double are carried into a kernel";
        }
        return null;
    }

    private static boolean isSupportedPrimitive(Type type) {
        return switch (type.getSort()) {
            case Type.INT, Type.LONG, Type.FLOAT, Type.DOUBLE -> true;
            default -> false;
        };
    }

    /** The local slots of the entry method's parameters, which are the only references it may load. */
    private static Set<Integer> parameterSlots(MethodNode method, List<InputSpec> inputs) {
        Set<Integer> slots = new HashSet<>();
        int slot = (method.access & Opcodes.ACC_STATIC) != 0 ? 0 : 1;
        for (Type argument : Type.getArgumentTypes(method.desc)) {
            slots.add(slot);
            slot += argument.getSize();
        }
        return slots;
    }

    /** Any cycle in the call graph, named for the diagnostic, or null when it is acyclic. */
    private static String findCycle(Map<String, Set<String>> callGraph) {
        Set<String> done = new HashSet<>();
        for (String start : callGraph.keySet()) {
            Deque<String> path = new ArrayDeque<>();
            String found = visit(start, callGraph, new HashSet<>(), done, path);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    private static String visit(String node, Map<String, Set<String>> graph, Set<String> onPath, Set<String> done, Deque<String> path) {
        if (onPath.contains(node)) {
            return node + " is reached from itself";
        }
        if (!done.add(node)) {
            return null;
        }
        onPath.add(node);
        for (String next : graph.getOrDefault(node, Set.of())) {
            String found = visit(next, graph, onPath, done, path);
            if (found != null) {
                return found;
            }
        }
        onPath.remove(node);
        return null;
    }

    private static MethodNode find(ClassNode classNode, String nameAndDescriptor) {
        for (MethodNode method : classNode.methods) {
            if ((method.name + method.desc).equals(nameAndDescriptor)) {
                return method;
            }
        }
        return null;
    }

    /**
     * Opcodes this compiler knows how to carry into a kernel unchanged.
     *
     * <p>
     * The list is positive on purpose. The previous version refused a set of opcodes it knew to be
     * dangerous and accepted everything else, which meant every opcode nobody had thought about was
     * admitted by default: loading a null, comparing two references, duplicating a stack slot whose
     * width changes once the value is unboxed. A whitelist cannot fail that way. Anything new is
     * refused until somebody decides it is safe and writes it down here.
     * </p>
     */
    private static final Set<Integer> ALLOWED_OPCODES = Set.of(
            Opcodes.NOP,
            // constants
            Opcodes.ICONST_M1, Opcodes.ICONST_0, Opcodes.ICONST_1, Opcodes.ICONST_2, Opcodes.ICONST_3,
            Opcodes.ICONST_4, Opcodes.ICONST_5, Opcodes.LCONST_0, Opcodes.LCONST_1,
            Opcodes.FCONST_0, Opcodes.FCONST_1, Opcodes.FCONST_2, Opcodes.DCONST_0, Opcodes.DCONST_1,
            Opcodes.BIPUSH, Opcodes.SIPUSH,
            // primitive locals
            Opcodes.ILOAD, Opcodes.LLOAD, Opcodes.FLOAD, Opcodes.DLOAD,
            Opcodes.ISTORE, Opcodes.LSTORE, Opcodes.FSTORE, Opcodes.DSTORE, Opcodes.IINC,
            // arithmetic
            Opcodes.IADD, Opcodes.LADD, Opcodes.FADD, Opcodes.DADD,
            Opcodes.ISUB, Opcodes.LSUB, Opcodes.FSUB, Opcodes.DSUB,
            Opcodes.IMUL, Opcodes.LMUL, Opcodes.FMUL, Opcodes.DMUL,
            Opcodes.FDIV, Opcodes.DDIV, Opcodes.FREM, Opcodes.DREM,
            Opcodes.INEG, Opcodes.LNEG, Opcodes.FNEG, Opcodes.DNEG,
            Opcodes.ISHL, Opcodes.ISHR, Opcodes.IUSHR, Opcodes.LSHL, Opcodes.LSHR, Opcodes.LUSHR,
            Opcodes.IAND, Opcodes.IOR, Opcodes.IXOR, Opcodes.LAND, Opcodes.LOR, Opcodes.LXOR,
            // comparisons and branches on primitives
            Opcodes.LCMP, Opcodes.FCMPL, Opcodes.FCMPG, Opcodes.DCMPL, Opcodes.DCMPG,
            Opcodes.IFEQ, Opcodes.IFNE, Opcodes.IFLT, Opcodes.IFGE, Opcodes.IFGT, Opcodes.IFLE,
            Opcodes.IF_ICMPEQ, Opcodes.IF_ICMPNE, Opcodes.IF_ICMPLT, Opcodes.IF_ICMPGE,
            Opcodes.IF_ICMPGT, Opcodes.IF_ICMPLE, Opcodes.GOTO,
            Opcodes.TABLESWITCH, Opcodes.LOOKUPSWITCH,
            // conversions between the four kinds this integration carries
            Opcodes.I2L, Opcodes.I2F, Opcodes.I2D, Opcodes.L2I, Opcodes.L2F, Opcodes.L2D,
            Opcodes.F2I, Opcodes.F2L, Opcodes.F2D, Opcodes.D2I, Opcodes.D2L, Opcodes.D2F,
            // returns
            Opcodes.IRETURN, Opcodes.LRETURN, Opcodes.FRETURN, Opcodes.DRETURN);

    private static String checkInstructions(MethodNode method, ClassNode classNode, Deque<String> pending, Set<String> seen, String self,
            Map<String, Set<String>> callGraph, Set<Integer> referenceSlots, ArrayParameters arrays) {
        String arrayProblem = arrays.check(method);
        if (arrayProblem != null) {
            return arrayProblem;
        }
        String owner = classNode.name;
        AbstractInsnNode[] instructions = method.instructions.toArray();
        for (int i = 0; i < instructions.length; i++) {
            AbstractInsnNode insn = instructions[i];
            int op = insn.getOpcode();
            if (op < 0) {
                continue; // labels, line numbers, frames
            }
            if (ALLOWED_OPCODES.contains(op)) {
                String divisionProblem = checkDivision(insn, i > 0 ? instructions[i - 1] : null, op);
                if (divisionProblem != null) {
                    return divisionProblem;
                }
                continue;
            }
            switch (op) {
                // Integer division and remainder are allowed only where the divisor is provably
                // non-zero, because the device has no way to raise Java's ArithmeticException.
                case Opcodes.IDIV, Opcodes.IREM, Opcodes.LDIV, Opcodes.LREM -> {
                    String divisionProblem = checkDivision(insn, i > 0 ? instructions[i - 1] : null, op);
                    if (divisionProblem != null) {
                        return divisionProblem;
                    }
                }
                case Opcodes.LDC -> {
                    Object constant = ((org.objectweb.asm.tree.LdcInsnNode) insn).cst;
                    if (!(constant instanceof Integer || constant instanceof Long || constant instanceof Float || constant instanceof Double)) {
                        return "the UDF loads the constant " + constant + ", which is not a number a kernel can hold";
                    }
                }
                case Opcodes.ALOAD -> {
                    int slot = ((org.objectweb.asm.tree.VarInsnNode) insn).var;
                    if (slot == 0 && (method.access & Opcodes.ACC_STATIC) == 0) {
                        return "the UDF uses this to reach an instance field, which is state the kernel does not have";
                    }
                    if (!referenceSlots.contains(slot)) {
                        return "the UDF loads a reference from local slot " + slot + ", which is not one of its arguments";
                    }
                }
                case Opcodes.ARETURN -> {
                    // Safe only because ACONST_NULL is refused below and the declared return type
                    // was matched against the column's type: the value returned is the boxed result.
                }
                case Opcodes.ACONST_NULL ->
                    { return "the UDF can return or use null, which a kernel column has no representation for"; }
                case Opcodes.IF_ACMPEQ, Opcodes.IF_ACMPNE, Opcodes.IFNULL, Opcodes.IFNONNULL ->
                    { return "the UDF compares references, which is identity rather than arithmetic and has no kernel meaning"; }
                case Opcodes.NEW, Opcodes.NEWARRAY, Opcodes.ANEWARRAY, Opcodes.MULTIANEWARRAY ->
                    { return "the UDF allocates an object or an array, which a kernel cannot do"; }
                case Opcodes.ATHROW -> { return "the UDF throws, and exception semantics are not supported on the device"; }
                case Opcodes.MONITORENTER, Opcodes.MONITOREXIT -> { return "the UDF synchronizes, which has no meaning on the device"; }
                case Opcodes.INVOKEDYNAMIC -> { return "the UDF uses invokedynamic (a lambda, a string concatenation or a switch on strings)"; }
                case Opcodes.INVOKEINTERFACE -> {
                    if (!arrays.readsAListParameter(insn)) {
                        return "the UDF makes an interface call, which cannot be resolved to one device function";
                    }
                }
                case Opcodes.INSTANCEOF, Opcodes.CHECKCAST -> {
                    if (insn instanceof org.objectweb.asm.tree.TypeInsnNode type && !isSupportedBox(type.desc)) {
                        return "the UDF casts to or tests " + type.desc.replace('/', '.') + ", which is not a supported numeric type";
                    }
                }
                case Opcodes.GETFIELD, Opcodes.PUTFIELD, Opcodes.GETSTATIC, Opcodes.PUTSTATIC -> {
                    String fieldProblem = checkFieldAccess((FieldInsnNode) insn, op, classNode);
                    if (fieldProblem != null) {
                        return fieldProblem;
                    }
                }
                case Opcodes.IALOAD, Opcodes.LALOAD, Opcodes.FALOAD, Opcodes.DALOAD, Opcodes.ARRAYLENGTH -> {
                    if (!arrays.readsAListParameter(insn)) {
                        return "the UDF reads an array that is not one of its LIST arguments, or reaches it other than directly";
                    }
                }
                case Opcodes.INVOKESTATIC, Opcodes.INVOKEVIRTUAL, Opcodes.INVOKESPECIAL -> {
                    MethodInsnNode call = (MethodInsnNode) insn;
                    String callProblem = checkCall(call, owner, pending, seen, self, callGraph);
                    if (callProblem != null) {
                        return callProblem;
                    }
                }
                default ->
                    { return "the UDF uses bytecode " + op + ", which is not in the set this compiler knows how to carry into a kernel"; }
            }
        }
        return null;
    }

    /**
     * Integer division and remainder divide by zero on the device without raising anything, so the
     * query would silently get a number where Java raises {@code ArithmeticException}. They are
     * admitted only when the divisor is a non-zero constant sitting directly in front of them.
     */
    private static String checkDivision(AbstractInsnNode insn, AbstractInsnNode previous, int op) {
        if (op != Opcodes.IDIV && op != Opcodes.IREM && op != Opcodes.LDIV && op != Opcodes.LREM) {
            return null;
        }
        Long divisor = constantOf(previous);
        if (divisor == null) {
            return "the UDF divides by a value this analysis cannot prove non-zero; Java would raise ArithmeticException and a kernel cannot";
        }
        if (divisor == 0) {
            return "the UDF divides by the constant zero";
        }
        if (divisor == -1) {
            // Integer.MIN_VALUE / -1 overflows and, in Java, wraps rather than raising; the device
            // agrees only by accident, so it is refused rather than reasoned about.
            return "the UDF divides by the constant -1, whose overflow behaviour is not established here";
        }
        return null;
    }

    /** The constant an instruction pushes, or null when it does not push one. */
    private static Long constantOf(AbstractInsnNode insn) {
        if (insn == null) {
            return null;
        }
        int op = insn.getOpcode();
        if (op >= Opcodes.ICONST_M1 && op <= Opcodes.ICONST_5) {
            return (long) (op - Opcodes.ICONST_0);
        }
        if (op == Opcodes.LCONST_0) {
            return 0L;
        }
        if (op == Opcodes.LCONST_1) {
            return 1L;
        }
        if (op == Opcodes.BIPUSH || op == Opcodes.SIPUSH) {
            return (long) ((org.objectweb.asm.tree.IntInsnNode) insn).operand;
        }
        if (op == Opcodes.LDC && ((org.objectweb.asm.tree.LdcInsnNode) insn).cst instanceof Number n
                && (((org.objectweb.asm.tree.LdcInsnNode) insn).cst instanceof Integer || ((org.objectweb.asm.tree.LdcInsnNode) insn).cst instanceof Long)) {
            return n.longValue();
        }
        return null;
    }

    /**
     * Only a {@code static final} primitive constant of the UDF's own class is readable: its value
     * is fixed before the kernel is generated, so the kernel sees the number the CPU would. An
     * instance field is state the kernel has no access to; a mutable static is state that can change
     * between batches; a field of another class is outside what was analysed at all.
     */
    private static String checkFieldAccess(FieldInsnNode field, int op, ClassNode classNode) {
        if (op == Opcodes.GETFIELD || op == Opcodes.PUTFIELD) {
            return "the UDF reads or writes an instance field, which is state the kernel does not have";
        }
        if (op == Opcodes.PUTSTATIC) {
            return "the UDF writes a static field, which is a side effect";
        }
        if (!field.owner.equals(classNode.name)) {
            return "the UDF reads the static field " + field.owner.replace('/', '.') + "." + field.name + ", which is outside its own class";
        }
        if (!isSupportedPrimitive(Type.getType(field.desc))) {
            return "the UDF reads the static field " + field.name + ", whose type is not one a kernel carries";
        }
        for (org.objectweb.asm.tree.FieldNode declared : classNode.fields) {
            if (declared.name.equals(field.name) && declared.desc.equals(field.desc)) {
                if ((declared.access & Opcodes.ACC_FINAL) == 0) {
                    return "the UDF reads the mutable static field " + field.name + ", whose value can change between batches";
                }
                return null;
            }
        }
        return "the static field " + field.name + " is not declared by the UDF's own class";
    }

    private static String checkCall(MethodInsnNode call, String owner, Deque<String> pending, Set<String> seen, String self,
            Map<String, Set<String>> callGraph) {
        if (call.owner.equals("java/lang/Math") || call.owner.equals("java/lang/StrictMath")) {
            if (!SUPPORTED_MATH.contains(call.name)) {
                return "the UDF calls Math." + call.name + ", which has no device equivalent here";
            }
            return null;
        }
        if (isSupportedBox("L" + call.owner + ";")) {
            // Only the exact boxing and conversion calls the lowering can replace precisely.
            if (BoxConversions.isKnown(call)) {
                return null;
            }
            return "the UDF calls " + call.owner.replace('/', '.') + "." + call.name + call.desc
                    + ", which this compiler cannot lower to an exact primitive equivalent";
        }
        if (!call.owner.equals(owner)) {
            return "the UDF calls " + call.owner.replace('/', '.') + "." + call.name + ", outside its own class";
        }
        if (call.getOpcode() == Opcodes.INVOKESPECIAL && call.name.equals("<init>")) {
            return "the UDF calls a constructor";
        }
        if (call.getOpcode() != Opcodes.INVOKESTATIC) {
            return "the UDF calls its own instance method " + call.name + "; only static helpers can be lowered";
        }
        String key = call.name + call.desc;
        callGraph.computeIfAbsent(self, k -> new HashSet<>()).add(key);
        if (seen.add(key)) {
            pending.add(key);
        }
        return null;
    }

    /** Whether {@code type} is a Scala sequence, which is how Spark delivers an ArrayType column. */
    private static boolean isScalaSequence(Class<?> type) {
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            if (c.getName().startsWith("scala.collection.")) {
                return true;
            }
            for (Class<?> implemented : c.getInterfaces()) {
                if (implemented.getName().startsWith("scala.collection.")) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean isSupportedBox(String internalNameOrDescriptor) {
        String name = internalNameOrDescriptor.startsWith("L") ? internalNameOrDescriptor.substring(1, internalNameOrDescriptor.length() - 1) : internalNameOrDescriptor;
        return name.equals("java/lang/Integer") || name.equals("java/lang/Long") || name.equals("java/lang/Float") || name.equals("java/lang/Double");
    }
}
