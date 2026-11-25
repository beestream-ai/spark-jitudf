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
            "pow", "min", "max", "fma", "signum", "hypot", "toRadians", "toDegrees", //
            // Integer division that rounds towards negative infinity. Not a device built-in and not
            // in TornadoVM's intrinsic table either: these reach the GPU because TC follows the call
            // into java.base and compiles the JDK's own body, so the semantics are Java's by
            // construction rather than by a mapping someone has to keep correct.
            "floorMod", "floorDiv");

    /**
     * Static {@code java.lang} methods that are compiled by inlining the JDK's own implementation.
     *
     * <p>
     * None of these is in TornadoVM's intrinsic table. They work because TC resolves a call it does
     * not recognise by reading the callee's bytecode, and every method here has a pure-arithmetic
     * body in {@code java.base} -- no allocation, no exceptions, no state. {@code Long.bitCount}
     * compiles to the JDK's own SWAR popcount; {@code Integer.rotateLeft} folds to a shift pair.
     * </p>
     *
     * <p>
     * They are listed explicitly rather than admitted by a rule, because "pure arithmetic in
     * java.base" is not a property the analyser can check: it is a property of these specific
     * bodies, which is why {@code CudfDeviceFunctionTest} compiles every one of them rather than
     * assuming the list is right.
     * </p>
     *
     * <p>
     * Bit manipulation is not an arbitrary choice of extension. It is what the integer-bitmap UDFs
     * of the packed-bitmap kind are made of -- counting set days in a packed {@code long}, shifting
     * a bitmap by a day -- and what a custom-seed Murmur3 bucketing UDF needs.
     * </p>
     */
    private static final Set<String> SUPPORTED_JDK_STATICS = Set.of( //
            "java/lang/Integer.bitCount(I)I", "java/lang/Integer.rotateLeft(II)I", "java/lang/Integer.rotateRight(II)I", //
            "java/lang/Integer.reverse(I)I", "java/lang/Integer.reverseBytes(I)I", "java/lang/Integer.signum(I)I", //
            "java/lang/Integer.numberOfLeadingZeros(I)I", "java/lang/Integer.numberOfTrailingZeros(I)I", //
            "java/lang/Integer.highestOneBit(I)I", "java/lang/Integer.lowestOneBit(I)I", //
            "java/lang/Long.bitCount(J)I", "java/lang/Long.rotateLeft(JI)J", "java/lang/Long.rotateRight(JI)J", //
            "java/lang/Long.reverse(J)J", "java/lang/Long.reverseBytes(J)J", "java/lang/Long.signum(J)I", //
            "java/lang/Long.numberOfLeadingZeros(J)I", "java/lang/Long.numberOfTrailingZeros(J)I", //
            "java/lang/Long.highestOneBit(J)J", "java/lang/Long.lowestOneBit(J)J");

    /** The methods of {@link #SUPPORTED_JDK_STATICS}, for the test that compiles each of them. */
    static Set<String> supportedJdkStatics() {
        return SUPPORTED_JDK_STATICS;
    }

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
        return analyse(udf, inputTypes, outputType, inputsNullable, deterministic, false);
    }

    /**
     * As above, for a backend that can report an arithmetic error back to Spark rather than having
     * to prove one cannot happen. cudf's transform can: its JIT kernel casts a non-void result from
     * the device function to {@code cudf::errc} and raises {@code cudf::evaluation_error} for the
     * call, which is a failed query, as Java's {@code ArithmeticException} would be. The TornadoVM
     * runtime backend has no such channel, so it passes false and keeps refusing.
     */
    public static EligibilityResult analyse(ResolvedUdf udf, List<DataType> inputTypes, DataType outputType, boolean inputsNullable, boolean deterministic,
            boolean canReportArithmeticErrors) {
        return analyse(udf, inputTypes, outputType, inputsNullable, deterministic, canReportArithmeticErrors, false);
    }

    /**
     * As above, for a backend that can also run the UDF over a nullable column.
     *
     * <p>
     * {@code canRunNullAware} says that two things are true at once: TC can compile the method in
     * its null-aware mode, and the binding can call it with {@code null_aware::YES}. Then a null
     * does not have to be guessed at -- it is passed to the compiled method, exactly as Spark
     * passes it to the method on the CPU, and whatever that method answers is the answer. Without
     * both halves a nullable column is still refused, because the alternative on offer -- copying
     * the input's validity mask onto the output -- is wrong for every UDF that does something other
     * than return null for a null.
     * </p>
     */
    public static EligibilityResult analyse(ResolvedUdf udf, List<DataType> inputTypes, DataType outputType, boolean inputsNullable, boolean deterministic,
            boolean canReportArithmeticErrors, boolean canRunNullAware) {
        if (udf == null) {
            return EligibilityResult.ineligible("the user's UDF could not be found behind Spark's wrapper");
        }
        if (!deterministic) {
            return EligibilityResult.ineligible("the UDF is not deterministic");
        }
        // Null-aware whenever nulls are in play at all: either the column can hold one, or the UDF
        // itself talks about nulls. The second half matters more than it looks. A UDF that opens
        // with a defensive `if (value == null)` is perfectly ordinary, and over a non-nullable
        // column that test is dead code -- but it is still in the bytecode, and the plain lowering
        // erases the boxes the test is about, so there is nothing left for `== null` to mean. The
        // choice is between refusing a UDF that would have run correctly and compiling it in the
        // mode that can express the test. The second costs an optional per value and is right.
        boolean nullAware = canRunNullAware && (inputsNullable || entryTestsForNull(udf));
        if (nullAware && udf.nullSensitiveAdapter()) {
            // The method is reached through a wrapper that boxes an argument or unboxes the result.
            // For every value that is exact; for a null it is not -- the wrapper raises (or, in
            // Scala, yields zero) where the compiled method would pass the null on.
            return EligibilityResult.ineligible("the UDF is reached through a wrapper that unboxes a value, which treats null differently from the method it "
                    + "forwards to, and nulls are in play here");
        }
        if (inputsNullable && !canRunNullAware) {
            // Not "not implemented yet": the obvious implementation is unsound here, and the measurement
            // is in bench/NullSemanticsProbe. Spark skips a UDF and yields null for a null argument only
            // where it knows the argument to be primitive (ScalaUDF.inputPrimitives). A Java UDF1<Long,
            // Integer> has a boxed parameter whatever the column is, so inputPrimitives is List(false)
            // and Spark calls the UDF with the null. What happens then is the UDF's own business: a body
            // that unboxes at once throws NullPointerException, and one that tests for null returns
            // whatever it chooses, which is rarely null. Propagating the input's validity mask onto the
            // output would silently replace both with null.
            //
            // Doing it properly means compiling the null test, which is TC's --null-aware mode: it takes
            // cuda::std::optional arguments, and the binding has to be able to ask for them. Where both
            // are available the branch above is taken instead; this one is the backend that cannot.
            return EligibilityResult.ineligible("an input column is nullable, and Spark passes a null argument to a UDF with a boxed parameter rather than "
                    + "short-circuiting it, so the UDF's own null behaviour would have to be compiled, which this backend cannot do");
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

        boolean[] needsErrorReporting = { false };
        boolean[] usesFloatingPoint = { false };
        String problem = walkReachableMethods(udf, inputs, canReportArithmeticErrors, needsErrorReporting, nullAware, usesFloatingPoint);
        if (problem != null) {
            return EligibilityResult.ineligible(problem);
        }
        // The signature counts too: a float column read and ignored still has to be decoded.
        boolean floatingPoint = usesFloatingPoint[0] || SparkSemantics.usesFloatingPoint(outputKind, inputs);
        if (nullAware) {
            // The compilation itself is the remaining gate. TC refuses a null-aware method that does
            // something it cannot give a null meaning to -- comparing two boxes with equals, for
            // instance -- and that refusal arrives at planning time, as a CPU fallback.
            return EligibilityResult.eligibleNullAware(udf, inputs, outputKind, needsErrorReporting[0]).withFloatingPoint(floatingPoint);
        }
        return (needsErrorReporting[0] ? EligibilityResult.eligibleWithArithmeticErrors(udf, inputs, outputKind) : EligibilityResult.eligible(udf, inputs, outputKind))
                .withFloatingPoint(floatingPoint);
    }

    /**
     * Whether the UDF's own method mentions null: loads one, or branches on one.
     *
     * <p>
     * Only the entry method is looked at, because only the entry method has boxed parameters. A
     * helper's arguments are primitives by the time it is called -- {@link #checkHelperSignature}
     * insists on it -- so a helper cannot be handed a null to test.
     * </p>
     */
    private static boolean entryTestsForNull(ResolvedUdf udf) {
        ClassNode classNode = new ClassNode();
        new ClassReader(udf.classFile()).accept(classNode, ClassReader.SKIP_FRAMES);
        MethodNode entry = find(classNode, udf.method().getName() + Type.getMethodDescriptor(udf.method()));
        if (entry == null) {
            return false;
        }
        for (AbstractInsnNode insn : entry.instructions) {
            switch (insn.getOpcode()) {
                case Opcodes.ACONST_NULL, Opcodes.IFNULL, Opcodes.IFNONNULL -> {
                    return true;
                }
                default -> {
                }
            }
        }
        return false;
    }

    /** Walks every method the UDF can reach inside its own class; returns the first problem or null. */
    private static String walkReachableMethods(ResolvedUdf udf, List<InputSpec> inputs, boolean canReportArithmeticErrors, boolean[] needsErrorReporting, boolean nullAware,
            boolean[] usesFloatingPoint) {
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
            ArrayParameters arrays = isEntry ? ArrayParameters.of(method, inputs) : ArrayParameters.none();
            String problem = checkInstructions(method, classNode, pending, seen, key, callGraph, referenceSlots,
                    arrays, canReportArithmeticErrors, needsErrorReporting, nullAware && isEntry, CastProof.of(owner, method, arrays, isEntry ? inputs : null),
                    usesFloatingPoint);
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
            // BOOLEAN is here for a static helper that returns or takes one, and for a static
            // final boolean constant. Whether a boolean may be an input or a result column is a
            // separate question, settled by PrimitiveKind and by the backend: the TornadoVM
            // runtime backend refuses a boolean result because it has no BooleanArray to put it in.
            case Type.INT, Type.LONG, Type.FLOAT, Type.DOUBLE, Type.BOOLEAN -> true;
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
            Map<String, Set<String>> callGraph, Set<Integer> referenceSlots, ArrayParameters arrays, boolean canReportArithmeticErrors, boolean[] needsErrorReporting,
            boolean nullAware, CastProof casts, boolean[] usesFloatingPoint) {
        String arrayProblem = arrays.check(method);
        if (arrayProblem != null) {
            return arrayProblem;
        }
        String owner = classNode.name;
        AbstractInsnNode[] instructions = method.instructions.toArray();
        // What every divisor can be, on every path that reaches its division (see DivisorFacts).
        DivisorFacts divisors = DivisorFacts.of(owner, method);
        for (int i = 0; i < instructions.length; i++) {
            AbstractInsnNode insn = instructions[i];
            int op = insn.getOpcode();
            if (op < 0) {
                continue; // labels, line numbers, frames
            }
            if (ComputationEffects.isFloatingPoint(insn)) {
                usesFloatingPoint[0] = true;
            }
            if (ALLOWED_OPCODES.contains(op)) {
                String divisionProblem = checkDivision(insn, divisorOf(divisors, method, insn), op, canReportArithmeticErrors, needsErrorReporting);
                if (divisionProblem != null) {
                    return divisionProblem;
                }
                continue;
            }
            switch (op) {
                // Integer division and remainder are allowed only where the divisor is provably
                // non-zero, because the device has no way to raise Java's ArithmeticException.
                case Opcodes.IDIV, Opcodes.IREM, Opcodes.LDIV, Opcodes.LREM -> {
                    String divisionProblem = checkDivision(insn, divisorOf(divisors, method, insn), op, canReportArithmeticErrors, needsErrorReporting);
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
                        return "the UDF uses this to reach an instance field or an instance method, which is state or dynamic dispatch the kernel does not have";
                    }
                    if (!referenceSlots.contains(slot)) {
                        return "the UDF loads a reference from local slot " + slot + ", which is not one of its arguments";
                    }
                }
                case Opcodes.ARETURN -> {
                    // Over a non-nullable column this is safe only because ACONST_NULL is refused
                    // below and the declared return type was matched against the column's: what is
                    // returned is the boxed result. Over a nullable one it may also be a null, which
                    // is a null in the output column.
                }
                case Opcodes.ACONST_NULL -> {
                    if (!nullAware) {
                        return "the UDF can return or use null, which a kernel column has no representation for";
                    }
                    // Null-aware: the output is a cuda::std::optional, so a null has a representation.
                }
                case Opcodes.IFNULL, Opcodes.IFNONNULL -> {
                    if (!nullAware) {
                        return "the UDF compares references, which is identity rather than arithmetic and has no kernel meaning";
                    }
                    // Null-aware: `x == null` is has_value(), which is the whole point of the mode.
                    // IF_ACMPEQ/NE stay refused below -- comparing two boxes is identity, and Java's
                    // own answer for it depends on the Integer cache rather than on the numbers.
                }
                case Opcodes.IF_ACMPEQ, Opcodes.IF_ACMPNE ->
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
                    org.objectweb.asm.tree.TypeInsnNode type = (org.objectweb.asm.tree.TypeInsnNode) insn;
                    if (!isSupportedBox(type.desc)) {
                        return "the UDF casts to or tests " + type.desc.replace('/', '.') + ", which is not a supported numeric type";
                    }
                    if (op == Opcodes.INSTANCEOF && nullAware) {
                        // Over a nullable column `x instanceof Integer` is `x != null`, which the
                        // null-aware compilation would have to express; it is not established that it does.
                        return "the UDF tests instanceof over a nullable column, which this compiler does not lower";
                    }
                    // Lowering erases the cast (or folds the test), which is only right when it
                    // cannot fail: the value must provably already be that box.
                    String castProblem = casts.whyUnproven(type);
                    if (castProblem != null) {
                        return castProblem;
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
                    if (call.owner.startsWith("java/") && !BoxConversions.isKnown(call)) {
                        // TC compiles this call by inlining (or intrinsifying) the JDK's own body, so
                        // what that body can do is part of what this UDF does.
                        String effectProblem = checkJdkCallEffects(call, divisorOf(divisors, method, insn), canReportArithmeticErrors, needsErrorReporting,
                                usesFloatingPoint);
                        if (effectProblem != null) {
                            return effectProblem;
                        }
                    }
                }
                default ->
                    { return "the UDF uses bytecode " + op + ", which is not in the set this compiler knows how to carry into a kernel"; }
            }
        }
        return null;
    }

    /**
     * Applies the effects of an admitted JDK call, found in its own bytecode (see
     * {@link ComputationEffects}): a division anywhere in its body is treated as one written in the
     * UDF, and floating point anywhere in it makes the UDF floating point.
     *
     * <p>
     * {@code Math.floorDiv} and {@code Math.floorMod} get one refinement, because they are what a
     * bucketing UDF is made of and their contract is narrow: they throw exactly when the divisor --
     * their last argument, the top of the stack at the call -- is zero. When the data-flow facts
     * show it non-zero on every path to the call they cannot throw, as {@code x % 1000} cannot,
     * and no error channel is needed. "On every path" is the point: a constant on one branch of a
     * conditional says nothing about the other.
     * </p>
     */
    private static String checkJdkCallEffects(MethodInsnNode call, DivisorFacts.Fact divisor, boolean canReportArithmeticErrors, boolean[] needsErrorReporting,
            boolean[] usesFloatingPoint) {
        ComputationEffects.Effects effects = ComputationEffects.ofJdkCall(call);
        if (effects.usesFloatingPoint()) {
            usesFloatingPoint[0] = true;
        }
        if (!effects.mayDivideByZero()) {
            return null;
        }
        boolean floorFamily = call.owner.equals("java/lang/Math") || call.owner.equals("java/lang/StrictMath");
        floorFamily &= call.name.equals("floorDiv") || call.name.equals("floorMod");
        if (floorFamily && divisorIsSafe(divisor, canReportArithmeticErrors)) {
            return null;
        }
        if (canReportArithmeticErrors) {
            needsErrorReporting[0] = true;
            return null;
        }
        return "the UDF calls " + call.owner.replace('/', '.') + "." + call.name + ", whose JDK body divides by a value this analysis cannot prove non-zero; "
                + "Java would raise ArithmeticException and this backend cannot";
    }

    /** The fact about the divisor of {@code insn} -- the value on top of the stack -- or unknown. */
    private static DivisorFacts.Fact divisorOf(DivisorFacts divisors, MethodNode method, AbstractInsnNode insn) {
        return divisors == null ? DivisorFacts.Fact.unknown(1) : divisors.operand(method, insn, 0);
    }

    /**
     * Whether a division by {@code divisor} needs no zero test.
     *
     * <p>
     * On the cudf backend, non-zero on every path is enough: TC compiles Java's division through
     * {@code java_idiv}/{@code java_irem}, which give Java's wrapped answer for
     * {@code MIN_VALUE / -1} instead of C++'s undefined behaviour. The TornadoVM runtime backend has
     * no such helper and no error channel, so it keeps the rule it always had: an exact constant
     * other than zero and minus one.
     * </p>
     */
    private static boolean divisorIsSafe(DivisorFacts.Fact divisor, boolean cudfBackend) {
        if (cudfBackend) {
            return divisor.provablyNonZero();
        }
        return divisor.isConstant() && divisor.constant != 0 && divisor.constant != -1;
    }

    /**
     * Integer division and remainder divide by zero on the device without raising anything, so the
     * query would silently get a number where Java raises {@code ArithmeticException}. A divisor
     * the data-flow facts show non-zero on every path needs nothing; any other divisor is compiled
     * with a zero test that raises on a backend that can report it, and refused on one that
     * cannot.
     */
    private static String checkDivision(AbstractInsnNode insn, DivisorFacts.Fact divisor, int op, boolean canReportArithmeticErrors, boolean[] needsErrorReporting) {
        if (op != Opcodes.IDIV && op != Opcodes.IREM && op != Opcodes.LDIV && op != Opcodes.LREM) {
            return null;
        }
        if (divisorIsSafe(divisor, canReportArithmeticErrors)) {
            return null;
        }
        if (canReportArithmeticErrors) {
            // The divisor may be zero on some row, and the backend can say so: compile with error
            // reporting on and let cudf raise for the call, which fails the query the way Java's
            // ArithmeticException would.
            needsErrorReporting[0] = true;
            return null;
        }
        if (divisor.isConstant() && divisor.constant == 0) {
            return "the UDF divides by the constant zero";
        }
        if (divisor.isConstant() && divisor.constant == -1) {
            // Integer.MIN_VALUE / -1 overflows and, in Java, wraps rather than raising; this backend
            // has no helper that guarantees the same, so it is refused rather than reasoned about.
            return "the UDF divides by the constant -1, whose overflow behaviour the TornadoVM runtime backend does not guarantee";
        }
        return "the UDF divides by a value this analysis cannot prove non-zero on every path; Java would raise ArithmeticException and a kernel cannot";
    }

    /**
     * Source-level primitive constant expressions are inlined by javac. A remaining GETSTATIC
     * can read a value set by the executor's class initializer, even when the field is final.
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
                return "the UDF reads static state " + field.name + " at execution time";
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
            // ... or a static whose JDK body the compiler inlines, which is a different thing: the
            // lowering does not replace it at all, it is compiled from java.base's own bytecode.
            if (call.getOpcode() == Opcodes.INVOKESTATIC && SUPPORTED_JDK_STATICS.contains(call.owner + "." + call.name + call.desc)) {
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

    /**
     * The boxes this integration carries, which is exactly the set {@link BoxConversions} can lower.
     *
     * <p>
     * This used to be a second copy of that list. The two drifted apart the moment {@code Boolean}
     * was added to one of them, so it now asks the table itself: a box the lowering cannot replace
     * must not be accepted here, and that is one fact, not two.
     * </p>
     */
    private static boolean isSupportedBox(String internalNameOrDescriptor) {
        return BoxConversions.isSupportedBox(internalNameOrDescriptor);
    }
}
