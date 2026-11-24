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
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;

/**
 * Finds the method a {@code ScalaUDF} actually runs.
 *
 * <p>
 * What Spark stores in {@code ScalaUDF.function} is never the user's object. Registering a Java
 * {@code UDF2} leaves a Scala lambda there that Spark's {@code ToScalaUDF} built around it, with
 * the user's instance as its single captured argument; registering a Scala function leaves the
 * lambda itself, whose body is a synthetic static method of the class that declared it. Both are
 * unwrapped here, the same way RAPIDS unwraps a {@code RapidsUDF} out of the first shape.
 * </p>
 *
 * <p>
 * Nothing is guessed. A lambda carrying captured state other than the UDF object, a wrapper whose
 * body does more than forward to the captured UDF, a method that cannot be read back as bytecode,
 * or a shape not listed above is reported as unresolvable rather than resolved to something that
 * might be the wrong method.
 * </p>
 */
public final class UdfResolver {

    private UdfResolver() {
    }

    /** Why a resolution did not work out, for the refusal message. */
    private static final ThreadLocal<String> LAST_FAILURE = new ThreadLocal<>();

    /** The reason the most recent {@link #resolve} on this thread returned empty. */
    public static String lastFailure() {
        String reason = LAST_FAILURE.get();
        return reason == null ? "no reason recorded" : reason;
    }

    private static Optional<ResolvedUdf> fail(String reason) {
        LAST_FAILURE.set(reason);
        return Optional.empty();
    }

    /**
     * Resolves the user's UDF method, or returns empty with the reason left to the caller to
     * classify as {@link RejectionReason#WRAPPER_EXTRACTION}; {@link #lastFailure()} says what it
     * ran into.
     */
    public static Optional<ResolvedUdf> resolve(Object function) {
        LAST_FAILURE.remove();
        if (function == null) {
            return fail("the UDF's function object is null");
        }
        Optional<ResolvedUdf> fromJavaUdf = resolveJavaUdf(function);
        if (fromJavaUdf.isPresent()) {
            return fromJavaUdf;
        }
        String javaUdfFailure = lastFailure();
        Optional<ResolvedUdf> fromLambda = resolveScalaLambda(function);
        if (fromLambda.isPresent()) {
            return fromLambda;
        }
        String lambdaFailure = lastFailure();
        return fail("not a Java UDF (" + javaUdfFailure + ") and not a capture-free Scala lambda (" + lambdaFailure + ")");
    }

    /**
     * A Java {@code UDF1}..{@code UDF22} implementation, reached either directly or through the
     * forwarding adapter Spark's {@code ToScalaUDF} wraps it in.
     *
     * <p>
     * A lambda that captures a Java UDF is <em>not</em> thereby Spark's adapter. A user's Scala or
     * Java function {@code x -> 1 + inner.call(x)} captures exactly one Java UDF too, and treating
     * the captured object as the registered computation compiles {@code inner} in place of the
     * function Spark actually calls. So the captured UDF is unwrapped only when the lambda's
     * implementation method is shown, from its own bytecode, to do nothing but forward its
     * arguments to that UDF's {@code call} and return the result; see {@link #isForwardingAdapter}.
     * </p>
     */
    private static Optional<ResolvedUdf> resolveJavaUdf(Object function) {
        Object candidate = function;
        if (!isJavaUdf(candidate.getClass())) {
            SerializedLambda lambda = serializedLambda(function);
            if (lambda == null) {
                return fail("the wrapper " + function.getClass().getName() + " is not a serializable lambda whose body can be inspected: " + lastFailure());
            }
            if (lambda.getCapturedArgCount() != 1) {
                return fail("the wrapper lambda captures " + lambda.getCapturedArgCount() + " values, not one");
            }
            Object captured = lambda.getCapturedArg(0);
            if (captured == null || !isJavaUdf(captured.getClass())) {
                return fail("the captured object " + (captured == null ? "null" : captured.getClass().getName()) + " does not implement a Spark Java UDF interface");
            }
            String notAdapter = whyNotForwardingAdapter(lambda, function.getClass().getClassLoader(), captured);
            if (notAdapter != null) {
                return fail(notAdapter);
            }
            candidate = captured;
        }
        Class<?> udfClass = candidate.getClass();
        Method call = mostSpecificCall(udfClass);
        if (call == null) {
            return fail(udfClass.getName() + " does not declare exactly one non-bridge call method");
        }
        byte[] bytes = classFileOf(udfClass);
        if (bytes == null) {
            return fail("the class file of " + udfClass.getName() + " could not be read back through its loader");
        }
        return Optional.of(new ResolvedUdf(candidate, call, udfClass, bytes));
    }

    /**
     * Why {@code lambda} is not a pure forwarding adapter around {@code captured}, or null when it is.
     *
     * <p>
     * The evidence is the implementation method's instructions, not its class or method name, so a
     * Spark version that moves the adapter keeps working and a user lambda that happens to share a
     * shape with it is judged by what it does. The accepted body is exactly what
     * {@code ToScalaUDF} compiles to for {@code UDFn} (labels, line numbers and frames aside):
     * </p>
     *
     * <pre>
     * static Object impl(UDFn f, Object a1, ..., Object an) {
     *   aload_0; aload_1; ...; aload_n; invokeinterface UDFn.call(Object x n)Object; areturn
     * }
     * </pre>
     *
     * <p>
     * Anything else -- arithmetic around the call, a cast, a reordered or dropped argument, a call
     * to a different interface -- means the registered function is not {@code captured.call}, and
     * the wrapper is declined rather than resolved to the wrong method.
     * </p>
     */
    static String whyNotForwardingAdapter(SerializedLambda lambda, ClassLoader loader, Object captured) {
        if (lambda.getImplMethodKind() != java.lang.invoke.MethodHandleInfo.REF_invokeStatic) {
            return "the wrapper's implementation is not a static method, so it is not Spark's forwarding adapter";
        }
        org.objectweb.asm.tree.MethodNode impl = implementationMethod(lambda, loader);
        if (impl == null) {
            return "the wrapper's implementation method " + lambda.getImplClass() + "." + lambda.getImplMethodName() + lambda.getImplMethodSignature() + " could not be read";
        }
        org.objectweb.asm.Type[] arguments = org.objectweb.asm.Type.getArgumentTypes(impl.desc);
        int arity = arguments.length - 1;
        if (arity < 0 || arguments[0].getSort() != org.objectweb.asm.Type.OBJECT) {
            return "the wrapper's implementation does not take the captured UDF as its first argument";
        }
        String udfInterface = arguments[0].getInternalName();
        if (!udfInterface.equals("org/apache/spark/sql/api/java/UDF" + arity)) {
            return "the wrapper's implementation takes " + udfInterface + " and " + arity + " values, which is not a Spark UDF" + arity + " adapter";
        }
        if (!implementsInterface(captured.getClass(), udfInterface.replace('/', '.'))) {
            return "the captured UDF does not implement " + udfInterface.replace('/', '.');
        }
        StringBuilder callDescriptor = new StringBuilder("(");
        for (int i = 1; i < arguments.length; i++) {
            if (!arguments[i].getDescriptor().equals("Ljava/lang/Object;")) {
                return "the wrapper's argument " + i + " is " + arguments[i].getClassName() + ", not the erased Object a forwarding adapter takes";
            }
            callDescriptor.append("Ljava/lang/Object;");
        }
        callDescriptor.append(")Ljava/lang/Object;");
        if (!org.objectweb.asm.Type.getReturnType(impl.desc).getDescriptor().equals("Ljava/lang/Object;")) {
            return "the wrapper's implementation does not return the UDF's result unchanged";
        }

        java.util.List<org.objectweb.asm.tree.AbstractInsnNode> code = new java.util.ArrayList<>();
        for (org.objectweb.asm.tree.AbstractInsnNode insn : impl.instructions) {
            if (insn.getOpcode() >= 0) {
                code.add(insn);
            }
        }
        // aload_0 .. aload_n, the call, the return.
        if (code.size() != arity + 3) {
            return "the wrapper does more than forward its arguments to the captured UDF (" + code.size() + " instructions where an adapter has " + (arity + 3)
                    + "); compiling the captured UDF alone would change what is computed";
        }
        for (int i = 0; i <= arity; i++) {
            if (!(code.get(i) instanceof org.objectweb.asm.tree.VarInsnNode load) || load.getOpcode() != org.objectweb.asm.Opcodes.ALOAD || load.var != i) {
                return "the wrapper does not pass its arguments to the captured UDF unchanged and in order";
            }
        }
        if (!(code.get(arity + 1) instanceof org.objectweb.asm.tree.MethodInsnNode call) || call.getOpcode() != org.objectweb.asm.Opcodes.INVOKEINTERFACE
                || !call.owner.equals(udfInterface) || !call.name.equals("call") || !call.desc.equals(callDescriptor.toString())) {
            return "the wrapper does not call " + udfInterface.replace('/', '.') + ".call on what it captured";
        }
        if (code.get(arity + 2).getOpcode() != org.objectweb.asm.Opcodes.ARETURN) {
            return "the wrapper does not return the captured UDF's result unchanged";
        }
        return null;
    }

    private static boolean implementsInterface(Class<?> type, String interfaceName) {
        for (Class<?> implemented : allInterfaces(type)) {
            if (implemented.getName().equals(interfaceName)) {
                return true;
            }
            for (Class<?> parent : allInterfaces(implemented)) {
                if (parent.getName().equals(interfaceName)) {
                    return true;
                }
            }
        }
        return false;
    }

    /** The lambda's implementation method, matched by name <em>and</em> descriptor, as bytecode. */
    private static org.objectweb.asm.tree.MethodNode implementationMethod(SerializedLambda lambda, ClassLoader loader) {
        Class<?> implClass = implementationClass(lambda, loader);
        if (implClass == null) {
            return null;
        }
        byte[] bytes = classFileOf(implClass);
        if (bytes == null) {
            return null;
        }
        org.objectweb.asm.tree.ClassNode node = new org.objectweb.asm.tree.ClassNode();
        new org.objectweb.asm.ClassReader(bytes).accept(node, org.objectweb.asm.ClassReader.SKIP_FRAMES);
        for (org.objectweb.asm.tree.MethodNode method : node.methods) {
            if (method.name.equals(lambda.getImplMethodName()) && method.desc.equals(lambda.getImplMethodSignature())) {
                return method;
            }
        }
        return null;
    }

    private static Class<?> implementationClass(SerializedLambda lambda, ClassLoader loader) {
        try {
            return Class.forName(lambda.getImplClass().replace('/', '.'), false, loader);
        } catch (ClassNotFoundException | LinkageError e) {
            return null;
        }
    }

    /**
     * A Scala lambda with no captured state: its body is a static synthetic method named in the
     * lambda's {@code SerializedLambda}. A lambda that captured anything is refused, because the
     * captured values are part of the computation and would have to be lowered too.
     *
     * <p>
     * The method is matched on its complete handle -- kind, name and descriptor -- not on its name
     * alone. Two static methods can share a name (an overload, or a Scala specialisation), and
     * {@code getDeclaredMethods} returns them in no particular order, so a name match can pick a
     * body the lambda never calls.
     * </p>
     */
    private static Optional<ResolvedUdf> resolveScalaLambda(Object function) {
        SerializedLambda lambda = serializedLambda(function);
        if (lambda == null) {
            return Optional.empty();
        }
        if (lambda.getCapturedArgCount() != 0) {
            return fail("the lambda captures " + lambda.getCapturedArgCount() + " values, which would have to be compiled as part of it");
        }
        if (lambda.getImplMethodKind() != java.lang.invoke.MethodHandleInfo.REF_invokeStatic) {
            return fail("the lambda's implementation is not a static method");
        }
        Class<?> implClass = implementationClass(lambda, function.getClass().getClassLoader());
        if (implClass == null) {
            return fail("the lambda's implementation class " + lambda.getImplClass() + " could not be loaded");
        }
        Method implMethod = null;
        for (Method method : implClass.getDeclaredMethods()) {
            if (method.getName().equals(lambda.getImplMethodName()) && Modifier.isStatic(method.getModifiers())
                    && org.objectweb.asm.Type.getMethodDescriptor(method).equals(lambda.getImplMethodSignature())) {
                implMethod = method;
                break;
            }
        }
        if (implMethod == null) {
            return fail("no static " + lambda.getImplMethodName() + lambda.getImplMethodSignature() + " in " + implClass.getName());
        }
        byte[] bytes = classFileOf(implClass);
        if (bytes == null) {
            return fail("the class file of " + implClass.getName() + " could not be read back through its loader");
        }
        return Optional.of(new ResolvedUdf(null, implMethod, implClass, bytes));
    }

    private static boolean isJavaUdf(Class<?> type) {
        for (Class<?> implemented : allInterfaces(type)) {
            String name = implemented.getName();
            if (name.startsWith("org.apache.spark.sql.api.java.UDF")) {
                return true;
            }
        }
        return false;
    }

    private static java.util.List<Class<?>> allInterfaces(Class<?> type) {
        java.util.List<Class<?>> result = new java.util.ArrayList<>();
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            result.addAll(Arrays.asList(c.getInterfaces()));
        }
        return result;
    }

    /**
     * The {@code call} the user wrote, not the bridge {@code call(Object...)Object} the compiler
     * added: the bridge only casts and delegates, so its bytecode says nothing about the UDF.
     */
    private static Method mostSpecificCall(Class<?> udfClass) {
        Method best = null;
        for (Method method : udfClass.getDeclaredMethods()) {
            if (!method.getName().equals("call") || method.isSynthetic() || method.isBridge()) {
                continue;
            }
            if (best != null) {
                // Two non-bridge call methods: which one Spark invokes is not clear enough to guess.
                return null;
            }
            best = method;
        }
        return best;
    }

    private static SerializedLambda serializedLambda(Object function) {
        Class<?> clazz = function.getClass();
        if (!clazz.getName().toLowerCase(java.util.Locale.ROOT).contains("lambda")) {
            LAST_FAILURE.set(clazz.getName() + " is not a lambda wrapper");
            return null;
        }
        try {
            Method writeReplace = clazz.getDeclaredMethod("writeReplace");
            writeReplace.setAccessible(true);
            Object replaced = writeReplace.invoke(function);
            return replaced instanceof SerializedLambda serialized ? serialized : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            LAST_FAILURE.set("writeReplace on " + clazz.getName() + " failed: " + e);
            return null;
        }
    }

    /** The class file of {@code type}, read back through its own loader. */
    public static byte[] classFileOf(Class<?> type) {
        String resource = type.getName().replace('.', '/') + ".class";
        ClassLoader loader = type.getClassLoader();
        try (InputStream in = loader == null ? ClassLoader.getSystemResourceAsStream(resource) : loader.getResourceAsStream(resource)) {
            return in == null ? null : in.readAllBytes();
        } catch (IOException e) {
            return null;
        }
    }
}
