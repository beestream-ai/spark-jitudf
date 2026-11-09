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
 * Nothing is guessed. A lambda carrying captured state other than the UDF object, a method that
 * cannot be read back as bytecode, or a shape not listed above is reported as unresolvable rather
 * than resolved to something that might be the wrong method.
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
        return fail("not a Java UDF (" + javaUdfFailure + ") and not a capture-free Scala lambda (" + lastFailure() + ")");
    }

    /**
     * A Java {@code UDF1}..{@code UDF22} implementation, reached either directly or through the
     * lambda Spark's {@code UDFRegistration} wrapped it in.
     */
    private static Optional<ResolvedUdf> resolveJavaUdf(Object function) {
        Object candidate = function;
        if (!isJavaUdf(candidate.getClass())) {
            Object captured = capturedUdfInstance(function);
            if (captured == null) {
                return fail("the wrapper " + function.getClass().getName() + " holds no single captured UDF: " + lastFailure());
            }
            if (!isJavaUdf(captured.getClass())) {
                return fail("the captured object " + captured.getClass().getName() + " does not implement a Spark Java UDF interface");
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
     * A Scala lambda with no captured state: its body is a static synthetic method named in the
     * lambda's {@code SerializedLambda}. A lambda that captured anything is refused, because the
     * captured values are part of the computation and would have to be lowered too.
     */
    private static Optional<ResolvedUdf> resolveScalaLambda(Object function) {
        SerializedLambda lambda = serializedLambda(function);
        if (lambda == null || lambda.getCapturedArgCount() != 0) {
            return Optional.empty();
        }
        String implClassName = lambda.getImplClass().replace('/', '.');
        Class<?> implClass;
        try {
            implClass = Class.forName(implClassName, false, function.getClass().getClassLoader());
        } catch (ClassNotFoundException e) {
            return Optional.empty();
        }
        Method implMethod = null;
        for (Method method : implClass.getDeclaredMethods()) {
            if (method.getName().equals(lambda.getImplMethodName()) && Modifier.isStatic(method.getModifiers())) {
                implMethod = method;
                break;
            }
        }
        if (implMethod == null) {
            return Optional.empty();
        }
        byte[] bytes = classFileOf(implClass);
        if (bytes == null) {
            return Optional.empty();
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

    /** The single captured argument of a lambda, which is where Spark puts a Java UDF instance. */
    private static Object capturedUdfInstance(Object function) {
        SerializedLambda lambda = serializedLambda(function);
        if (lambda != null && lambda.getCapturedArgCount() == 1) {
            return lambda.getCapturedArg(0);
        }
        if (lambda != null) {
            LAST_FAILURE.set("the wrapper lambda captures " + lambda.getCapturedArgCount() + " values, not one");
            return null;
        }
        // A hidden lambda class whose writeReplace is not reachable still has exactly one field
        // holding what it captured, which is what Spark's UDF wrappers are made of.
        java.lang.reflect.Field[] fields = function.getClass().getDeclaredFields();
        if (fields.length != 1) {
            LAST_FAILURE.set("the wrapper " + function.getClass().getName() + " has " + fields.length + " captured fields, not one");
            return null;
        }
        try {
            fields[0].setAccessible(true);
            return fields[0].get(function);
        } catch (ReflectiveOperationException | RuntimeException e) {
            LAST_FAILURE.set("the wrapper's captured field could not be read: " + e);
            return null;
        }
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
