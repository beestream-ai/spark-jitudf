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
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Finds the method a {@code ScalaUDF} actually runs, by following the JVM's own rules from the
 * call Spark makes.
 *
 * <p>
 * What Spark stores in {@code ScalaUDF.function} is never the user's object. Registering a Java
 * {@code UDF2} leaves a Scala lambda there that Spark's {@code ToScalaUDF} built around it;
 * registering a Scala function leaves the lambda itself. Spark calls that function's
 * {@code apply}, and everything after that is ordinary JVM linkage: a lambda runs its
 * implementation method, an {@code invokeinterface UDF2.call(Object,Object)Object} runs whichever
 * method <em>virtual dispatch</em> selects for the receiver's runtime class, and a compiler bridge
 * runs whatever its own {@code invokevirtual} selects in turn.
 * </p>
 *
 * <p>
 * This class follows exactly those steps and nothing else. It used to pick "the" {@code call}
 * declared on the receiver's class, which is not a JVM rule at all: a subclass that adds an
 * overload {@code call(int,int)} to an inherited {@code call(Integer,Integer)} declares exactly one
 * non-bridge {@code call}, and it is not the one Spark runs. Selection is now by exact descriptor
 * from the runtime class upwards (JVMS 5.4.6), so the inherited implementation is found where it
 * lives and the overload is never considered, because its descriptor is not the one being called.
 * </p>
 *
 * <h2>The shapes followed</h2>
 * <ul>
 * <li>A Java UDF object: dispatch {@code call} with the interface's erased descriptor; if that
 * lands on a compiler bridge, follow the bridge's single forwarding call by dispatch again.</li>
 * <li>A serializable lambda (Scala function, or a Java {@code UDFn} lambda) that captures nothing:
 * its static implementation method is the computation.</li>
 * <li>A bound method reference ({@code inner::call}): the referenced method, dispatched on the
 * captured receiver.</li>
 * <li>A serializable lambda that captures exactly one object and whose implementation does nothing
 * but forward its own arguments, in order, to one method of that object and return the result --
 * Spark's adapter, and equally a user's typed forwarder such as {@code x -> inner.call(x)}:
 * resolve the forwarded call on the captured object, recursively. Boxing a primitive argument,
 * casting an argument to the type the target declares, and casting or unboxing the result are
 * accepted as part of forwarding; see {@link #forwardedCall}.</li>
 * </ul>
 *
 * <p>
 * Nothing is guessed. A dispatch that reaches an interface default method, a non-public or static
 * method, an abstract one, a bridge that does more than forward, or a wrapper that computes
 * anything of its own is reported as unresolvable, and the UDF keeps its CPU plan.
 * </p>
 */
public final class UdfResolver {

    private UdfResolver() {
    }

    /** How many wrappers deep resolution follows before giving up. */
    private static final int MAX_DEPTH = 6;

    /** Why a resolution did not work out, for the refusal message. */
    private static final ThreadLocal<String> LAST_FAILURE = new ThreadLocal<>();

    /** The reason the most recent {@link #resolve} on this thread returned empty. */
    public static String lastFailure() {
        String reason = LAST_FAILURE.get();
        return reason == null ? "no reason recorded" : reason;
    }

    /** Thrown inside resolution to unwind with a reason. */
    private static final class Unresolvable extends Exception {
        private static final long serialVersionUID = 1L;

        Unresolvable(String reason) {
            super(reason, null, false, false);
        }
    }

    /**
     * Resolves the user's UDF method, or returns empty with the reason left to the caller to
     * classify as {@link RejectionReason#WRAPPER_EXTRACTION}; {@link #lastFailure()} says what it
     * ran into.
     */
    public static Optional<ResolvedUdf> resolve(Object function) {
        LAST_FAILURE.remove();
        if (function == null) {
            LAST_FAILURE.set("the UDF's function object is null");
            return Optional.empty();
        }
        try {
            return Optional.of(resolveFunction(function, 0));
        } catch (Unresolvable e) {
            LAST_FAILURE.set(e.getMessage());
            return Optional.empty();
        }
    }

    /** What Spark calls on {@code function}: a lambda's implementation, or a Java UDF's {@code call}. */
    private static ResolvedUdf resolveFunction(Object function, int depth) throws Unresolvable {
        if (depth > MAX_DEPTH) {
            throw new Unresolvable("more than " + MAX_DEPTH + " nested wrappers");
        }
        SerializedLambda lambda = serializedLambda(function);
        if (lambda != null) {
            return resolveLambda(function, lambda, depth);
        }
        if (isHiddenOrLambda(function.getClass())) {
            throw new Unresolvable(function.getClass().getName() + " is a lambda that is not serializable, so its body cannot be inspected (and Spark could not ship it)");
        }
        int arity = javaUdfArity(function.getClass());
        if (arity < 0) {
            throw new Unresolvable(function.getClass().getName() + " is neither a serializable lambda nor an implementation of exactly one Spark Java UDF interface");
        }
        return dispatch(function, "call", erasedCallDescriptor(arity), depth);
    }

    /**
     * A serializable lambda: either its implementation is the computation (nothing captured), or it
     * forwards to the one object it captured.
     */
    private static ResolvedUdf resolveLambda(Object function, SerializedLambda lambda, int depth) throws Unresolvable {
        ResolvedUdf target = lambdaTarget(function, lambda, depth);
        // LambdaMetafactory can insert a cast, unbox, or widen before the implementation
        // method. In particular (Integer)->int can reference a method accepting long.
        var signature = java.lang.invoke.MethodType.fromMethodDescriptorString(lambda.getInstantiatedMethodType(), function.getClass().getClassLoader());
        Class<?>[] parameters = target.method().getParameterTypes();
        if (signature.parameterCount() != parameters.length) {
            throw new Unresolvable("lambda arity differs from the selected implementation");
        }
        var invocation = signature.wrap();
        var implementation = java.lang.invoke.MethodType.methodType(target.method().getReturnType(), parameters).wrap();
        for (int i = 0; i < parameters.length; i++) {
            if (!invocation.parameterType(i).isAssignableFrom(implementation.parameterType(i))) {
                throw new Unresolvable("lambda argument adaptation changes the accepted type at position " + (i + 1));
            }
        }
        if (!invocation.returnType().isAssignableFrom(implementation.returnType())) {
            throw new Unresolvable("lambda result requires a conversion after the selected implementation");
        }
        if (lambda.getCapturedArgCount() == 0 && lambda.getImplMethodKind() == MethodHandleInfo.REF_invokeStatic) {
            var sam = java.lang.invoke.MethodType.fromMethodDescriptorString(lambda.getFunctionalInterfaceMethodSignature(), function.getClass().getClassLoader());
            var entry = java.lang.invoke.MethodType.methodType(target.method().getReturnType(), parameters);
            var adapted = entry;
            for (int i = 0; i < parameters.length; i++) {
                if (sam.parameterType(i).isPrimitive()) {
                    // A specialized SAM can implement erased apply(Object) with its own
                    // conversion (Scala, for example, maps null to zero). Only skip that
                    // bridge when Spark has already excluded null from the call.
                    target = target.withNullSensitiveAdapter();
                } else if (parameters[i].isPrimitive()) {
                    adapted = adapted.changeParameterType(i, implementation.parameterType(i));
                }
            }
            if (!entry.equals(adapted)) {
                target = target.withInputUnboxing(adapted.toMethodDescriptorString());
            }
            if ((signature.returnType().isPrimitive() || sam.returnType().isPrimitive()) && !target.method().getReturnType().isPrimitive()) {
                target = target.withNullSensitiveAdapter();
            }
        }
        return target;
    }

    private static ResolvedUdf lambdaTarget(Object function, SerializedLambda lambda, int depth) throws Unresolvable {
        int kind = lambda.getImplMethodKind();
        if ((kind == MethodHandleInfo.REF_invokeVirtual || kind == MethodHandleInfo.REF_invokeInterface) && lambda.getCapturedArgCount() == 1) {
            return resolveBoundMethodReference(lambda, depth);
        }
        if (kind != MethodHandleInfo.REF_invokeStatic) {
            throw new Unresolvable("the lambda's implementation " + lambda.getImplMethodName() + " is not a static method (it captures an enclosing instance)");
        }
        Class<?> implClass = implementationClass(lambda, function.getClass().getClassLoader());
        if (implClass == null) {
            throw new Unresolvable("the lambda's implementation class " + lambda.getImplClass() + " could not be loaded");
        }
        byte[] bytes = classFileOf(implClass);
        if (bytes == null) {
            throw new Unresolvable("the class file of " + implClass.getName() + " could not be read back through its loader");
        }
        MethodNode impl = methodNode(bytes, lambda.getImplMethodName(), lambda.getImplMethodSignature());
        if (impl == null) {
            throw new Unresolvable("no " + lambda.getImplMethodName() + lambda.getImplMethodSignature() + " in " + implClass.getName());
        }
        if (lambda.getCapturedArgCount() == 0) {
            Method implMethod = declaredMethod(implClass, lambda.getImplMethodName(), lambda.getImplMethodSignature());
            if (implMethod == null || !Modifier.isStatic(implMethod.getModifiers())) {
                throw new Unresolvable("no static " + lambda.getImplMethodName() + lambda.getImplMethodSignature() + " in " + implClass.getName());
            }
            return new ResolvedUdf(null, implMethod, implClass, bytes);
        }
        if (lambda.getCapturedArgCount() != 1) {
            throw new Unresolvable("the lambda captures " + lambda.getCapturedArgCount() + " values, which would have to be compiled as part of it");
        }
        Object captured = lambda.getCapturedArg(0);
        if (captured == null) {
            throw new Unresolvable("the lambda captures null");
        }
        Forward forward;
        try {
            forward = forwardedCall(impl, captured);
        } catch (Unresolvable e) {
            throw new Unresolvable("the wrapper " + lambda.getImplMethodName() + " does more than forward to what it captured (" + e.getMessage()
                    + "); compiling the captured object alone would change what is computed");
        }
        ResolvedUdf target = resolveForwardedCall(captured, forward.call, depth + 1);
        String mismatch = forward.typeMismatchWith(target.method());
        if (mismatch != null) {
            throw new Unresolvable(mismatch);
        }
        return forward.nullSensitive ? target.withNullSensitiveAdapter() : target;
    }

    /**
     * A bound method reference, {@code inner::call}: the lambda's implementation is the instance
     * method itself, invoked on the one captured receiver with the lambda's arguments. That is pure
     * forwarding by construction -- there is no body of the lambda's own -- so the method is
     * resolved on the receiver by dispatch, as the {@code invokevirtual} the metafactory emits
     * would. The metafactory's adaptation from the erased interface to the method's own types
     * casts references, and unboxes or boxes primitives; the latter is exact except for null.
     */
    private static ResolvedUdf resolveBoundMethodReference(SerializedLambda lambda, int depth) throws Unresolvable {
        Object receiver = lambda.getCapturedArg(0);
        if (receiver == null) {
            throw new Unresolvable("the method reference is bound to null");
        }
        MethodInsnNode call = new MethodInsnNode(
                lambda.getImplMethodKind() == MethodHandleInfo.REF_invokeInterface ? Opcodes.INVOKEINTERFACE : Opcodes.INVOKEVIRTUAL,
                lambda.getImplClass(), lambda.getImplMethodName(), lambda.getImplMethodSignature());
        ResolvedUdf target = resolveForwardedCall(receiver, call, depth + 1);
        boolean primitive = Type.getReturnType(call.desc).getSort() != Type.OBJECT && Type.getReturnType(call.desc).getSort() != Type.ARRAY;
        for (Type argument : Type.getArgumentTypes(call.desc)) {
            primitive |= argument.getSort() != Type.OBJECT && argument.getSort() != Type.ARRAY;
        }
        return primitive ? target.withNullSensitiveAdapter() : target;
    }

    /** The method {@code call} reaches on {@code receiver}. */
    private static ResolvedUdf resolveForwardedCall(Object receiver, MethodInsnNode call, int depth) throws Unresolvable {
        if (isHiddenOrLambda(receiver.getClass())) {
            // A lambda object implements exactly its functional method; anything else on it is Object's.
            SerializedLambda inner = serializedLambda(receiver);
            if (inner == null) {
                throw new Unresolvable("the captured " + receiver.getClass().getName() + " is a lambda that is not serializable");
            }
            if (!call.name.equals(inner.getFunctionalInterfaceMethodName()) || !call.desc.equals(inner.getFunctionalInterfaceMethodSignature())) {
                throw new Unresolvable("the wrapper calls " + call.name + call.desc + " on a lambda whose functional method is "
                        + inner.getFunctionalInterfaceMethodName() + inner.getFunctionalInterfaceMethodSignature());
            }
            return resolveLambda(receiver, inner, depth);
        }
        if (!isSubtypeOf(receiver.getClass(), call.owner)) {
            throw new Unresolvable("the captured " + receiver.getClass().getName() + " is not a " + call.owner.replace('/', '.'));
        }
        return dispatch(receiver, call.name, call.desc, depth);
    }

    /**
     * Virtual dispatch of {@code name desc} on {@code receiver}'s runtime class, as JVMS 5.4.6
     * selects it, followed through a compiler bridge if that is what is selected.
     */
    private static ResolvedUdf dispatch(Object receiver, String name, String desc, int depth) throws Unresolvable {
        if (depth > MAX_DEPTH) {
            throw new Unresolvable("more than " + MAX_DEPTH + " nested bridges or wrappers");
        }
        Selected selected = select(receiver.getClass(), name, desc);
        if (selected.method.isBridge() || selected.method.isSynthetic()) {
            byte[] bytes = classFileOf(selected.owner);
            MethodNode bridge = bytes == null ? null : methodNode(bytes, name, desc);
            if (bridge == null) {
                throw new Unresolvable("the bridge " + selected.owner.getName() + "." + name + desc + " could not be read");
            }
            MethodInsnNode next = bridgeTarget(bridge);
            if (next == null) {
                throw new Unresolvable("the bridge " + selected.owner.getName() + "." + name + desc + " does more than cast its arguments and forward them");
            }
            if (!isSubtypeOf(receiver.getClass(), next.owner)) {
                throw new Unresolvable("the bridge forwards to " + next.owner + ", which the receiver is not");
            }
            return dispatch(receiver, next.name, next.desc, depth + 1);
        }
        byte[] bytes = classFileOf(selected.owner);
        if (bytes == null) {
            throw new Unresolvable("the class file of " + selected.owner.getName() + " could not be read back through its loader");
        }
        return new ResolvedUdf(receiver, selected.method, selected.owner, bytes);
    }

    private record Selected(Class<?> owner, Method method) {
    }

    /**
     * The method {@code invokevirtual}/{@code invokeinterface} of {@code name desc} runs on an
     * instance of {@code runtimeClass}: the first declaration of exactly that name and descriptor
     * walking up the superclass chain.
     *
     * <p>
     * Only the shapes whose outcome is certain are accepted. A {@code private} or {@code static}
     * declaration in the chain, or a package-private one (whose overriding depends on runtime
     * packages), is refused rather than modelled; so is reaching no class declaration at all, which
     * means an interface default method or an abstract method.
     * </p>
     */
    private static Selected select(Class<?> runtimeClass, String name, String desc) throws Unresolvable {
        for (Class<?> c = runtimeClass; c != null; c = c.getSuperclass()) {
            Method found = declaredMethod(c, name, desc);
            if (found == null) {
                continue;
            }
            int mods = found.getModifiers();
            if (Modifier.isStatic(mods) || Modifier.isPrivate(mods)) {
                throw new Unresolvable(c.getName() + "." + name + desc + " is " + (Modifier.isStatic(mods) ? "static" : "private")
                        + ", which this resolver does not model for virtual dispatch");
            }
            if (!Modifier.isPublic(mods) && !Modifier.isProtected(mods)) {
                throw new Unresolvable(c.getName() + "." + name + desc + " is package-private, whose overriding depends on runtime packages");
            }
            if (Modifier.isAbstract(mods)) {
                throw new Unresolvable(c.getName() + "." + name + desc + " is abstract in the receiver's class chain");
            }
            return new Selected(c, found);
        }
        throw new Unresolvable("no class in " + runtimeClass.getName() + "'s superclass chain declares " + name + desc
                + " (it would be an interface default method, which is not followed)");
    }

    /**
     * The call a compiler bridge forwards to, or null if it is not one: {@code aload_0}, each
     * argument loaded in order and optionally cast, one {@code invokevirtual} of a method with the
     * same name, and the result returned (optionally cast).
     */
    private static MethodInsnNode bridgeTarget(MethodNode bridge) {
        List<AbstractInsnNode> code = realInstructions(bridge);
        Type[] arguments = Type.getArgumentTypes(bridge.desc);
        int at = 0;
        if (at >= code.size() || !isLoad(code.get(at++), Opcodes.ALOAD, 0)) {
            return null;
        }
        String[] casts = new String[arguments.length];
        int index = 0;
        int slot = 1;
        for (Type argument : arguments) {
            if (at >= code.size() || !isLoad(code.get(at++), argument.getOpcode(Opcodes.ILOAD), slot)) {
                return null;
            }
            if (at < code.size() && code.get(at).getOpcode() == Opcodes.CHECKCAST) {
                casts[index] = ((TypeInsnNode) code.get(at++)).desc;
            }
            index++;
            slot += argument.getSize();
        }
        if (at >= code.size() || !(code.get(at) instanceof MethodInsnNode call)
                || (call.getOpcode() != Opcodes.INVOKEVIRTUAL && call.getOpcode() != Opcodes.INVOKEINTERFACE) || !call.name.equals(bridge.name)
                || Type.getArgumentTypes(call.desc).length != arguments.length) {
            return null;
        }
        Type[] params = Type.getArgumentTypes(call.desc);
        for (int i = 0; i < params.length; i++) {
            if (casts[i] != null && !casts[i].equals(params[i].getInternalName())) {
                return null;
            }
        }
        at++;
        if (at < code.size() && code.get(at).getOpcode() == Opcodes.CHECKCAST) {
            if (!((TypeInsnNode) code.get(at++)).desc.equals(Type.getReturnType(call.desc).getInternalName())) {
                return null;
            }
        }
        if (at != code.size() - 1 || code.get(at).getOpcode() != Type.getReturnType(bridge.desc).getOpcode(Opcodes.IRETURN)) {
            return null;
        }
        return call;
    }

    /** One forwarding call, and the conversions around it. */
    private static final class Forward {
        final MethodInsnNode call;
        /** The wrapper's declared argument types, after the captured receiver. */
        final Type[] wrapperArguments;
        final String[] argumentCasts;
        /** The type the result is cast to before it is returned, or null. */
        final String resultCast;
        /** Whether a null argument or result would behave differently in the wrapper than in the target. */
        final boolean nullSensitive;

        Forward(MethodInsnNode call, Type[] wrapperArguments, String[] argumentCasts, String resultCast, boolean nullSensitive) {
            this.call = call;
            this.wrapperArguments = wrapperArguments;
            this.argumentCasts = argumentCasts;
            this.resultCast = resultCast;
            this.nullSensitive = nullSensitive;
        }

        /**
         * Why the wrapper's typing would make the CPU fail where the GPU would not, or null.
         *
         * <p>
         * A typed wrapper casts each argument to its declared type before forwarding it. If that
         * type is not the one the target declares, the CPU throws {@code ClassCastException} --
         * either in the wrapper's cast or in the target's bridge -- for values the compiled target
         * would accept. Requiring the two to agree (or the wrapper to be erased, passing the value
         * through untouched) makes the forwarding exact. The same holds for a cast on the result.
         * </p>
         */
        String typeMismatchWith(Method target) {
            Class<?>[] declared = target.getParameterTypes();
            if (declared.length != wrapperArguments.length) {
                return "the wrapper forwards " + wrapperArguments.length + " arguments to a method taking " + declared.length;
            }
            for (int i = 0; i < declared.length; i++) {
                if (argumentCasts[i] != null && !isSubtypeOf(declared[i], argumentCasts[i])) {
                    return "the wrapper casts argument " + (i + 1) + " to " + argumentCasts[i] + ", which can fail before the forwarded call";
                }
                Type wrapper = wrapperArguments[i];
                Type wanted = Type.getType(declared[i]);
                if (wrapper.getSort() == Type.OBJECT && wrapper.getInternalName().equals("java/lang/Object")) {
                    continue;
                }
                if (wrapper.equals(wanted)) {
                    continue;
                }
                if (wrapper.getSort() != Type.OBJECT && wrapper.getSort() != Type.ARRAY && boxOf(wrapper).equals(wanted)) {
                    continue; // a primitive boxed to exactly the target's box
                }
                return "the wrapper's argument " + (i + 1) + " is " + wrapper.getClassName() + " but the method it reaches takes " + declared[i].getName()
                        + "; the CPU would raise ClassCastException where the compiled method would not";
            }
            if (resultCast != null && !resultCast.equals("java/lang/Object") && !resultCast.equals(Type.getInternalName(target.getReturnType()))) {
                return "the wrapper casts the result to " + resultCast.replace('/', '.') + " but the method it reaches returns " + target.getReturnType().getName();
            }
            return null;
        }
    }

    /**
     * Java's UDF interfaces declare checked exceptions. A typed adapter can satisfy that
     * declaration by wrapping checked exceptions while rethrowing runtime exceptions unchanged.
     * Eligible numeric bodies can raise unchecked exceptions only. Interpret the handler with
     * that exception on its stack and require it to rethrow the identical object, without effects.
     */
    private static boolean rethrowsRuntimeException(AbstractInsnNode handler, int limit) {
        var aliases = new java.util.HashSet<Integer>();
        final int empty = 0, exception = 1, trueValue = 2;
        int stack = exception;
        AbstractInsnNode pc = handler;
        for (int steps = 0; pc != null && steps <= limit; steps++) {
            int op = pc.getOpcode();
            if (op == Opcodes.ATHROW) {
                return stack == exception;
            }
            if (op == Opcodes.ASTORE && stack == exception) {
                aliases.add(((VarInsnNode) pc).var);
                stack = empty;
            } else if (op == Opcodes.ALOAD && stack == empty && aliases.contains(((VarInsnNode) pc).var)) {
                stack = exception;
            } else if ((op == Opcodes.INSTANCEOF || op == Opcodes.CHECKCAST) && stack == exception
                    && isSubtypeOf(RuntimeException.class, ((TypeInsnNode) pc).desc)) {
                stack = op == Opcodes.INSTANCEOF ? trueValue : exception;
            } else if ((op == Opcodes.IFNE || op == Opcodes.IFEQ) && stack == trueValue) {
                stack = empty;
                if (op == Opcodes.IFNE) {
                    pc = ((org.objectweb.asm.tree.JumpInsnNode) pc).label;
                    continue;
                }
            } else if (op >= 0) {
                return false;
            }
            pc = pc.getNext();
        }
        return false;
    }

    /**
     * The single call a wrapper's implementation forwards to, if forwarding is all it does.
     *
     * <p>
     * The accepted body is: load the captured receiver (slot 0); load each of the wrapper's own
     * arguments exactly once, in order, each optionally boxed by its exact {@code valueOf} (or
     * Scala's {@code BoxesRunTime.boxToX}) and optionally cast; one {@code invokeinterface} or
     * {@code invokevirtual} on the receiver; the result optionally cast and optionally unboxed by
     * its exact accessor; return. That is what {@code ToScalaUDF} compiles to, what javac emits
     * for {@code x -> inner.call(x)} with a typed or an erased parameter. Scala forwarders that call
     * {@code Predef} conversion methods fall outside this shape. Anything else -- arithmetic, a second call, a reordered,
     * repeated or dropped argument, a field read -- is a computation of the wrapper's own.
     * </p>
     *
     * <p>
     * Boxing an argument or unboxing the result is exact for every non-null value, but not for
     * null: unboxing a null result raises (or, in Scala, yields zero) in the wrapper where the target
     * would have returned the null. The forward is marked null-sensitive and the analysis refuses
     * to compile it over a nullable column.
     * </p>
     */
    private static Forward forwardedCall(MethodNode impl, Object captured) throws Unresolvable {
        Type[] arguments = Type.getArgumentTypes(impl.desc);
        if (arguments.length < 1 || arguments[0].getSort() != Type.OBJECT) {
            throw new Unresolvable("its first argument is not the captured object");
        }
        List<AbstractInsnNode> code = realInstructions(impl);
        if (!impl.tryCatchBlocks.isEmpty()) {
            var handler = impl.tryCatchBlocks.get(0);
            if (impl.tryCatchBlocks.size() != 1 || !("java/lang/Exception".equals(handler.type) || "java/lang/RuntimeException".equals(handler.type))
                    || impl.instructions.indexOf(handler.end) > impl.instructions.indexOf(handler.handler)
                    || !rethrowsRuntimeException(handler.handler, impl.instructions.size())) {
                throw new Unresolvable("an exception handler can change the outcome of the forwarded call");
            }
            int handlerIndex = impl.instructions.indexOf(handler.handler);
            code.removeIf(insn -> impl.instructions.indexOf(insn) >= handlerIndex);
        }
        int at = 0;
        if (at >= code.size() || !isLoad(code.get(at++), Opcodes.ALOAD, 0)) {
            throw new Unresolvable("it does not start by loading what it captured");
        }
        if (at < code.size() && code.get(at) instanceof TypeInsnNode cast && cast.getOpcode() == Opcodes.CHECKCAST) {
            if (!isSubtypeOf(captured.getClass(), cast.desc)) {
                throw new Unresolvable("the cast on the captured receiver can fail");
            }
            at++;
        }
        boolean nullSensitive = false;
        int slot = arguments[0].getSize();
        Type[] own = new Type[arguments.length - 1];
        String[] casts = new String[own.length];
        for (int i = 1; i < arguments.length; i++) {
            Type argument = arguments[i];
            own[i - 1] = argument;
            if (at >= code.size() || !isLoad(code.get(at++), argument.getOpcode(Opcodes.ILOAD), slot)) {
                throw new Unresolvable("it does not pass its argument " + i + " on unchanged and in order");
            }
            slot += argument.getSize();
            if (argument.getSort() != Type.OBJECT && argument.getSort() != Type.ARRAY) {
                if (at >= code.size() || !isBoxing(code.get(at), argument)) {
                    throw new Unresolvable("its primitive argument " + i + " is not boxed exactly before being forwarded");
                }
                at++;
                nullSensitive = true;
            }
            if (at < code.size() && code.get(at) instanceof TypeInsnNode cast && cast.getOpcode() == Opcodes.CHECKCAST) {
                casts[i - 1] = cast.desc;
                at++;
            }
        }
        if (at >= code.size() || !(code.get(at) instanceof MethodInsnNode call)
                || (call.getOpcode() != Opcodes.INVOKEINTERFACE && call.getOpcode() != Opcodes.INVOKEVIRTUAL)
                || Type.getArgumentTypes(call.desc).length != arguments.length - 1) {
            throw new Unresolvable("it does not forward to one instance method of what it captured");
        }
        at++;
        String resultCast = null;
        if (at < code.size() && code.get(at) instanceof TypeInsnNode cast && cast.getOpcode() == Opcodes.CHECKCAST) {
            resultCast = cast.desc;
            at++;
        }
        Type returned = Type.getReturnType(impl.desc);
        if (returned.getSort() != Type.OBJECT && returned.getSort() != Type.ARRAY) {
            if (at >= code.size() || !isUnboxing(code.get(at), returned)) {
                throw new Unresolvable("its primitive result is not unboxed exactly from what was forwarded");
            }
            at++;
            nullSensitive = true;
        }
        if (at != code.size() - 1 || code.get(at).getOpcode() != returned.getOpcode(Opcodes.IRETURN)) {
            throw new Unresolvable("it does more after the forwarded call than return its result");
        }
        if (Type.getReturnType(call.desc).getSort() == Type.VOID) {
            throw new Unresolvable("the forwarded call returns nothing");
        }
        return new Forward(call, own, casts, resultCast, nullSensitive);
    }

    private static boolean isBoxing(AbstractInsnNode insn, Type primitive) {
        if (!(insn instanceof MethodInsnNode call) || call.getOpcode() != Opcodes.INVOKESTATIC) {
            return false;
        }
        String box = boxOf(primitive).getInternalName();
        if (call.owner.equals(box) && call.name.equals("valueOf") && call.desc.equals("(" + primitive.getDescriptor() + ")L" + box + ";")) {
            return true;
        }
        String scalaName = "boxTo" + scalaSuffix(primitive);
        return call.owner.equals("scala/runtime/BoxesRunTime") && call.name.equals(scalaName) && call.desc.equals("(" + primitive.getDescriptor() + ")L" + box + ";");
    }

    private static boolean isUnboxing(AbstractInsnNode insn, Type primitive) {
        if (!(insn instanceof MethodInsnNode call)) {
            return false;
        }
        String box = boxOf(primitive).getInternalName();
        if (call.getOpcode() == Opcodes.INVOKEVIRTUAL && call.owner.equals(box) && call.name.equals(primitive.getClassName() + "Value")
                && call.desc.equals("()" + primitive.getDescriptor())) {
            return true;
        }
        return call.getOpcode() == Opcodes.INVOKESTATIC && call.owner.equals("scala/runtime/BoxesRunTime") && call.name.equals("unboxTo" + scalaSuffix(primitive))
                && call.desc.equals("(Ljava/lang/Object;)" + primitive.getDescriptor());
    }

    private static String scalaSuffix(Type primitive) {
        return switch (primitive.getSort()) {
            case Type.INT -> "Int";
            case Type.LONG -> "Long";
            case Type.FLOAT -> "Float";
            case Type.DOUBLE -> "Double";
            case Type.BOOLEAN -> "Boolean";
            default -> "?";
        };
    }

    private static Type boxOf(Type primitive) {
        return switch (primitive.getSort()) {
            case Type.INT -> Type.getObjectType("java/lang/Integer");
            case Type.LONG -> Type.getObjectType("java/lang/Long");
            case Type.FLOAT -> Type.getObjectType("java/lang/Float");
            case Type.DOUBLE -> Type.getObjectType("java/lang/Double");
            case Type.BOOLEAN -> Type.getObjectType("java/lang/Boolean");
            case Type.SHORT -> Type.getObjectType("java/lang/Short");
            case Type.BYTE -> Type.getObjectType("java/lang/Byte");
            case Type.CHAR -> Type.getObjectType("java/lang/Character");
            default -> primitive;
        };
    }

    private static boolean isLoad(AbstractInsnNode insn, int opcode, int slot) {
        return insn instanceof VarInsnNode load && load.getOpcode() == opcode && load.var == slot;
    }

    private static List<AbstractInsnNode> realInstructions(MethodNode method) {
        List<AbstractInsnNode> code = new ArrayList<>();
        for (AbstractInsnNode insn : method.instructions) {
            if (insn.getOpcode() >= 0) {
                code.add(insn);
            }
        }
        return code;
    }

    private static String erasedCallDescriptor(int arity) {
        return "(" + "Ljava/lang/Object;".repeat(arity) + ")Ljava/lang/Object;";
    }

    /** The {@code n} of the one {@code UDFn} the class implements, or -1 if none or several. */
    private static int javaUdfArity(Class<?> type) {
        int arity = -1;
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            for (Class<?> implemented : allSuperInterfaces(c)) {
                String name = implemented.getName();
                if (name.startsWith("org.apache.spark.sql.api.java.UDF")) {
                    int n;
                    try {
                        n = Integer.parseInt(name.substring("org.apache.spark.sql.api.java.UDF".length()));
                    } catch (NumberFormatException e) {
                        continue;
                    }
                    if (arity >= 0 && arity != n) {
                        return -1;
                    }
                    arity = n;
                }
            }
        }
        return arity;
    }

    private static List<Class<?>> allSuperInterfaces(Class<?> c) {
        List<Class<?>> result = new ArrayList<>();
        java.util.ArrayDeque<Class<?>> pending = new java.util.ArrayDeque<>(List.of(c.getInterfaces()));
        while (!pending.isEmpty()) {
            Class<?> next = pending.poll();
            if (!result.contains(next)) {
                result.add(next);
                pending.addAll(List.of(next.getInterfaces()));
            }
        }
        return result;
    }

    private static boolean isSubtypeOf(Class<?> type, String internalName) {
        String name = internalName.replace('/', '.');
        for (Class<?> c = type; c != null; c = c.getSuperclass()) {
            if (c.getName().equals(name)) {
                return true;
            }
            for (Class<?> implemented : allSuperInterfaces(c)) {
                if (implemented.getName().equals(name)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean isHiddenOrLambda(Class<?> type) {
        return type.isHidden() || type.isSynthetic() && type.getName().contains("$$Lambda");
    }

    private static Method declaredMethod(Class<?> type, String name, String desc) {
        for (Method method : type.getDeclaredMethods()) {
            if (method.getName().equals(name) && Type.getMethodDescriptor(method).equals(desc)) {
                return method;
            }
        }
        return null;
    }

    private static MethodNode methodNode(byte[] bytes, String name, String desc) {
        ClassNode node = new ClassNode();
        new ClassReader(bytes).accept(node, ClassReader.SKIP_FRAMES);
        for (MethodNode method : node.methods) {
            if (method.name.equals(name) && method.desc.equals(desc)) {
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

    private static SerializedLambda serializedLambda(Object function) {
        Class<?> clazz = function.getClass();
        if (!isHiddenOrLambda(clazz) && !clazz.getName().toLowerCase(java.util.Locale.ROOT).contains("lambda")) {
            return null;
        }
        try {
            Method writeReplace = clazz.getDeclaredMethod("writeReplace");
            writeReplace.setAccessible(true);
            Object replaced = writeReplace.invoke(function);
            return replaced instanceof SerializedLambda serialized ? serialized : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
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
