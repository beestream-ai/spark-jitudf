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

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Replaces a UDF's {@code final} primitive instance fields with the values an executor will see,
 * when -- and only when -- Java serialization fixes those values on the driver.
 *
 * <p>
 * A UDF configured through its constructor is the ordinary way people write one:
 * {@code spark.udf().register("f", new ScoringUdf(weight, cutoff), ...)}. The body then reads
 * {@code this.weight}, which is state, and state is exactly what a kernel does not have. Folding
 * the field into the bytecode as a constant turns a refused UDF into an ordinary one. The cache key
 * falls out right: two instances with different values fold to different bytes and so to different
 * kernels.
 * </p>
 *
 * <h2>Which value an executor sees</h2>
 *
 * <p>
 * Spark ships the UDF by Java serialization; the object an executor runs is the one
 * <em>deserialization in the executor's JVM</em> produces. That is not "the driver's value", and it
 * is not "what a round trip on the driver produces" either: code that runs during deserialization
 * runs on the executor, against the executor's properties, environment and classpath, and two
 * agreeing round trips on the driver say nothing about it. (Codex's counterexample:
 * {@code readResolve} rebuilds the UDF from {@code Integer.getInteger("review.factor", 9)}; the
 * driver folds 9, an executor started with the property at 77 computes {@code x + 77}.)
 * </p>
 *
 * <p>
 * So the criterion is the serialization protocol itself (Java Object Serialization Specification
 * section 3). A field's executor value is fixed on the driver exactly when no code of the UDF's
 * runs on the executor to produce it:
 * </p>
 * <ul>
 * <li>the class that declares it is {@code Serializable} -- a field of a non-serializable
 * superclass is set by that superclass's no-argument constructor, which runs on the executor;</li>
 * <li>no class in the object's chain defines {@code readObjectNoData} or {@code writeReplace},
 * none is {@code Externalizable}, and the declaring class does not narrow its stream with
 * {@code serialPersistentFields} -- each of those is code or a rule this class does not follow;</li>
 * <li>with no hook at all, the stream carries the value the driver wrote, and default
 * deserialization assigns it unchanged: a non-{@code transient} field has the driver instance's
 * value, and a {@code transient} one has its type's default (zero, false), whatever the
 * constructor set;</li>
 * <li>{@code writeObject}, {@code readObject} and {@code readResolve} declared by the class itself
 * are followed only when their bytecode, and that of the constructors they run, is a pure function
 * of the object's own primitive state moved through the stream's field API
 * ({@link #whyNotDeterministic}): then running them on the driver gives what every executor gets,
 * and the values are read from that round trip. A hook that consults anything else -- a system
 * property, the environment, reflection, a static -- leaves the fields unfolded.</li>
 * </ul>
 *
 * <p>
 * Anything else is left unfolded, and the analysis then refuses the field read as before: the UDF
 * stays on the CPU, where the executor's own deserialization decides the value. Only fields are at
 * stake there -- the computation the class's bytecode describes is the same in every JVM.
 * </p>
 *
 * <p>
 * Which <em>class</em> an executor runs is a separate question, answered by
 * {@link #executorViewProblem}: {@code writeReplace} and {@code readResolve} can substitute another
 * object, and then the analysed method is not the one that runs.
 * </p>
 */
public final class ConstantFieldFolding {

    private ConstantFieldFolding() {
    }

    /**
     * {@code udf} with its foldable final primitive instance fields folded, or {@code udf}
     * unchanged when there is nothing to fold or the instance is not available (a static lambda).
     */
    public static ResolvedUdf fold(ResolvedUdf udf) {
        Object instance = udf.instance();
        if (instance == null) {
            return udf;
        }
        Object executorView = executorView(instance);
        if (executorView == null) {
            return udf;
        }
        ClassNode node = new ClassNode();
        new ClassReader(udf.classFile()).accept(node, ClassReader.SKIP_FRAMES);

        int folded = 0;
        for (MethodNode method : node.methods) {
            folded += foldIn(method, node.name, udf.declaringClass(), executorView, executorView != instance);
        }
        if (folded == 0) {
            return udf;
        }
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES) {
            @Override
            protected String getCommonSuperClass(String first, String second) {
                return first.equals(second) ? first : "java/lang/Object";
            }
        };
        node.accept(writer);
        return udf.withClassFile(writer.toByteArray());
    }

    /**
     * Why an instance-based UDF cannot be compiled from its driver-side class at all, or null.
     *
     * <p>
     * The compiled code is a method of the instance's class. If serialization fails, Spark cannot
     * ship the UDF and the CPU plan fails too, so there is nothing to compile. If
     * {@code writeReplace} or {@code readResolve} can hand the executor an object of another class,
     * the executor runs that class's method, and which class it is can depend on the executor
     * (its properties, its classpath). {@code writeReplace} is refused outright. A
     * {@code readResolve} is accepted when its own bytecode proves the class cannot change: the
     * class is {@code final} and every value it returns is {@code this} or a {@code new} instance of
     * that same class. Its fields are then still not folded (see {@link #fold}).
     * </p>
     */
    public static String executorViewProblem(Object instance) {
        if (instance == null) {
            return null;
        }
        Class<?> type = instance.getClass();
        if (findHook(type, "writeReplace", "()Ljava/lang/Object;") != null) {
            return "the UDF object defines writeReplace, so what executors deserialize is whatever that returns, not the class that was analysed";
        }
        Class<?> resolver = findHook(type, "readResolve", "()Ljava/lang/Object;");
        if (resolver != null) {
            String why = whyReadResolveMayChangeClass(type, resolver);
            if (why != null) {
                return "the UDF object defines readResolve, which runs on each executor and " + why + ", so executors may not run the method that was analysed";
            }
        }
        try {
            String problem = serializationProblem(instance);
            if (problem != null) {
                return "the UDF's serialization can affect executor behavior: " + problem;
            }
            roundTrip(instance);
            return null;
        } catch (Exception | LinkageError e) {
            return "the UDF object does not survive the Java serialization Spark ships it with: " + e;
        }
    }

    /**
     * Shipping a CUDA payload bypasses deserialization of the original receiver. Even an unread
     * reference can run a callback (including validation of a cyclic graph) or fail on the worker.
     * Follow the actual default-serialized fields, stopping at primitive data and trusted leaves.
     */
    private static String serializationProblem(Object instance) throws IllegalAccessException {
        var pending = new java.util.ArrayDeque<Object>();
        var seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<Object, Boolean>());
        pending.add(instance);
        while (!pending.isEmpty()) {
            Object object = pending.remove();
            Class<?> type = object.getClass();
            if (!seen.add(object) || type == String.class || type == Boolean.class || type == Character.class
                    || type == Byte.class || type == Short.class || type == Integer.class || type == Long.class || type == Float.class || type == Double.class) {
                continue;
            }
            if (type.isArray()) {
                if (!type.getComponentType().isPrimitive()) {
                    for (Object element : (Object[]) object) {
                        if (element != null) {
                            pending.add(element);
                        }
                    }
                }
                continue;
            }
            String hook = hookThatDecidesFields(type);
            if (hook != null) {
                return type.getName() + " uses " + hook;
            }
            java.util.List<String> hooks = new java.util.ArrayList<>();
            for (String[] spec : HOOKS) {
                if (findHook(type, spec[0], spec[1]) != null) {
                    hooks.add(spec[0] + spec[1]);
                }
            }
            if (!hooks.isEmpty()) {
                String why = whyNotDeterministic(type, hooks);
                if (why != null) {
                    return "cannot establish executor-invariant instance fields or callback behavior: " + why;
                }
            }
            for (Class<?> c = type; c != Object.class; c = c.getSuperclass()) {
                if (!java.io.Serializable.class.isAssignableFrom(c)) {
                    if (!trivialConstructorChain(c)) {
                        return c.getName() + " is initialized by a constructor on the executor";
                    }
                    break; // No state from this superclass is in the stream.
                }
                for (Field field : c.getDeclaredFields()) {
                    int mods = field.getModifiers();
                    if (Modifier.isStatic(mods)) {
                        if (field.getName().equals("serialPersistentFields") && Modifier.isPrivate(mods) && Modifier.isFinal(mods)
                                && field.getType() == java.io.ObjectStreamField[].class) {
                            return c.getName() + " supplies its own persistent field list";
                        }
                    } else if (!field.getType().isPrimitive() && !Modifier.isTransient(mods)) {
                        if (!field.trySetAccessible()) {
                            return "cannot inspect serialized reference " + field;
                        }
                        Object value = field.get(object);
                        if (value != null) {
                            pending.add(value);
                        }
                    }
                }
            }
        }
        return null;
    }

    /** A default superclass constructor that only delegates has no executor-specific effects. */
    private static boolean trivialConstructorChain(Class<?> type) {
        for (Class<?> c = type; c != Object.class; c = c.getSuperclass()) {
            byte[] bytes = UdfResolver.classFileOf(c);
            if (bytes == null) {
                return false;
            }
            ClassNode node = new ClassNode();
            new ClassReader(bytes).accept(node, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
            var constructor = node.methods.stream().filter(m -> m.name.equals("<init>") && m.desc.equals("()V")).findFirst();
            if (constructor.isEmpty()) {
                return false;
            }
            var ops = java.util.Arrays.stream(constructor.get().instructions.toArray()).filter(op -> op.getOpcode() >= 0).toList();
            if (ops.size() != 3 || !(ops.get(0) instanceof VarInsnNode load) || load.getOpcode() != Opcodes.ALOAD || load.var != 0
                    || !(ops.get(1) instanceof org.objectweb.asm.tree.MethodInsnNode init) || init.getOpcode() != Opcodes.INVOKESPECIAL
                    || !init.owner.equals(node.superName) || !init.name.equals("<init>") || !init.desc.equals("()V") || ops.get(2).getOpcode() != Opcodes.RETURN) {
                return false;
            }
        }
        return true;
    }

    /**
     * The serialization hook or rule in {@code type}'s chain that this class never follows when
     * folding, named for a diagnostic; null when there is none. {@code readObject},
     * {@code writeObject} and {@code readResolve} are not listed: they are followed when their
     * bytecode is deterministic (see {@link #executorView}).
     */
    static String hookThatDecidesFields(Class<?> type) {
        if (java.io.Externalizable.class.isAssignableFrom(type)) {
            return "Externalizable";
        }
        for (String[] hook : new String[][] { { "readObjectNoData", "()V" }, { "writeReplace", "()Ljava/lang/Object;" } }) {
            Class<?> owner = findHook(type, hook[0], hook[1]);
            if (owner != null) {
                return owner.getName() + "." + hook[0];
            }
        }
        return null;
    }

    /**
     * The class in {@code type}'s chain declaring a non-static method {@code name desc}, or null.
     * Private hooks apply to their own class only, inherited non-private ones to subclasses; either
     * way their presence anywhere in the chain is what matters here, so any declaration counts.
     */
    private static Class<?> findHook(Class<?> type, String name, String desc) {
        for (Class<?> c = type; c != null && c != Object.class; c = c.getSuperclass()) {
            for (java.lang.reflect.Method method : c.getDeclaredMethods()) {
                if (method.getName().equals(name) && !Modifier.isStatic(method.getModifiers())
                        && org.objectweb.asm.Type.getMethodDescriptor(method).equals(desc)) {
                    return c;
                }
            }
        }
        return null;
    }

    /**
     * Why {@code readResolve} (declared in {@code owner}) might return an object of another class
     * than {@code type}, or null when its bytecode shows it cannot.
     */
    private static String whyReadResolveMayChangeClass(Class<?> type, Class<?> owner) {
        if (!Modifier.isFinal(type.getModifiers())) {
            return "the class is not final, so a new instance of it says nothing about the class of the result";
        }
        byte[] bytes = UdfResolver.classFileOf(owner);
        if (bytes == null) {
            return "its bytecode could not be read";
        }
        ClassNode node = new ClassNode();
        new ClassReader(bytes).accept(node, ClassReader.SKIP_FRAMES);
        MethodNode method = null;
        for (MethodNode candidate : node.methods) {
            if (candidate.name.equals("readResolve") && candidate.desc.equals("()Ljava/lang/Object;")) {
                method = candidate;
            }
        }
        if (method == null || method.instructions.size() == 0) {
            return "its bytecode could not be read";
        }
        String self = org.objectweb.asm.Type.getInternalName(type);
        for (AbstractInsnNode insn : method.instructions) {
            if (insn instanceof VarInsnNode store && store.getOpcode() == Opcodes.ASTORE && store.var == 0) {
                return "reassigns this";
            }
        }
        org.objectweb.asm.tree.analysis.Frame<org.objectweb.asm.tree.analysis.SourceValue>[] frames;
        try {
            frames = new org.objectweb.asm.tree.analysis.Analyzer<>(new org.objectweb.asm.tree.analysis.SourceInterpreter()).analyze(node.name, method);
        } catch (org.objectweb.asm.tree.analysis.AnalyzerException e) {
            return "its bytecode could not be analysed (" + e.getMessage() + ")";
        }
        AbstractInsnNode[] code = method.instructions.toArray();
        for (int i = 0; i < code.length; i++) {
            if (code[i].getOpcode() == Opcodes.ATHROW) {
                continue; // a failed deserialization fails the task on the CPU plan too
            }
            if (code[i].getOpcode() != Opcodes.ARETURN || frames[i] == null) {
                continue;
            }
            org.objectweb.asm.tree.analysis.Frame<org.objectweb.asm.tree.analysis.SourceValue> frame = frames[i];
            if (!onlyThisOrNewSelf(frame.getStack(frame.getStackSize() - 1), method, frames, self, 0)) {
                return "can return something other than this or a new " + type.getSimpleName();
            }
        }
        return null;
    }

    /**
     * Whether every instruction that can have produced {@code value} is {@code aload_0} or a
     * {@code new} of {@code self}, looking through the {@code dup} that a constructor call leaves.
     */
    private static boolean onlyThisOrNewSelf(org.objectweb.asm.tree.analysis.SourceValue value, MethodNode method,
            org.objectweb.asm.tree.analysis.Frame<org.objectweb.asm.tree.analysis.SourceValue>[] frames, String self, int depth) {
        if (depth > 8 || value.insns.isEmpty()) {
            return false;
        }
        for (AbstractInsnNode source : value.insns) {
            if (source instanceof VarInsnNode load && load.getOpcode() == Opcodes.ALOAD && load.var == 0) {
                continue;
            }
            if (source instanceof org.objectweb.asm.tree.TypeInsnNode created && created.getOpcode() == Opcodes.NEW && created.desc.equals(self)) {
                continue;
            }
            if (source.getOpcode() == Opcodes.DUP) {
                org.objectweb.asm.tree.analysis.Frame<org.objectweb.asm.tree.analysis.SourceValue> before = frames[method.instructions.indexOf(source)];
                if (before != null && onlyThisOrNewSelf(before.getStack(before.getStackSize() - 1), method, frames, self, depth + 1)) {
                    continue;
                }
            }
            return false;
        }
        return true;
    }

    /**
     * The object whose fields an executor's copy will have, as far as the driver can know it: the
     * instance itself under default deserialization, or what a deterministic {@code readResolve}
     * builds from it. Null when some field-deciding code is not known to give the same answer in
     * every JVM, in which case nothing is folded.
     */
    private static Object executorView(Object instance) {
        Class<?> type = instance.getClass();
        if (hookThatDecidesFields(type) != null) {
            return null;
        }
        java.util.List<String> hooks = new java.util.ArrayList<>();
        for (String[] hook : HOOKS) {
            Class<?> owner = findHook(type, hook[0], hook[1]);
            if (owner == null) {
                continue;
            }
            if (owner != type) {
                return null; // a hook inherited from, or declared in, another class of the chain
            }
            hooks.add(hook[0] + hook[1]);
        }
        if (hooks.isEmpty()) {
            return instance;
        }
        if (whyNotDeterministic(type, hooks) != null) {
            return null;
        }
        // Every hook is a deterministic function of the object's own primitive state and the
        // stream: running them here -- writeObject as Spark's serialization will, readObject and
        // readResolve as each executor will -- gives exactly what every executor gets.
        try {
            Object resolved = roundTrip(instance);
            return resolved != null && resolved.getClass() == type ? resolved : null;
        } catch (Exception | LinkageError e) {
            return null;
        }
    }

    /** The per-class serialization hooks that are followed when their bytecode is deterministic. */
    private static final String[][] HOOKS = { { "writeObject", "(Ljava/io/ObjectOutputStream;)V" }, { "readObject", "(Ljava/io/ObjectInputStream;)V" },
            { "readResolve", "()Ljava/lang/Object;" } };

    /**
     * The calls on the serialization streams a deterministic hook may make: reading and writing the
     * object's own fields through the stream's field API, and nothing that reaches outside it.
     */
    private static final java.util.Set<String> STREAM_CALLS = java.util.Set.of( //
            "java/io/ObjectOutputStream.defaultWriteObject()V", "java/io/ObjectInputStream.defaultReadObject()V", //
            "java/io/ObjectOutputStream.putFields()Ljava/io/ObjectOutputStream$PutField;", "java/io/ObjectOutputStream.writeFields()V", //
            "java/io/ObjectOutputStream$PutField.put(Ljava/lang/String;I)V", "java/io/ObjectOutputStream$PutField.put(Ljava/lang/String;J)V", //
            "java/io/ObjectOutputStream$PutField.put(Ljava/lang/String;Z)V", "java/io/ObjectOutputStream$PutField.put(Ljava/lang/String;F)V", //
            "java/io/ObjectOutputStream$PutField.put(Ljava/lang/String;D)V");

    /**
     * Why the serialization hooks {@code hooks} of {@code type} might compute different fields in
     * different JVMs (or at different times), or null when they cannot.
     *
     * <p>
     * They cannot when each of them, and every constructor of the class they run, does nothing but
     * arithmetic on the object's own primitive fields and constants, and moves fields through the
     * stream's own field API ({@link #STREAM_CALLS}): no other method calls, no static fields, no
     * reference fields, no arrays, no exceptions. Their inputs are then the object's state -- the
     * driver's values, or the bytes the driver wrote -- and their result is a function of it
     * alone, the same in every JVM. Anything that could consult the environment
     * ({@code Integer.getInteger}, {@code System.getenv}, a clock, reflection, a static initialised
     * per JVM) is a call or a static and is refused here.
     * </p>
     */
    static String whyNotDeterministic(Class<?> type, java.util.List<String> hooks) {
        if (type.getSuperclass() != Object.class) {
            return "the class's state is spread over a class hierarchy";
        }
        byte[] bytes = UdfResolver.classFileOf(type);
        if (bytes == null) {
            return "its bytecode could not be read";
        }
        ClassNode node = new ClassNode();
        new ClassReader(bytes).accept(node, ClassReader.SKIP_FRAMES);
        java.util.ArrayDeque<String> pending = new java.util.ArrayDeque<>(hooks);
        java.util.Set<String> seen = new java.util.HashSet<>(pending);
        while (!pending.isEmpty()) {
            String key = pending.poll();
            MethodNode method = null;
            for (MethodNode candidate : node.methods) {
                if ((candidate.name + candidate.desc).equals(key)) {
                    method = candidate;
                }
            }
            if (method == null) {
                return key + " could not be read";
            }
            if (method.tryCatchBlocks != null && !method.tryCatchBlocks.isEmpty()) {
                return key + " catches exceptions";
            }
            for (AbstractInsnNode insn : method.instructions) {
                int op = insn.getOpcode();
                if (op < 0) {
                    continue;
                }
                if (insn instanceof org.objectweb.asm.tree.MethodInsnNode call) {
                    if (call.getOpcode() == Opcodes.INVOKESPECIAL && call.name.equals("<init>") && call.owner.equals(node.name)) {
                        if (seen.add(call.name + call.desc)) {
                            pending.add(call.name + call.desc);
                        }
                        continue;
                    }
                    if (call.getOpcode() == Opcodes.INVOKESPECIAL && call.name.equals("<init>") && call.owner.equals("java/lang/Object") && call.desc.equals("()V")) {
                        continue;
                    }
                    if (STREAM_CALLS.contains(call.owner + "." + call.name + call.desc)) {
                        continue;
                    }
                    return key + " calls " + call.owner.replace('/', '.') + "." + call.name + ", which may depend on the JVM it runs in";
                }
                if (insn instanceof FieldInsnNode field) {
                    if (op == Opcodes.PUTFIELD && method.name.equals("writeObject")) {
                        return "writeObject mutates the driver's instance while serializing it";
                    }
                    if (op == Opcodes.GETSTATIC || op == Opcodes.PUTSTATIC) {
                        return key + " uses the static field " + field.name + ", which each JVM initialises for itself";
                    }
                    if (!field.owner.equals(node.name) || kindOf(field.desc) == null) {
                        return key + " uses the field " + field.name + ", which is not a primitive field of the class";
                    }
                    continue;
                }
                if (insn instanceof org.objectweb.asm.tree.TypeInsnNode t) {
                    if (op == Opcodes.NEW && t.desc.equals(node.name)) {
                        continue;
                    }
                    return key + " creates or tests " + t.desc.replace('/', '.');
                }
                switch (op) {
                    case Opcodes.INVOKEDYNAMIC, Opcodes.NEWARRAY, Opcodes.ANEWARRAY, Opcodes.MULTIANEWARRAY, Opcodes.MONITORENTER, Opcodes.MONITOREXIT,
                            Opcodes.IALOAD, Opcodes.LALOAD, Opcodes.FALOAD, Opcodes.DALOAD, Opcodes.AALOAD, Opcodes.BALOAD, Opcodes.CALOAD, Opcodes.SALOAD,
                            Opcodes.IASTORE, Opcodes.LASTORE, Opcodes.FASTORE, Opcodes.DASTORE, Opcodes.AASTORE, Opcodes.BASTORE, Opcodes.CASTORE, Opcodes.SASTORE,
                            Opcodes.ARRAYLENGTH, Opcodes.ATHROW:
                        return key + " uses opcode " + op + ", which this check does not admit";
                    case Opcodes.LDC:
                        if (!(((LdcInsnNode) insn).cst instanceof Number) && !(((LdcInsnNode) insn).cst instanceof String)) {
                            return key + " loads a constant that is neither a number nor a field name";
                        }
                        break;
                    default:
                        break;
                }
            }
        }
        return null;
    }

    private static Object roundTrip(Object instance) throws java.io.IOException, ClassNotFoundException {
        java.io.ByteArrayOutputStream bytes = new java.io.ByteArrayOutputStream();
        try (java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(bytes)) {
            out.writeObject(instance);
        }
        ClassLoader loader = instance.getClass().getClassLoader();
        try (java.io.ObjectInputStream in = new java.io.ObjectInputStream(new java.io.ByteArrayInputStream(bytes.toByteArray())) {
            @Override
            protected Class<?> resolveClass(java.io.ObjectStreamClass desc) throws java.io.IOException, ClassNotFoundException {
                try {
                    return Class.forName(desc.getName(), false, loader);
                } catch (ClassNotFoundException e) {
                    return super.resolveClass(desc);
                }
            }
        }) {
            return in.readObject();
        }
    }

    private static int foldIn(MethodNode method, String owner, Class<?> declaringClass, Object instance, boolean resolved) {
        if ((method.access & Opcodes.ACC_STATIC) != 0) {
            return 0; // Slot zero is a parameter in a static helper.
        }
        for (AbstractInsnNode op : method.instructions) {
            if (op instanceof VarInsnNode var && var.var == 0 && var.getOpcode() == Opcodes.ASTORE) {
                return 0; // Slot zero no longer identifies the original receiver.
            }
        }
        int folded = 0;
        AbstractInsnNode insn = method.instructions.getFirst();
        while (insn != null) {
            AbstractInsnNode next = insn.getNext();
            if (insn.getOpcode() == Opcodes.ALOAD && ((VarInsnNode) insn).var == 0
                    && next != null && next.getOpcode() == Opcodes.GETFIELD) {
                FieldInsnNode get = (FieldInsnNode) next;
                Number value = constantValue(owner, declaringClass, get, instance, resolved);
                if (value != null) {
                    AbstractInsnNode after = next.getNext();
                    method.instructions.remove(insn);
                    method.instructions.remove(next);
                    // COMPUTE_FRAMES rebuilds the stack map, so pushing a wider value is safe here.
                    method.instructions.insertBefore(after, new LdcInsnNode(value));
                    folded++;
                    insn = after;
                    continue;
                }
            }
            insn = next;
        }
        return folded;
    }

    /**
     * The field {@code get} reads, resolved as the JVM resolves a field reference (JVMS 5.4.3.2):
     * declared in the named class or inherited from a superclass.
     */
    private static Field resolveField(Class<?> declaringClass, FieldInsnNode get) {
        for (Class<?> c = declaringClass; c != null; c = c.getSuperclass()) {
            for (Field field : c.getDeclaredFields()) {
                if (field.getName().equals(get.name) && org.objectweb.asm.Type.getDescriptor(field.getType()).equals(get.desc)) {
                    return field;
                }
            }
        }
        return null;
    }

    /** The field's executor value, when serialization fixes it on the driver; null otherwise. */
    private static Number constantValue(String owner, Class<?> declaringClass, FieldInsnNode get, Object instance, boolean resolved) {
        if (!get.owner.equals(owner)) {
            return null;
        }
        PrimitiveKind kind = kindOf(get.desc);
        if (kind == null) {
            return null;
        }
        try {
            Field field = resolveField(declaringClass, get);
            if (field == null || !field.getDeclaringClass().isInstance(instance)) {
                return null;
            }
            int mods = field.getModifiers();
            if (!Modifier.isFinal(mods) || Modifier.isStatic(mods)) {
                // A non-final field can change between planning and execution, and a value read now
                // would be a guess about what it will be later.
                return null;
            }
            Class<?> fieldOwner = field.getDeclaringClass();
            if (!java.io.Serializable.class.isAssignableFrom(fieldOwner)) {
                // Set on the executor by the no-argument constructor of the first non-serializable
                // superclass -- code that runs there, not a value the driver wrote.
                return null;
            }
            for (Field declared : fieldOwner.getDeclaredFields()) {
                if (declared.getName().equals("serialPersistentFields") && Modifier.isStatic(declared.getModifiers())) {
                    return null; // the stream's fields are not the class's fields
                }
            }
            if (resolved) {
                // The object readResolve built, whose fields its (pure) constructor set: read them.
                field.setAccessible(true);
                return asNumber(field.get(instance));
            }
            if (Modifier.isTransient(mods)) {
                // Not in the stream at all: default deserialization leaves it at its type's default,
                // whatever the driver's constructor put there.
                return defaultOf(kind);
            }
            field.setAccessible(true);
            return asNumber(field.get(instance));
        } catch (ReflectiveOperationException | RuntimeException e) {
            // Inaccessible under a module system or a security policy: leave the field alone and
            // let the analysis refuse the UDF as it would have before.
            return null;
        }
    }

    private static Number defaultOf(PrimitiveKind kind) {
        return switch (kind) {
            case LONG -> 0L;
            case FLOAT -> 0.0f;
            case DOUBLE -> 0.0d;
            default -> 0;
        };
    }

    private static Number asNumber(Object value) {
        if (value instanceof Boolean b) {
            // A boolean is an int on the JVM stack, so ldc 0/1 is exactly what getfield pushed.
            return b ? 1 : 0;
        }
        return value instanceof Number number ? number : null;
    }

    private static PrimitiveKind kindOf(String descriptor) {
        return switch (descriptor) {
            case "Z" -> PrimitiveKind.BOOL;
            case "I" -> PrimitiveKind.INT;
            case "J" -> PrimitiveKind.LONG;
            case "F" -> PrimitiveKind.FLOAT;
            case "D" -> PrimitiveKind.DOUBLE;
            default -> null;
        };
    }
}
