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
 * Replaces a UDF's {@code final} primitive instance fields with the values an executor will see.
 *
 * <p>
 * A UDF configured through its constructor is the ordinary way people write one:
 * {@code spark.udf().register("f", new ScoringUdf(weight, cutoff), ...)}. The body then reads
 * {@code this.weight}, which is state, and state is exactly what a kernel does not have -- so until
 * now every such UDF was refused for using {@code this}.
 * </p>
 *
 * <p>
 * It need not be state. The instance exists on the driver while the query is being planned, the
 * field is {@code final}, and so its value is a constant from the kernel's point of view. Reading
 * it here and folding it into the bytecode turns a refused UDF into an ordinary one.
 * </p>
 *
 * <p>
 * This runs before anything else looks at the code, which is what keeps it cheap. The eligibility
 * analysis, the lowering, both kernel generators and the cache key all read the class file this
 * produces, so none of them needs to know that a field was ever involved. The cache key in
 * particular is the right thing to fall out: two instances of the same class with different field
 * values fold to different bytes, so they get different kernels instead of one silently serving the
 * other.
 * </p>
 *
 * <p>
 * Only the exact pattern {@code ALOAD 0; GETFIELD} is folded, only for {@code final} primitive
 * fields of the UDF's own class, and only when the instance is available. Anything else is left
 * alone for the analysis to refuse as it did before.
 * </p>
 *
 * <p>
 * <strong>Which value.</strong> Not the driver object's. Spark ships the UDF to executors by Java
 * serialization, and the object an executor runs is the one that comes out of it. A
 * {@code final transient} field set by the constructor is 9 on the driver and 0 on every executor;
 * a custom {@code readObject} can set a field to something else again; {@code writeReplace} or
 * {@code readResolve} can substitute another object altogether. So the instance is put through a
 * serialization round trip here, exactly the contract Spark uses, and the values are read from the
 * restored copy. The round trip is done twice and a field is folded only when both copies agree,
 * so a {@code readObject} that is not deterministic leaves its field unfolded (and the UDF refused
 * for reading state) rather than frozen at one arbitrary value. If serialization fails, or yields an
 * object of another class, nothing is folded and {@link #executorViewProblem} says why the UDF must
 * not be compiled at all.
 * </p>
 */
public final class ConstantFieldFolding {

    private ConstantFieldFolding() {
    }

    /**
     * {@code udf} with its final primitive instance fields folded, or {@code udf} unchanged when
     * there is nothing to fold or the instance is not available (a static Scala lambda).
     */
    public static ResolvedUdf fold(ResolvedUdf udf) {
        Object instance = udf.instance();
        if (instance == null) {
            return udf;
        }
        Object[] views = executorViews(instance);
        if (views == null) {
            return udf;
        }
        ClassNode node = new ClassNode();
        new ClassReader(udf.classFile()).accept(node, ClassReader.SKIP_FRAMES);

        int folded = 0;
        for (MethodNode method : node.methods) {
            folded += foldIn(method, node.name, views);
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
        return new ResolvedUdf(instance, udf.method(), udf.declaringClass(), writer.toByteArray());
    }

    /**
     * Why an instance-based UDF cannot be compiled from its driver-side class at all, or null.
     *
     * <p>
     * The compiled code is the driver class's {@code call}. If serialization fails, Spark cannot
     * ship the UDF and the CPU plan fails; if it produces an object of another class, executors
     * run that class's {@code call}. Either way the driver's bytecode is not the computation.
     * </p>
     */
    public static String executorViewProblem(Object instance) {
        if (instance == null) {
            return null;
        }
        try {
            Object restored = roundTrip(instance);
            if (restored == null || restored.getClass() != instance.getClass()) {
                return "the UDF object is replaced during serialization (writeReplace/readResolve) by "
                        + (restored == null ? "null" : restored.getClass().getName()) + ", so executors do not run the method that was analysed";
            }
            return null;
        } catch (Exception | LinkageError e) {
            return "the UDF object does not survive the Java serialization Spark ships it with: " + e;
        }
    }

    /** Two independent serialization round trips of {@code instance}, or null if either fails. */
    static Object[] executorViews(Object instance) {
        try {
            Object first = roundTrip(instance);
            Object second = roundTrip(instance);
            if (first == null || second == null || first.getClass() != instance.getClass() || second.getClass() != instance.getClass()) {
                return null;
            }
            return new Object[] { first, second };
        } catch (Exception | LinkageError e) {
            return null;
        }
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

    private static int foldIn(MethodNode method, String owner, Object[] views) {
        int folded = 0;
        AbstractInsnNode insn = method.instructions.getFirst();
        while (insn != null) {
            AbstractInsnNode next = insn.getNext();
            if (insn.getOpcode() == Opcodes.ALOAD && ((VarInsnNode) insn).var == 0
                    && next != null && next.getOpcode() == Opcodes.GETFIELD) {
                FieldInsnNode get = (FieldInsnNode) next;
                Number value = constantValue(owner, get, views);
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

    /** The field's value, when it is one this can safely treat as a constant; null otherwise. */
    private static Number constantValue(String owner, FieldInsnNode get, Object[] views) {
        if (!get.owner.equals(owner)) {
            return null;
        }
        PrimitiveKind kind = kindOf(get.desc);
        if (kind == null) {
            return null;
        }
        try {
            Field field = views[0].getClass().getDeclaredField(get.name);
            if (!Modifier.isFinal(field.getModifiers()) || Modifier.isStatic(field.getModifiers())) {
                // A non-final field can change between planning and execution, and a value read now
                // would be a guess about what it will be later.
                return null;
            }
            field.setAccessible(true);
            Number first = asNumber(field.get(views[0]));
            Number second = asNumber(field.get(views[1]));
            if (first == null || second == null || !sameBits(first, second)) {
                // Deserialization did not settle on one value: not a constant.
                return null;
            }
            return first;
        } catch (ReflectiveOperationException | RuntimeException e) {
            // Inaccessible under a module system or a security policy: leave the field alone and
            // let the analysis refuse the UDF as it would have before.
            return null;
        }
    }

    private static Number asNumber(Object value) {
        if (value instanceof Boolean b) {
            // A boolean is an int on the JVM stack, so ldc 0/1 is exactly what getfield pushed.
            return b ? 1 : 0;
        }
        return value instanceof Number number ? number : null;
    }

    private static boolean sameBits(Number a, Number b) {
        if (a instanceof Double || a instanceof Float) {
            return Double.doubleToRawLongBits(a.doubleValue()) == Double.doubleToRawLongBits(b.doubleValue());
        }
        return a.longValue() == b.longValue() && a.getClass() == b.getClass();
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
