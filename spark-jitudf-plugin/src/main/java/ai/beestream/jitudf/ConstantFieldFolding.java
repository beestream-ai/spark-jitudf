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
 * Replaces a UDF's {@code final} primitive instance fields with the values they actually hold.
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
        ClassNode node = new ClassNode();
        new ClassReader(udf.classFile()).accept(node, ClassReader.SKIP_FRAMES);

        int folded = 0;
        for (MethodNode method : node.methods) {
            folded += foldIn(method, node.name, instance);
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

    private static int foldIn(MethodNode method, String owner, Object instance) {
        int folded = 0;
        AbstractInsnNode insn = method.instructions.getFirst();
        while (insn != null) {
            AbstractInsnNode next = insn.getNext();
            if (insn.getOpcode() == Opcodes.ALOAD && ((VarInsnNode) insn).var == 0
                    && next != null && next.getOpcode() == Opcodes.GETFIELD) {
                FieldInsnNode get = (FieldInsnNode) next;
                Number value = constantValue(owner, get, instance);
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
    private static Number constantValue(String owner, FieldInsnNode get, Object instance) {
        if (!get.owner.equals(owner)) {
            return null;
        }
        PrimitiveKind kind = kindOf(get.desc);
        if (kind == null) {
            return null;
        }
        try {
            Field field = instance.getClass().getDeclaredField(get.name);
            if (!Modifier.isFinal(field.getModifiers()) || Modifier.isStatic(field.getModifiers())) {
                // A non-final field can change between planning and execution, and a value read now
                // would be a guess about what it will be later.
                return null;
            }
            field.setAccessible(true);
            Object value = field.get(instance);
            return value instanceof Number number ? number : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            // Inaccessible under a module system or a security policy: leave the field alone and
            // let the analysis refuse the UDF as it would have before.
            return null;
        }
    }

    private static PrimitiveKind kindOf(String descriptor) {
        return switch (descriptor) {
            case "I" -> PrimitiveKind.INT;
            case "J" -> PrimitiveKind.LONG;
            case "F" -> PrimitiveKind.FLOAT;
            case "D" -> PrimitiveKind.DOUBLE;
            default -> null;
        };
    }
}
