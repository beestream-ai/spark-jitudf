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

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

/**
 * Builds the class TC compiles into a cuDF {@code transform} device function.
 *
 * <p>
 * This is the counterpart of {@link ParallelKernelGenerator}, and the difference between them is
 * the whole difference between the two architectures. That one wraps the UDF in a {@code @Parallel}
 * loop over the rows, because TornadoVM's runtime launches it and decides the grid. This one does
 * not loop at all: cuDF's generated kernel owns the row loop and calls the UDF once per row, so
 * what it needs is a function for <em>one</em> row.
 * </p>
 *
 * <p>
 * The shape is fixed by cuDF. Its kernel calls {@code GENERIC_TRANSFORM_OP(out_ptrs..., ins...)} --
 * outputs first, as pointers, then inputs by value -- so the generated entry is:
 * </p>
 *
 * <pre>{@code
 * static void cudfEntry(IntArray out, long in0, double in1) {
 *     out.set(0, udfBody(in0, in1));
 * }
 * }</pre>
 *
 * <p>
 * The output is a TornadoVM native array rather than a Java array on purpose: compiled with a
 * {@code DeviceView} fact it is <em>headerless</em>, so TC addresses it from the pointer itself and
 * emits {@code __device__ void f(int *out, long long in0, double in1)} with {@code out[0] = ...}.
 * That is the signature cuDF binds, produced by the compiler rather than by rewriting its output.
 * </p>
 *
 * <p>
 * Only scalar arguments are generated here. A {@code LIST} argument has no path through
 * {@code cudf::transform}, which rejects any input that is not fixed-width or a string, so
 * {@link UdfEligibilityAnalyzer} must refuse one before this is reached; it is checked again here
 * rather than silently emitting something that could not run.
 * </p>
 */
public final class CudfDeviceFunctionGenerator {

    /** The generated entry method's name. The emitted CUDA symbol is chosen separately. */
    public static final String ENTRY_METHOD = "cudfEntry";

    public static final String BODY_METHOD = "udfBody";
    private static final String PACKAGE = "ai/beestream/jitudf/generated/cudf/";

    private CudfDeviceFunctionGenerator() {
    }

    /** The generated class: its binary name, its bytes, and the entry method's descriptor. */
    /**
     * The generated class, and the method inside it that TC compiles.
     *
     * <p>
     * {@code bodyMethod}/{@code bodyDescriptor} name a static method over primitives that returns the
     * UDF's result. That is the shape {@code TcCompiler.compileDeviceFunction} takes: it derives the
     * {@code R *out} parameter cudf passes from the return type, so the output is not a parameter here.
     * {@code entryDescriptor} describes the older {@code cudfEntry} wrapper, which is still emitted for
     * callers that inspect it but is no longer what gets compiled.
     * </p>
     */
    public record Generated(String className, String internalName, byte[] classFile, String entryDescriptor, String bodyMethod, String bodyDescriptor) {
    }

    public static Generated generate(ResolvedUdf udf, List<InputSpec> inputs, PrimitiveKind outputKind, String identity) {
        for (InputSpec input : inputs) {
            if (input.isList()) {
                throw new IllegalArgumentException("a LIST argument cannot be compiled for cudf::transform, which takes only fixed-width and string inputs");
            }
        }

        ClassNode source = new ClassNode();
        new ClassReader(udf.classFile()).accept(source, ClassReader.SKIP_FRAMES);

        String methodDescriptor = org.objectweb.asm.Type.getMethodDescriptor(udf.method());
        MethodNode userMethod = null;
        for (MethodNode method : source.methods) {
            if (method.name.equals(udf.method().getName()) && method.desc.equals(methodDescriptor)) {
                userMethod = method;
                break;
            }
        }
        if (userMethod == null) {
            throw new IllegalStateException("the UDF method " + udf.method() + " is not in the class file it came from");
        }
        boolean isStatic = (userMethod.access & Opcodes.ACC_STATIC) != 0;

        String simpleName = sanitize(udf.declaringClass().getSimpleName()) + "$" + KernelIdentity.shortForm(identity);
        String internalName = PACKAGE + simpleName;

        // The same lowering the runtime architecture uses: the receiver goes, boxing goes, and what
        // is left is a static method over primitives. That method is already the device function's
        // body; only the wrapper around it is new.
        MethodNode body = AsmUdfLowering.lower(userMethod, BODY_METHOD, inputs, outputKind, isStatic);
        List<MethodNode> helpers = AsmUdfLowering.reachableHelpers(source.methods, userMethod, source.name);

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES) {
            @Override
            protected String getCommonSuperClass(String first, String second) {
                return first.equals(second) ? first : "java/lang/Object";
            }
        };
        writer.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER, internalName, null, "java/lang/Object", null);
        emitConstructor(writer);
        String entryDescriptor = emitEntry(writer, internalName, inputs, outputKind);

        retargetOwner(body, source.name, internalName);
        body.accept(writer);
        for (MethodNode helper : helpers) {
            if ((helper.access & Opcodes.ACC_STATIC) == 0) {
                throw new IllegalStateException("the UDF calls the instance method " + helper.name + ", which the analysis should have refused");
            }
            MethodNode copy = new MethodNode(Opcodes.ASM9, (helper.access & ~Opcodes.ACC_PUBLIC) | Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, helper.name, helper.desc, null, null);
            helper.accept(copy);
            retargetOwner(copy, source.name, internalName);
            copy.accept(writer);
        }
        writer.visitEnd();
        return new Generated(internalName.replace('/', '.'), internalName, writer.toByteArray(), entryDescriptor, BODY_METHOD, AsmUdfLowering.descriptorOf(inputs, outputKind));
    }

    public static final String BOXED_BODY_METHOD = "udfNullable";

    /**
     * The class TC compiles for a <em>nullable</em> column, in its null-aware mode.
     *
     * <p>
     * {@link #generate} erases the UDF's boxing on the way to a kernel over values that are known
     * to be present. That is exactly wrong when a column has nulls, because the boxes are where the
     * nulls live: Spark does not short-circuit a null for a Java UDF -- {@code inputPrimitives} is
     * false for every boxed parameter -- so it calls the method with the null and the method's own
     * answer is the SQL answer. Here the method is therefore carried over unchanged apart from
     * losing its receiver, and TC's null-aware mode compiles the boxes into
     * {@code cuda::std::optional}s.
     * </p>
     *
     * <p>
     * There is no {@code cudfEntry} wrapper in this shape. {@code TcCompiler.compileDeviceFunction}
     * derives the {@code cuda::std::optional<R> *out} parameter from the return type, and the
     * function returns an error code rather than nothing, so what is generated is just the method.
     * </p>
     */
    public static Generated generateNullAware(ResolvedUdf udf, List<InputSpec> inputs, String identity) {
        for (InputSpec input : inputs) {
            if (input.isList()) {
                throw new IllegalArgumentException("a LIST argument cannot be compiled for cudf::transform, which takes only fixed-width and string inputs");
            }
        }

        ClassNode source = new ClassNode();
        new ClassReader(udf.classFile()).accept(source, ClassReader.SKIP_FRAMES);

        String methodDescriptor = org.objectweb.asm.Type.getMethodDescriptor(udf.method());
        MethodNode userMethod = null;
        for (MethodNode method : source.methods) {
            if (method.name.equals(udf.method().getName()) && method.desc.equals(methodDescriptor)) {
                userMethod = method;
                break;
            }
        }
        if (userMethod == null) {
            throw new IllegalStateException("the UDF method " + udf.method() + " is not in the class file it came from");
        }
        boolean isStatic = (userMethod.access & Opcodes.ACC_STATIC) != 0;

        String simpleName = sanitize(udf.declaringClass().getSimpleName()) + "$N" + KernelIdentity.shortForm(identity);
        String internalName = PACKAGE + simpleName;

        MethodNode body = BoxedUdfLowering.lower(userMethod, BOXED_BODY_METHOD, isStatic);
        List<MethodNode> helpers = AsmUdfLowering.reachableHelpers(source.methods, userMethod, source.name);

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES) {
            @Override
            protected String getCommonSuperClass(String first, String second) {
                return first.equals(second) ? first : "java/lang/Object";
            }
        };
        writer.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER, internalName, null, "java/lang/Object", null);
        emitConstructor(writer);

        retargetOwner(body, source.name, internalName);
        body.accept(writer);
        for (MethodNode helper : helpers) {
            if ((helper.access & Opcodes.ACC_STATIC) == 0) {
                throw new IllegalStateException("the UDF calls the instance method " + helper.name + ", which the analysis should have refused");
            }
            MethodNode copy = new MethodNode(Opcodes.ASM9, (helper.access & ~Opcodes.ACC_PUBLIC) | Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, helper.name, helper.desc, null, null);
            helper.accept(copy);
            retargetOwner(copy, source.name, internalName);
            copy.accept(writer);
        }
        writer.visitEnd();
        return new Generated(internalName.replace('/', '.'), internalName, writer.toByteArray(), body.desc, BOXED_BODY_METHOD, body.desc);
    }

    /**
     * Builds an entry class around an already-lowered element method, for the list-reduction path.
     *
     * <p>
     * {@link #generate} starts from the user's own method and lowers it; here the body has already
     * been produced by {@link ListReductionSplit}, which extracted it from inside a loop. Everything
     * after that is the same: the same cuDF ABI wrapper, compiled by the same compiler.
     * </p>
     */
    public static Generated generateFromElement(MethodNode element, PrimitiveKind elementKind, PrimitiveKind outputKind, String identity) {
        String simpleName = "Element$" + KernelIdentity.shortForm(identity);
        String internalName = PACKAGE + simpleName;

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES) {
            @Override
            protected String getCommonSuperClass(String first, String second) {
                return first.equals(second) ? first : "java/lang/Object";
            }
        };
        writer.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER, internalName, null, "java/lang/Object", null);
        emitConstructor(writer);

        // out.set(0, udfElement(in0)) -- the same shape as the scalar entry, over one element.
        String entryDescriptor = "(L" + outputKind.arrayInternalName() + ";" + elementKind.descriptor() + ")V";
        MethodVisitor entry = writer.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, ENTRY_METHOD, entryDescriptor, null, null);
        entry.visitCode();
        entry.visitVarInsn(Opcodes.ALOAD, 0);
        entry.visitInsn(Opcodes.ICONST_0);
        entry.visitVarInsn(loadOpcode(elementKind), 1);
        entry.visitMethodInsn(Opcodes.INVOKESTATIC, internalName, element.name, element.desc, false);
        entry.visitMethodInsn(Opcodes.INVOKEVIRTUAL, outputKind.arrayInternalName(), "set", "(I" + outputKind.descriptor() + ")V", false);
        entry.visitInsn(Opcodes.RETURN);
        entry.visitMaxs(0, 0);
        entry.visitEnd();

        element.accept(writer);
        writer.visitEnd();
        return new Generated(internalName.replace('/', '.'), internalName, writer.toByteArray(), entryDescriptor, element.name, element.desc);
    }

    private static void emitConstructor(ClassWriter writer) {
        MethodVisitor init = writer.visitMethod(Opcodes.ACC_PRIVATE, "<init>", "()V", null, null);
        init.visitCode();
        init.visitVarInsn(Opcodes.ALOAD, 0);
        init.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        init.visitInsn(Opcodes.RETURN);
        init.visitMaxs(0, 0);
        init.visitEnd();
    }

    /** Emits {@code out.set(0, udfBody(in0, ..., inN))}; returns the entry's descriptor. */
    private static String emitEntry(ClassWriter writer, String internalName, List<InputSpec> inputs, PrimitiveKind outputKind) {
        StringBuilder descriptor = new StringBuilder("(");
        descriptor.append('L').append(outputKind.arrayInternalName()).append(';');
        for (InputSpec input : inputs) {
            descriptor.append(input.kind().descriptor());
        }
        descriptor.append(")V");

        MethodVisitor entry = writer.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, ENTRY_METHOD, descriptor.toString(), null, null);
        entry.visitCode();

        // Slot 0 is the output array; the inputs follow it, each as wide as its kind.
        entry.visitVarInsn(Opcodes.ALOAD, 0);
        entry.visitInsn(Opcodes.ICONST_0);
        int slot = 1;
        for (InputSpec input : inputs) {
            entry.visitVarInsn(loadOpcode(input.kind()), slot);
            slot += input.kind().slots();
        }
        entry.visitMethodInsn(Opcodes.INVOKESTATIC, internalName, BODY_METHOD, AsmUdfLowering.descriptorOf(inputs, outputKind), false);
        entry.visitMethodInsn(Opcodes.INVOKEVIRTUAL, outputKind.arrayInternalName(), "set", "(I" + outputKind.descriptor() + ")V", false);
        entry.visitInsn(Opcodes.RETURN);
        entry.visitMaxs(0, 0);
        entry.visitEnd();
        return descriptor.toString();
    }

    private static int loadOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.ILOAD;
            case LONG -> Opcodes.LLOAD;
            case FLOAT -> Opcodes.FLOAD;
            case DOUBLE -> Opcodes.DLOAD;
            case BOOL -> Opcodes.ILOAD;
        };
    }

    private static void retargetOwner(MethodNode method, String oldOwner, String newOwner) {
        for (var insn : method.instructions) {
            if (insn instanceof MethodInsnNode call && call.owner.equals(oldOwner)) {
                call.owner = newOwner;
            }
        }
    }

    private static String sanitize(String name) {
        StringBuilder clean = new StringBuilder();
        for (char c : name.toCharArray()) {
            clean.append(Character.isJavaIdentifierPart(c) ? c : '_');
        }
        return clean.length() == 0 ? "Udf" : clean.toString();
    }
}
