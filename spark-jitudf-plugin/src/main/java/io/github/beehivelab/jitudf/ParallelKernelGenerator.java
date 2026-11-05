/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.TypeReference;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

/**
 * Builds the class TornadoVM compiles: a {@code static} kernel method with a {@code @Parallel} loop
 * over the rows, calling the user's lowered UDF body once per row.
 *
 * <p>
 * There is one generator, not a catalogue of per-signature templates. The loop is the same shape
 * whatever the UDF's arity and types are -- read one element from each input column, call the body,
 * write one element to the output -- so the signature is built from the descriptor and the loop is
 * emitted around it:
 * </p>
 *
 * <pre>{@code
 * static void kernel(DoubleArray in0, DoubleArray in1, DoubleArray out, int numRows) {
 *     for (@Parallel int i = 0; i < numRows; i++) {
 *         out.set(i, udfBody(in0.get(i), in1.get(i)));
 *     }
 * }
 * }</pre>
 *
 * <p>
 * A {@code LIST} argument contributes two parameters instead of one, and the loop hands the body
 * the row's range into the flattened child rather than an array:
 * </p>
 *
 * <pre>{@code
 * static void kernel(IntArray off0, DoubleArray val0, DoubleArray out, int numRows) {
 *     for (@Parallel int i = 0; i < numRows; i++) {
 *         out.set(i, udfBody(val0, off0.get(i), off0.get(i + 1)));
 *     }
 * }
 * }</pre>
 *
 * <p>
 * The row count is a parameter rather than {@code in0.getSize()} on purpose: the inputs are device
 * views rebound per batch, and a kernel that read its length from them would be recompiled every
 * time the batch size changed.
 * </p>
 */
public final class ParallelKernelGenerator {

    public static final String KERNEL_METHOD = "kernel";
    private static final String BODY_METHOD = "udfBody";
    private static final String PACKAGE = "uk/ac/manchester/tornado/spark/generated/";
    private static final String INT_ARRAY = "uk/ac/manchester/tornado/api/types/arrays/IntArray";

    private static final AtomicInteger COUNTER = new AtomicInteger();

    private ParallelKernelGenerator() {
    }

    /** The generated class, ready to be loaded and handed to a task graph. */
    public record Generated(String className, byte[] classFile, String kernelDescriptor) {
    }

    public static Generated generate(ResolvedUdf udf, List<InputSpec> inputs, PrimitiveKind outputKind) {
        return generate(udf, inputs, outputKind, KernelIdentity.of(udf, inputs, outputKind));
    }

    /**
     * Generates the kernel class, naming it after {@code identity} so that the name is derived
     * from the code it was generated from. Two UDFs that share a binary name but differ in body
     * then produce two differently named classes instead of colliding in the loader.
     */
    public static Generated generate(ResolvedUdf udf, List<InputSpec> inputs, PrimitiveKind outputKind, String identity) {
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

        MethodNode body;
        List<MethodNode> helpers;
        body = AsmUdfLowering.lower(userMethod, BODY_METHOD, inputs, outputKind, isStatic);
        helpers = staticHelpers(source, userMethod);

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES) {
            @Override
            protected String getCommonSuperClass(String first, String second) {
                // Nothing in a generated kernel merges two different reference types; if something
                // did, guessing Object would hide it.
                return first.equals(second) ? first : "java/lang/Object";
            }
        };
        writer.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER, internalName, null, "java/lang/Object", null);

        emitConstructor(writer);
        String kernelDescriptor = emitKernel(writer, internalName, inputs, outputKind);
        // The body and its helpers call each other by the generated class's name now.
        retargetOwner(body, source.name, internalName);
        body.accept(writer);
        for (MethodNode helper : helpers) {
            retargetOwner(helper, source.name, internalName);
            helper.accept(writer);
        }

        writer.visitEnd();
        return new Generated(internalName.replace('/', '.'), writer.toByteArray(), kernelDescriptor);
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

    /** Emits the {@code @Parallel} loop; returns the kernel method's descriptor. */
    private static String emitKernel(ClassWriter writer, String internalName, List<InputSpec> inputs, PrimitiveKind outputKind) {
        StringBuilder descriptor = new StringBuilder("(");
        // Where each input's kernel parameters sit, in order.
        int[] firstSlot = new int[inputs.size()];
        int slot = 0;
        for (int i = 0; i < inputs.size(); i++) {
            InputSpec input = inputs.get(i);
            firstSlot[i] = slot;
            if (input.isList()) {
                // offsets (numRows + 1 ints), then the flattened child values
                descriptor.append("Luk/ac/manchester/tornado/api/types/arrays/IntArray;");
                descriptor.append('L').append(input.kind().arrayInternalName()).append(';');
                slot += 2;
            } else {
                descriptor.append('L').append(input.kind().arrayInternalName()).append(';');
                slot += 1;
            }
        }
        descriptor.append('L').append(outputKind.arrayInternalName()).append(';');
        descriptor.append("I)V");

        int outSlot = slot;
        int rowsSlot = outSlot + 1;
        int indexSlot = rowsSlot + 1;

        MethodVisitor kernel = writer.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, KERNEL_METHOD, descriptor.toString(), null, null);
        kernel.visitCode();

        Label loopHead = new Label();
        Label loopEnd = new Label();

        kernel.visitInsn(Opcodes.ICONST_0);
        kernel.visitVarInsn(Opcodes.ISTORE, indexSlot);

        kernel.visitLabel(loopHead);
        kernel.visitVarInsn(Opcodes.ILOAD, indexSlot);
        kernel.visitVarInsn(Opcodes.ILOAD, rowsSlot);
        kernel.visitJumpInsn(Opcodes.IF_ICMPGE, loopEnd);

        // out.set(i, udfBody(<one or three arguments per input>))
        kernel.visitVarInsn(Opcodes.ALOAD, outSlot);
        kernel.visitVarInsn(Opcodes.ILOAD, indexSlot);
        for (int i = 0; i < inputs.size(); i++) {
            InputSpec input = inputs.get(i);
            if (input.isList()) {
                int offsetsSlot = firstSlot[i];
                int valuesSlot = offsetsSlot + 1;
                // values, offsets.get(i), offsets.get(i + 1)
                kernel.visitVarInsn(Opcodes.ALOAD, valuesSlot);
                kernel.visitVarInsn(Opcodes.ALOAD, offsetsSlot);
                kernel.visitVarInsn(Opcodes.ILOAD, indexSlot);
                kernel.visitMethodInsn(Opcodes.INVOKEVIRTUAL, INT_ARRAY, "get", "(I)I", false);
                kernel.visitVarInsn(Opcodes.ALOAD, offsetsSlot);
                kernel.visitVarInsn(Opcodes.ILOAD, indexSlot);
                kernel.visitInsn(Opcodes.ICONST_1);
                kernel.visitInsn(Opcodes.IADD);
                kernel.visitMethodInsn(Opcodes.INVOKEVIRTUAL, INT_ARRAY, "get", "(I)I", false);
            } else {
                PrimitiveKind kind = input.kind();
                kernel.visitVarInsn(Opcodes.ALOAD, firstSlot[i]);
                kernel.visitVarInsn(Opcodes.ILOAD, indexSlot);
                kernel.visitMethodInsn(Opcodes.INVOKEVIRTUAL, kind.arrayInternalName(), "get", "(I)" + kind.descriptor(), false);
            }
        }
        kernel.visitMethodInsn(Opcodes.INVOKESTATIC, internalName, BODY_METHOD, AsmUdfLowering.descriptorOf(inputs, outputKind), false);
        kernel.visitMethodInsn(Opcodes.INVOKEVIRTUAL, outputKind.arrayInternalName(), "set", "(I" + outputKind.descriptor() + ")V", false);

        kernel.visitIincInsn(indexSlot, 1);
        kernel.visitJumpInsn(Opcodes.GOTO, loopHead);

        kernel.visitLabel(loopEnd);
        kernel.visitInsn(Opcodes.RETURN);

        // The loop variable, and the @Parallel that tells TornadoVM to spread the rows over threads.
        // The annotation's range has to cover the loop header, which is where TornadoVM looks.
        kernel.visitLocalVariable("i", "I", null, loopHead, loopEnd, indexSlot);
        kernel.visitLocalVariableAnnotation(TypeReference.newTypeReference(TypeReference.LOCAL_VARIABLE).getValue(), null, new Label[] { loopHead }, new Label[] { loopEnd },
                new int[] { indexSlot }, "Luk/ac/manchester/tornado/api/annotations/Parallel;", true).visitEnd();

        kernel.visitMaxs(0, 0);
        kernel.visitEnd();
        return descriptor.toString();
    }

    /**
     * The static helper methods of the UDF's class that the body reaches. They already take and
     * return primitives (the analysis refused anything else), so they are copied as they are, with
     * their owner retargeted to the generated class.
     */
    private static List<MethodNode> staticHelpers(ClassNode source, MethodNode entry) {
        List<MethodNode> helpers = new ArrayList<>();
        for (MethodNode candidate : AsmUdfLowering.reachableHelpers(source.methods, entry, source.name)) {
            if ((candidate.access & Opcodes.ACC_STATIC) == 0) {
                throw new IllegalStateException("the UDF calls the instance method " + candidate.name + ", which the analysis should have refused");
            }
            MethodNode copy = new MethodNode(Opcodes.ASM9, (candidate.access & ~Opcodes.ACC_PUBLIC) | Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, candidate.name, candidate.desc, null, null);
            candidate.accept(copy);
            helpers.add(copy);
        }
        return helpers;
    }

    /** Points calls that named the UDF's class at the generated class instead. */
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
