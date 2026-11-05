/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FrameNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.VarInsnNode;

/**
 * Rewrites the user's UDF method into a {@code static} method over primitives and device arrays,
 * which is the only shape a TornadoVM kernel can call.
 *
 * <p>
 * Three things are in the way of the original. It is an instance method, so slot 0 is a receiver the
 * kernel does not have. The Java UDF interfaces are generic, so its scalar arguments and its result
 * are {@code Integer}, {@code Long}, {@code Float} or {@code Double} and its body is full of
 * {@code doubleValue()} and {@code Double.valueOf()} around the arithmetic that actually matters.
 * And a {@code float[]} argument does not exist on the device at all: a cuDF {@code LIST} column is
 * offsets plus one flattened child, so a per-row array is a <em>range</em> of that child.
 * </p>
 *
 * <p>
 * So: the receiver goes; every local-variable slot moves to account for it and for primitives now
 * being one or two slots wide instead of a reference's one; each boxing and unboxing call becomes
 * nothing at all; each {@code checkcast} to a box goes the same way; {@code areturn} becomes the
 * return of the primitive's kind; and each array argument becomes three parameters -- the child
 * values, and the row's start and end offsets -- with {@code arr.length} becoming
 * {@code end - start} and {@code arr[i]} becoming {@code values.get(start + i)}. Nothing is
 * reinterpreted: this is the same arithmetic on the same values, reaching them where libcudf put
 * them.
 * </p>
 *
 * <p>
 * {@link UdfEligibilityAnalyzer} has already established that nothing else in the method needs
 * rewriting, and {@link ArrayParameters} has already established which array accesses read which
 * argument. Anything this lowering meets that it does not recognise is a disagreement with one of
 * those, and it fails loudly rather than emitting a kernel of uncertain meaning.
 * </p>
 */
final class AsmUdfLowering {

    private AsmUdfLowering() {
    }

    /** Where one original parameter ended up in the lowered method. */
    private record ParameterSlots(InputSpec spec, int primary, int start, int end) {
    }

    /**
     * Lowers {@code source} to a static method over {@code inputs}, returning {@code outputKind}.
     *
     * @param isStatic
     *         whether {@code source} is already static (a Scala lambda's implementation method), in
     *         which case there is no receiver slot to remove.
     */
    static MethodNode lower(MethodNode source, String newName, List<InputSpec> inputs, PrimitiveKind outputKind, boolean isStatic) {
        ArrayParameters arrays = ArrayParameters.of(source, inputs);
        String problem = arrays.check(source);
        if (problem != null) {
            throw new IllegalStateException("the UDF's array arguments cannot be lowered, which the analysis should have caught: " + problem);
        }

        Map<Integer, ParameterSlots> parameters = new HashMap<>();
        int[] slotMap = buildSlotMap(source, inputs, isStatic, parameters);
        Map<Integer, PrimitiveKind> scalarKinds = new HashMap<>();
        for (Map.Entry<Integer, ParameterSlots> entry : parameters.entrySet()) {
            if (!entry.getValue().spec().isList()) {
                scalarKinds.put(entry.getKey(), entry.getValue().spec().kind());
            }
        }

        MethodNode lowered = new MethodNode(Opcodes.ASM9, Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC, newName, descriptorOf(inputs, outputKind), null, null);
        InsnList out = new InsnList();
        for (AbstractInsnNode insn : source.instructions) {
            emit(out, insn, slotMap, scalarKinds, parameters, arrays, outputKind);
        }
        lowered.instructions = out;
        lowered.maxStack = source.maxStack + 6;
        lowered.maxLocals = slotMap.length + 6;
        return lowered;
    }

    /**
     * The lowered method's descriptor: a scalar argument stays one primitive, a list argument
     * becomes {@code (values, start, end)}.
     */
    static String descriptorOf(List<InputSpec> inputs, PrimitiveKind outputKind) {
        StringBuilder descriptor = new StringBuilder("(");
        for (InputSpec input : inputs) {
            if (input.isList()) {
                descriptor.append('L').append(input.kind().arrayInternalName()).append(";II");
            } else {
                descriptor.append(input.kind().descriptor());
            }
        }
        return descriptor.append(')').append(outputKind.descriptor()).toString();
    }

    /**
     * Where each of the original method's local slots ends up.
     *
     * <p>
     * The original frame is {@code [this?][arg0..argN-1][locals...]}, every argument one slot wide
     * because they are all references (boxed numbers or arrays). The lowered one is
     * {@code [lowered arguments...][locals...]}, so the arguments move individually and the locals
     * move as a block, keeping their relative offsets and therefore their widths.
     * </p>
     */
    private static int[] buildSlotMap(MethodNode source, List<InputSpec> inputs, boolean isStatic, Map<Integer, ParameterSlots> parameters) {
        int receiver = isStatic ? 0 : 1;
        Type[] originalArguments = Type.getArgumentTypes(source.desc);
        int firstLocal = receiver;
        for (Type argument : originalArguments) {
            firstLocal += argument.getSize();
        }
        int[] map = new int[Math.max(source.maxLocals, firstLocal) + 1];
        java.util.Arrays.fill(map, -1);

        int newSlot = 0;
        int oldSlot = receiver;
        for (int i = 0; i < originalArguments.length; i++) {
            InputSpec spec = inputs.get(i);
            map[oldSlot] = newSlot;
            if (spec.isList()) {
                parameters.put(oldSlot, new ParameterSlots(spec, newSlot, newSlot + 1, newSlot + 2));
                newSlot += 3;
            } else {
                parameters.put(oldSlot, new ParameterSlots(spec, newSlot, -1, -1));
                newSlot += spec.kind().slots();
            }
            oldSlot += originalArguments[i].getSize();
        }
        int localBase = newSlot;
        for (int slot = firstLocal; slot < map.length; slot++) {
            map[slot] = localBase + (slot - firstLocal);
        }
        return map;
    }

    private static void emit(InsnList out, AbstractInsnNode insn, int[] slotMap, Map<Integer, PrimitiveKind> scalarKinds, Map<Integer, ParameterSlots> parameters, ArrayParameters arrays,
            PrimitiveKind outputKind) {
        if (insn.getOpcode() < 0) {
            // Labels and line numbers are copied as they are; frames are recomputed on write.
            if (!(insn instanceof FrameNode)) {
                out.add(insn);
            }
            return;
        }
        switch (insn.getOpcode()) {
            case Opcodes.ILOAD, Opcodes.LLOAD, Opcodes.FLOAD, Opcodes.DLOAD, //
                    Opcodes.ISTORE, Opcodes.LSTORE, Opcodes.FSTORE, Opcodes.DSTORE -> {
                VarInsnNode variable = (VarInsnNode) insn;
                out.add(new VarInsnNode(variable.getOpcode(), mapSlot(slotMap, variable.var)));
            }
            case Opcodes.ALOAD, Opcodes.ASTORE -> {
                VarInsnNode variable = (VarInsnNode) insn;
                ParameterSlots slots = parameters.get(variable.var);
                if (slots != null && slots.spec().isList()) {
                    // The reference is now the child-values array; the range travels beside it.
                    out.add(new VarInsnNode(variable.getOpcode(), slots.primary()));
                    return;
                }
                PrimitiveKind kind = scalarKinds.get(variable.var);
                if (kind == null) {
                    throw new IllegalStateException("a reference in local slot " + variable.var + " is neither a boxed argument nor a LIST argument");
                }
                out.add(new VarInsnNode(primitiveVariableOpcode(variable.getOpcode(), kind), mapSlot(slotMap, variable.var)));
            }
            case Opcodes.IINC -> {
                IincInsnNode increment = (IincInsnNode) insn;
                out.add(new IincInsnNode(mapSlot(slotMap, increment.var), increment.incr));
            }
            case Opcodes.ARETURN -> out.add(new InsnNode(returnOpcode(outputKind)));
            case Opcodes.CHECKCAST -> {
                TypeInsnNode cast = (TypeInsnNode) insn;
                if (!isBox(cast.desc)) {
                    throw new IllegalStateException("unexpected checkcast to " + cast.desc + " in a method the analysis accepted");
                }
                // The value is already the primitive.
            }
            case Opcodes.ARRAYLENGTH -> {
                ParameterSlots slots = slotsFor(parameters, arrays, insn);
                // The array reference is on the stack and is no longer meaningful: the length is
                // the difference of this row's two offsets.
                out.add(new InsnNode(Opcodes.POP));
                out.add(new VarInsnNode(Opcodes.ILOAD, slots.end()));
                out.add(new VarInsnNode(Opcodes.ILOAD, slots.start()));
                out.add(new InsnNode(Opcodes.ISUB));
            }
            case Opcodes.IALOAD, Opcodes.LALOAD, Opcodes.FALOAD, Opcodes.DALOAD -> {
                ParameterSlots slots = slotsFor(parameters, arrays, insn);
                PrimitiveKind kind = slots.spec().kind();
                // Stack is [values, index]; the element is at the row's start plus that index.
                out.add(new VarInsnNode(Opcodes.ILOAD, slots.start()));
                out.add(new InsnNode(Opcodes.IADD));
                out.add(new MethodInsnNode(Opcodes.INVOKEVIRTUAL, kind.arrayInternalName(), "get", "(I)" + kind.descriptor(), false));
            }
            case Opcodes.INVOKESTATIC, Opcodes.INVOKEVIRTUAL -> {
                MethodInsnNode call = (MethodInsnNode) insn;
                if (BoxConversions.isSupportedBox(call.owner)) {
                    // Exactly what this call did, by owner/name/descriptor: nothing for real
                    // packaging, the matching conversion opcode for a call that also converts.
                    // Erasing the second kind along with the first changes the answer.
                    int replacement = BoxConversions.replacementFor(call);
                    if (replacement != BoxConversions.IDENTITY) {
                        out.add(new InsnNode(replacement));
                    }
                    return;
                }
                out.add(insn);
            }
            case Opcodes.INVOKEINTERFACE -> {
                // Spark hands a Java UDF a scala.collection.Seq for an ArrayType column, so
                // `seq.length()` and `seq.apply(i)` are the same two reads as `arr.length` and
                // `arr[i]` and lower the same way. The cast and unboxing that follow `apply` are
                // removed by the CHECKCAST and INVOKEVIRTUAL cases above.
                MethodInsnNode call = (MethodInsnNode) insn;
                ParameterSlots slots = slotsFor(parameters, arrays, insn);
                PrimitiveKind kind = slots.spec().kind();
                if (call.name.equals("apply")) {
                    out.add(new VarInsnNode(Opcodes.ILOAD, slots.start()));
                    out.add(new InsnNode(Opcodes.IADD));
                    out.add(new MethodInsnNode(Opcodes.INVOKEVIRTUAL, kind.arrayInternalName(), "get", "(I)" + kind.descriptor(), false));
                } else {
                    out.add(new InsnNode(Opcodes.POP));
                    out.add(new VarInsnNode(Opcodes.ILOAD, slots.end()));
                    out.add(new VarInsnNode(Opcodes.ILOAD, slots.start()));
                    out.add(new InsnNode(Opcodes.ISUB));
                }
            }
            default -> out.add(insn);
        }
    }

    private static ParameterSlots slotsFor(Map<Integer, ParameterSlots> parameters, ArrayParameters arrays, AbstractInsnNode insn) {
        if (!arrays.readsAListParameter(insn)) {
            throw new IllegalStateException("an array access the analysis did not attribute to a LIST argument reached the lowering");
        }
        ParameterSlots slots = parameters.get(arrays.slotFor(insn));
        if (slots == null || !slots.spec().isList()) {
            throw new IllegalStateException("an array access attributed to a parameter that is not a LIST argument");
        }
        return slots;
    }

    /** An {@code aload}/{@code astore} of what used to be a boxed argument becomes its primitive. */
    private static int primitiveVariableOpcode(int referenceOpcode, PrimitiveKind kind) {
        boolean load = referenceOpcode == Opcodes.ALOAD;
        return switch (kind) {
            case INT -> load ? Opcodes.ILOAD : Opcodes.ISTORE;
            case LONG -> load ? Opcodes.LLOAD : Opcodes.LSTORE;
            case FLOAT -> load ? Opcodes.FLOAD : Opcodes.FSTORE;
            case DOUBLE -> load ? Opcodes.DLOAD : Opcodes.DSTORE;
        };
    }

    private static int mapSlot(int[] slotMap, int slot) {
        if (slot < 0 || slot >= slotMap.length || slotMap[slot] < 0) {
            throw new IllegalStateException("local slot " + slot + " is outside the method's frame");
        }
        return slotMap[slot];
    }

    private static int returnOpcode(PrimitiveKind kind) {
        return switch (kind) {
            case INT -> Opcodes.IRETURN;
            case LONG -> Opcodes.LRETURN;
            case FLOAT -> Opcodes.FRETURN;
            case DOUBLE -> Opcodes.DRETURN;
        };
    }

    private static boolean isBox(String internalNameOrDescriptor) {
        return BoxConversions.isSupportedBox(internalNameOrDescriptor);
    }

    /** The helper methods of the UDF's class that {@code entry} reaches, in call order. */
    static List<MethodNode> reachableHelpers(List<MethodNode> all, MethodNode entry, String owner) {
        List<MethodNode> helpers = new ArrayList<>();
        java.util.Deque<MethodNode> pending = new java.util.ArrayDeque<>();
        java.util.Set<String> seen = new java.util.HashSet<>();
        pending.add(entry);
        seen.add(entry.name + entry.desc);
        while (!pending.isEmpty()) {
            MethodNode method = pending.poll();
            for (AbstractInsnNode insn : method.instructions) {
                if (insn instanceof MethodInsnNode call && call.owner.equals(owner) && seen.add(call.name + call.desc)) {
                    for (MethodNode candidate : all) {
                        if (candidate.name.equals(call.name) && candidate.desc.equals(call.desc)) {
                            helpers.add(candidate);
                            pending.add(candidate);
                            break;
                        }
                    }
                }
            }
        }
        return helpers;
    }
}
