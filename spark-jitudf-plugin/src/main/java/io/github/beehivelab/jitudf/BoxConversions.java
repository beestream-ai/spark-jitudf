/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.util.HashMap;
import java.util.Map;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.MethodInsnNode;

/**
 * What a call on a boxed number actually does, by exact owner, name and descriptor.
 *
 * <p>
 * The Java UDF interfaces are generic, so a UDF's arithmetic arrives wrapped in calls on
 * {@code Integer}, {@code Long}, {@code Float} and {@code Double}. Most of those calls are pure
 * packaging and have to disappear when the body is lowered to primitives. Some of them are not:
 * {@code Integer.shortValue()} narrows, {@code Double.intValue()} truncates toward zero, and
 * {@code Integer.doubleValue()} widens. Deciding by the shape of the method's name -- "ends with
 * Value, therefore unboxing" -- erases those conversions along with the packaging, which changes
 * the answer, and in the truncating case leaves a {@code double} on the stack where an {@code int}
 * is required, so the generated class does not even verify.
 * </p>
 *
 * <p>
 * This table replaces that guess. Every entry is an exact {@code owner.name descriptor} and says
 * precisely what has to be emitted in its place: nothing at all for a true identity, or the JVM
 * conversion opcode that does what the call did. A call not in the table is not lowered and not
 * accepted; {@link UdfEligibilityAnalyzer} refuses the UDF rather than guessing.
 * </p>
 *
 * <p>
 * {@code shortValue}, {@code byteValue} and {@code charValue} are deliberately absent. They are
 * real conversions, but their results are types this integration does not carry end to end, so a
 * UDF using one is refused and keeps its CPU plan.
 * </p>
 */
final class BoxConversions {

    /** What to emit in place of a box call: an opcode, or {@link #IDENTITY} for nothing. */
    static final int IDENTITY = 0;

    private static final Map<String, Integer> OPS = new HashMap<>();

    private static void put(String owner, String name, String descriptor, int opcode) {
        OPS.put(owner + "." + name + descriptor, opcode);
    }

    static {
        // valueOf: the box itself, which is pure packaging in every case.
        put("java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", IDENTITY);
        put("java/lang/Long", "valueOf", "(J)Ljava/lang/Long;", IDENTITY);
        put("java/lang/Float", "valueOf", "(F)Ljava/lang/Float;", IDENTITY);
        put("java/lang/Double", "valueOf", "(D)Ljava/lang/Double;", IDENTITY);

        // The accessor that returns the box's own type: pure unboxing.
        put("java/lang/Integer", "intValue", "()I", IDENTITY);
        put("java/lang/Long", "longValue", "()J", IDENTITY);
        put("java/lang/Float", "floatValue", "()F", IDENTITY);
        put("java/lang/Double", "doubleValue", "()D", IDENTITY);

        // The accessors that convert. Each is unboxing followed by exactly this opcode.
        put("java/lang/Integer", "longValue", "()J", Opcodes.I2L);
        put("java/lang/Integer", "floatValue", "()F", Opcodes.I2F);
        put("java/lang/Integer", "doubleValue", "()D", Opcodes.I2D);

        put("java/lang/Long", "intValue", "()I", Opcodes.L2I);
        put("java/lang/Long", "floatValue", "()F", Opcodes.L2F);
        put("java/lang/Long", "doubleValue", "()D", Opcodes.L2D);

        put("java/lang/Float", "intValue", "()I", Opcodes.F2I);
        put("java/lang/Float", "longValue", "()J", Opcodes.F2L);
        put("java/lang/Float", "doubleValue", "()D", Opcodes.F2D);

        put("java/lang/Double", "intValue", "()I", Opcodes.D2I);
        put("java/lang/Double", "longValue", "()J", Opcodes.D2L);
        put("java/lang/Double", "floatValue", "()F", Opcodes.D2F);
    }

    private BoxConversions() {
    }

    /** Whether {@code owner} is one of the four boxes this integration carries. */
    static boolean isSupportedBox(String internalNameOrDescriptor) {
        String name = internalNameOrDescriptor.startsWith("L")
                ? internalNameOrDescriptor.substring(1, internalNameOrDescriptor.length() - 1)
                : internalNameOrDescriptor;
        return name.equals("java/lang/Integer") || name.equals("java/lang/Long")
                || name.equals("java/lang/Float") || name.equals("java/lang/Double");
    }

    /** Whether this call is one the lowering knows how to replace exactly. */
    static boolean isKnown(MethodInsnNode call) {
        return OPS.containsKey(key(call));
    }

    /**
     * The opcode to emit in place of {@code call}, or {@link #IDENTITY} when it disappears.
     *
     * @throws IllegalStateException
     *         if the call is not in the table, which means the analyzer let through something it
     *         should have refused.
     */
    static int replacementFor(MethodInsnNode call) {
        Integer opcode = OPS.get(key(call));
        if (opcode == null) {
            throw new IllegalStateException("no exact lowering for " + key(call) + "; the analyzer should have refused this UDF");
        }
        return opcode;
    }

    private static String key(MethodInsnNode call) {
        return call.owner + "." + call.name + call.desc;
    }
}
