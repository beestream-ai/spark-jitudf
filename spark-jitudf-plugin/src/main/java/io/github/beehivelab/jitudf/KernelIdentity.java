/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

/**
 * What makes one generated kernel a different kernel from another.
 *
 * <p>
 * The answer has to be the code, not the name it happens to have. A UDF class can be recompiled
 * with a changed body and re-registered under the same fully-qualified name in a new session or a
 * new classloader, which is ordinary in a notebook or a long-lived application. Keyed on the name
 * and signature alone, the second registration finds the first one's entry in a JVM-wide cache and
 * runs the wrong arithmetic, with nothing anywhere reporting a problem.
 * </p>
 *
 * <p>
 * So the key is a digest of the class file the UDF was read from, which covers its entry method and
 * every helper it can reach, together with the method it enters through, the column types it was
 * compiled for and a version for the lowering itself. Digesting the whole class file is broader
 * than strictly necessary -- an unrelated edit elsewhere in the class produces a new kernel -- and
 * that is the safe direction to be wrong in: it costs one extra compilation, where being too narrow
 * costs a wrong answer.
 * </p>
 */
final class KernelIdentity {

    /**
     * Bumped whenever the lowering or the generator changes what it emits for the same input, so
     * that kernels cached by an older version of this code are never reused by a newer one.
     */
    private static final String LOWERING_VERSION = "2";

    private KernelIdentity() {
    }

    /** The cache key for this UDF compiled for these column types. */
    static String of(ResolvedUdf udf, List<InputSpec> inputs, PrimitiveKind outputKind) {
        MessageDigest digest = sha256();
        digest.update(LOWERING_VERSION.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(udf.declaringClass().getName().getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(udf.method().getName().getBytes(StandardCharsets.UTF_8));
        digest.update(org.objectweb.asm.Type.getMethodDescriptor(udf.method()).getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(inputs.toString().getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(String.valueOf(outputKind).getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        // The bytes that actually decide what the kernel computes.
        digest.update(udf.classFile());
        return HexFormat.of().formatHex(digest.digest());
    }

    /** A short form of a key, for use inside a generated class name. */
    static String shortForm(String key) {
        return key.substring(0, 16);
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required to identify generated kernels", e);
        }
    }
}
