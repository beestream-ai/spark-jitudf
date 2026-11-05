/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.io.PrintWriter;
import java.io.StringWriter;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.util.CheckClassAdapter;

/**
 * Checks that a generated kernel class is actually valid, on the driver, before the planner commits
 * the query to it.
 *
 * <p>
 * ASM recomputing stack-map frames is not the same thing as the class being correct. A lowering
 * that leaves a {@code double} where an {@code int} belongs produces frames that describe exactly
 * that, and the class is written out without complaint. The error arrives on an executor as a
 * {@code VerifyError}, which fails the query: there is no mechanism anywhere in this design that
 * turns an executor-side failure back into a CPU plan, and a Spark retry would simply choose the
 * same GPU plan again.
 * </p>
 *
 * <p>
 * So the class is verified and loaded here, while refusing is still free. This catches invalid
 * bytecode and anything that cannot be linked. It deliberately does <em>not</em> claim to catch
 * everything: TornadoVM's own compilation for a specific device happens on the executor and can
 * still fail there for reasons the driver cannot see.
 * </p>
 */
final class GeneratedClassCheck {

    private GeneratedClassCheck() {
    }

    /** Returns null when the class is valid and loadable, or the reason to refuse the UDF. */
    static String verify(ParallelKernelGenerator.Generated generated) {
        StringWriter problems = new StringWriter();
        try {
            CheckClassAdapter.verify(new ClassReader(generated.classFile()), false, new PrintWriter(problems));
        } catch (RuntimeException | LinkageError e) {
            return "the generated kernel does not verify: " + e;
        }
        String text = problems.toString();
        if (!text.isBlank()) {
            return "the generated kernel does not verify: " + firstLine(text);
        }
        try {
            // Loading it proves the constant pool and the references in it resolve, which the
            // verifier above does not do on its own.
            Class<?> loaded = TornadoKernelCache.loader().define(generated.className(), generated.classFile());
            boolean found = false;
            for (var method : loaded.getDeclaredMethods()) {
                if (method.getName().equals(ParallelKernelGenerator.KERNEL_METHOD)) {
                    found = true;
                }
            }
            if (!found) {
                return "the generated kernel has no " + ParallelKernelGenerator.KERNEL_METHOD + " method";
            }
        } catch (LinkageError | RuntimeException e) {
            return "the generated kernel could not be loaded: " + e;
        }
        return null;
    }

    private static String firstLine(String text) {
        int newline = text.indexOf('\n');
        return newline < 0 ? text : text.substring(0, newline);
    }
}
