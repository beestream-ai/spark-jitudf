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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * Compiles one {@code java.lang} static through the real driver path, to check it reaches CUDA.
 *
 * <p>
 * The point is to test the claim rather than a proxy for it. {@link UdfEligibilityAnalyzer} admits
 * a short list of JDK statics on the grounds that TC compiles them by inlining {@code java.base}'s
 * own bytecode; this builds the smallest entry class that calls one, runs TC on it exactly as the
 * driver would, and reports what came back. A method that no longer compiles fails the test here
 * instead of failing NVRTC inside a Spark task.
 * </p>
 */
final class JdkStaticProbe {

    private static final AtomicInteger COUNTER = new AtomicInteger();

    private JdkStaticProbe() {
    }

    /** Null when the method compiles to CUDA, or the reason it does not. */
    static String compile(String ownerInternalName, String methodName, String descriptor) {
        Type[] arguments = Type.getArgumentTypes(descriptor);
        Type returned = Type.getReturnType(descriptor);
        PrimitiveKind outputKind = kindOf(returned);
        if (outputKind == null) {
            return "the probe cannot express the return type " + returned;
        }
        List<InputSpec> inputs = new ArrayList<>();
        for (Type argument : arguments) {
            PrimitiveKind kind = kindOf(argument);
            if (kind == null) {
                return "the probe cannot express the parameter type " + argument;
            }
            inputs.add(InputSpec.scalar(kind));
        }

        String internalName = "ai/beestream/jitudf/generated/cudf/Probe$" + COUNTER.incrementAndGet();
        byte[] classFile = entryClass(internalName, ownerInternalName, methodName, descriptor, arguments, outputKind);
        String entryDescriptor = entryDescriptor(inputs, outputKind);

        CudfDeviceFunctionGenerator.Generated generated = new CudfDeviceFunctionGenerator.Generated(internalName.replace('/', '.'), internalName, classFile, entryDescriptor,
                CudfDeviceFunctionGenerator.BODY_METHOD, descriptor);
        TcCudaCompiler.Result result = TcCudaCompiler.compile(generated, inputs, JdkStaticProbe.class.getClassLoader());
        return result.refusal();
    }

    /** {@code static ret udfBody(args...) { return Owner.name(args...); }}, the shape TC compiles. */
    private static byte[] entryClass(String internalName, String owner, String methodName, String descriptor, Type[] arguments, PrimitiveKind outputKind) {
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS | ClassWriter.COMPUTE_FRAMES);
        writer.visit(Opcodes.V21, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER, internalName, null, "java/lang/Object", null);

        Type returned = Type.getReturnType(descriptor);
        MethodVisitor method = writer.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, CudfDeviceFunctionGenerator.BODY_METHOD, descriptor, null, null);
        method.visitCode();
        int slot = 0;
        for (Type argument : arguments) {
            method.visitVarInsn(argument.getOpcode(Opcodes.ILOAD), slot);
            slot += argument.getSize();
        }
        method.visitMethodInsn(Opcodes.INVOKESTATIC, owner, methodName, descriptor, false);
        method.visitInsn(returned.getOpcode(Opcodes.IRETURN));
        method.visitMaxs(0, 0);
        method.visitEnd();
        writer.visitEnd();
        return writer.toByteArray();
    }

    private static String entryDescriptor(List<InputSpec> inputs, PrimitiveKind outputKind) {
        StringBuilder descriptor = new StringBuilder("(L").append(outputKind.arrayInternalName()).append(';');
        for (InputSpec input : inputs) {
            descriptor.append(input.kind().descriptor());
        }
        return descriptor.append(")V").toString();
    }

    private static PrimitiveKind kindOf(Type type) {
        return switch (type.getSort()) {
            case Type.INT -> PrimitiveKind.INT;
            case Type.LONG -> PrimitiveKind.LONG;
            case Type.FLOAT -> PrimitiveKind.FLOAT;
            case Type.DOUBLE -> PrimitiveKind.DOUBLE;
            default -> null;
        };
    }
}
