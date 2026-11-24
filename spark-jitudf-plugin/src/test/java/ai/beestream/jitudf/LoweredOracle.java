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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import org.apache.spark.sql.types.DataType;

/**
 * Runs, on the JVM, exactly what the cudf backend hands to TC -- the resolved, folded and lowered
 * {@code udfBody}, or the extracted LIST element function together with the reduction the
 * executor applies -- so a test can compare it with the registered Java UDF.
 *
 * <p>
 * This is the oracle for every transformation this plugin performs (resolution, field folding,
 * cast erasure, lowering, the LIST split). It deliberately does not share any of that code with
 * the reference side: the reference is the user's own object, called as Spark would call it after
 * shipping it. TC's translation of the lowered body into CUDA is not covered here; that is what the
 * GPU integration test checks.
 * </p>
 */
final class LoweredOracle {

    private LoweredOracle() {
    }

    /** The scalar pipeline, up to and including the lowered body, as a callable method. */
    static Method scalarBody(Object function, List<DataType> inputs, DataType output) throws Exception {
        ResolvedUdf found = UdfResolver.resolve(function).orElseThrow(() -> new AssertionError(UdfResolver.lastFailure()));
        ResolvedUdf folded = ConstantFieldFolding.fold(found);
        EligibilityResult e = UdfEligibilityAnalyzer.analyse(folded, inputs, output, false, true, true, true);
        if (!e.isEligible()) {
            throw new AssertionError("not eligible: " + e.reason());
        }
        CudfDeviceFunctionGenerator.Generated g = CudfDeviceFunctionGenerator.generate(folded, e.inputs(), e.outputKind(), "0123456789abcdef0123");
        Class<?> cls = new GeneratedKernelClassLoader(LoweredOracle.class.getClassLoader()).define(g.className(), g.classFile());
        for (Method m : cls.getDeclaredMethods()) {
            if (m.getName().equals(CudfDeviceFunctionGenerator.BODY_METHOD)) {
                m.setAccessible(true);
                return m;
            }
        }
        throw new AssertionError("no body method");
    }

    /** Invokes {@code body}; an exception is reported as {@code threw:Name}, as {@link #reference} does. */
    static Object run(Method body, Object... args) {
        try {
            Object result = body.invoke(null, args);
            return normalise(result);
        } catch (InvocationTargetException e) {
            return "threw:" + e.getCause().getClass().getSimpleName();
        } catch (IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    /** The extracted element function of a LIST fold, as a callable method. */
    static Method elementFunction(Object function, DataType listType, DataType output) throws Exception {
        ResolvedUdf resolved = ConstantFieldFolding.fold(UdfResolver.resolve(function).orElseThrow());
        EligibilityResult e = UdfEligibilityAnalyzer.analyse(resolved, List.of(listType), output, false, true, true, true);
        if (!e.isEligible()) {
            throw new AssertionError("not eligible: " + e.reason());
        }
        org.objectweb.asm.tree.ClassNode node = new org.objectweb.asm.tree.ClassNode();
        new org.objectweb.asm.ClassReader(resolved.classFile()).accept(node, org.objectweb.asm.ClassReader.SKIP_FRAMES);
        org.objectweb.asm.tree.MethodNode user = node.methods.stream()
                .filter(m -> m.name.equals(resolved.method().getName()) && m.desc.equals(org.objectweb.asm.Type.getMethodDescriptor(resolved.method())))
                .findFirst().orElseThrow();
        boolean isStatic = java.lang.reflect.Modifier.isStatic(resolved.method().getModifiers());
        org.objectweb.asm.tree.MethodNode body = AsmUdfLowering.lower(user, "udfBody", e.inputs(), e.outputKind(), isStatic);
        var split = ListReductionSplit.split(body, e.inputs().get(0).kind(), e.outputKind());
        if (split.isEmpty()) {
            throw new AssertionError("no split: " + ListReductionSplit.lastReason());
        }
        CudfDeviceFunctionGenerator.Generated g = CudfDeviceFunctionGenerator.generateFromElement(split.get().elementMethod(), e.inputs().get(0).kind(), e.outputKind(),
                "fedcba9876543210fedc");
        Class<?> cls = new GeneratedKernelClassLoader(LoweredOracle.class.getClassLoader()).define(g.className(), g.classFile());
        for (Method m : cls.getDeclaredMethods()) {
            if (m.getName().equals(g.bodyMethod())) {
                m.setAccessible(true);
                return m;
            }
        }
        throw new AssertionError("no element method");
    }

    /**
     * What {@link CudfListReductionUdf} computes for one row: a null row fails, otherwise the
     * element function over every element summed with Java's wrapping arithmetic, an empty row 0.
     */
    static Object reduceRow(Method element, PrimitiveKind result, long[] row) {
        if (row == null) {
            return "threw:NullListException";
        }
        long intAcc = 0;
        double fpAcc = 0;
        for (long v : row) {
            Object value;
            try {
                value = element.getParameterTypes()[0] == int.class ? element.invoke(null, (int) v) : element.invoke(null, v);
            } catch (InvocationTargetException e) {
                return "threw:" + e.getCause().getClass().getSimpleName();
            } catch (IllegalAccessException e) {
                throw new AssertionError(e);
            }
            if (result == PrimitiveKind.INT) {
                intAcc = (int) (intAcc + ((Number) value).intValue());
            } else if (result == PrimitiveKind.LONG) {
                intAcc += ((Number) value).longValue();
            } else {
                fpAcc += ((Number) value).doubleValue();
            }
        }
        return switch (result) {
            case INT -> (int) intAcc;
            case LONG -> intAcc;
            case FLOAT -> (float) fpAcc;
            case DOUBLE -> fpAcc;
            case BOOL -> throw new IllegalStateException();
        };
    }

    /** {@code udf} as an executor runs it: after Java serialization, called reflectively. */
    static Object reference(Object udf, Object... args) {
        Object shipped = roundTrip(udf);
        for (Method m : shipped.getClass().getMethods()) {
            if (m.getName().equals("call") && !m.isBridge() && m.getParameterCount() == args.length) {
                try {
                    m.setAccessible(true);
                    return normalise(m.invoke(shipped, args));
                } catch (InvocationTargetException e) {
                    return "threw:" + e.getCause().getClass().getSimpleName();
                } catch (IllegalAccessException e) {
                    throw new AssertionError(e);
                }
            }
        }
        throw new AssertionError("no call method on " + shipped.getClass());
    }

    private static Object normalise(Object value) {
        return value instanceof Boolean b ? (b ? 1 : 0) : value;
    }

    static Object roundTrip(Object udf) {
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
                out.writeObject(udf);
            }
            try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
                return in.readObject();
            }
        } catch (Exception e) {
            throw new AssertionError("the UDF does not survive serialization: " + e, e);
        }
    }
}
