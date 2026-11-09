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
import java.io.InputStream;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Holds a generated kernel class and, crucially, serves its bytes back as a <em>resource</em>.
 *
 * <p>
 * TornadoVM's compiler does not work from a loaded {@code Class}: it reads the class file, because
 * it needs the bytecode and the {@code @Parallel} type annotations, neither of which reflection
 * gives it. A class defined with {@code defineClass} alone would load and run on the CPU and then
 * fail to compile, because {@code getResourceAsStream("....class")} would return nothing. So this
 * loader answers for the class twice: once to define it, and once as the resource TornadoVM asks
 * for.
 * </p>
 */
public final class GeneratedKernelClassLoader extends ClassLoader {

    static {
        registerAsParallelCapable();
    }

    private final Map<String, byte[]> classFiles = new ConcurrentHashMap<>();

    public GeneratedKernelClassLoader(ClassLoader parent) {
        super("tornado-generated-kernels", parent);
    }

    /** Defines {@code className} from {@code classFile} and keeps the bytes for the compiler. */
    public synchronized Class<?> define(String className, byte[] classFile) {
        String resource = className.replace('.', '/') + ".class";
        byte[] held = classFiles.get(resource);
        if (held != null && !java.util.Arrays.equals(held, classFile)) {
            // Replacing the bytes of a name that is already defined would leave the loaded class
            // and the bytes TornadoVM reads back through getResourceAsStream describing different
            // code, which compiles one thing and runs another.
            throw new IllegalStateException("refusing to redefine " + className + " with different bytes; "
                    + "a generated kernel's name is derived from its content, so this means two different kernels collided");
        }
        Class<?> existing = findLoadedClass(className);
        if (existing != null) {
            return existing;
        }
        classFiles.put(resource, classFile);
        return defineClass(className, classFile, 0, classFile.length);
    }

    @Override
    public InputStream getResourceAsStream(String name) {
        byte[] bytes = classFiles.get(name);
        if (bytes != null) {
            return new ByteArrayInputStream(bytes);
        }
        return super.getResourceAsStream(name);
    }

    @Override
    public URL getResource(String name) {
        // A generated class has no URL; callers that need the bytes use getResourceAsStream.
        return classFiles.containsKey(name) ? null : super.getResource(name);
    }
}
