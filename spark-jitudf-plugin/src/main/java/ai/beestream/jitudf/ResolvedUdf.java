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

import java.lang.reflect.Method;

/**
 * The user's own UDF behind whatever Spark wrapped it in: the object, the method it runs, and the
 * class file that method came from.
 */
public final class ResolvedUdf {

    private final Object instance;
    private final Method method;
    private final Class<?> declaringClass;
    private final byte[] classFile;

    public ResolvedUdf(Object instance, Method method, Class<?> declaringClass, byte[] classFile) {
        this.instance = instance;
        this.method = method;
        this.declaringClass = declaringClass;
        this.classFile = classFile;
    }

    /** The UDF object, or {@code null} for a static implementation. */
    public Object instance() {
        return instance;
    }

    public Method method() {
        return method;
    }

    public Class<?> declaringClass() {
        return declaringClass;
    }

    public byte[] classFile() {
        return classFile;
    }

    @Override
    public String toString() {
        return "ResolvedUdf[" + declaringClass.getName() + "." + method.getName() + "]";
    }
}
