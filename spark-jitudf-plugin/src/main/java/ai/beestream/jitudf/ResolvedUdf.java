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
    private final boolean nullSensitiveAdapter;
    private final String adapterDescriptor;

    public ResolvedUdf(Object instance, Method method, Class<?> declaringClass, byte[] classFile) {
        this(instance, method, declaringClass, classFile, false, null);
    }

    private ResolvedUdf(Object instance, Method method, Class<?> declaringClass, byte[] classFile, boolean nullSensitiveAdapter, String adapterDescriptor) {
        this.instance = instance;
        this.method = method;
        this.declaringClass = declaringClass;
        this.classFile = classFile;
        this.nullSensitiveAdapter = nullSensitiveAdapter;
        this.adapterDescriptor = adapterDescriptor;
    }

    /**
     * This UDF as reached through a forwarding wrapper that boxes a primitive argument or unboxes
     * the result: exact for every value except null, where the wrapper raises (or yields zero) and
     * the method itself would not. Such a UDF must not be compiled over a nullable column.
     */
    ResolvedUdf withNullSensitiveAdapter() {
        return new ResolvedUdf(instance, method, declaringClass, classFile, true, adapterDescriptor);
    }

    ResolvedUdf withInputUnboxing(String descriptor) {
        return new ResolvedUdf(instance, method, declaringClass, classFile, nullSensitiveAdapter, descriptor);
    }

    String adapterDescriptor() {
        return adapterDescriptor;
    }

    /** Whether a wrapper on the way here treats null differently from the method itself. */
    public boolean nullSensitiveAdapter() {
        return nullSensitiveAdapter;
    }

    /** The same UDF with its class file replaced (by constant folding); everything else carried over. */
    ResolvedUdf withClassFile(byte[] rewritten) {
        return new ResolvedUdf(instance, method, declaringClass, rewritten, nullSensitiveAdapter, adapterDescriptor);
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
