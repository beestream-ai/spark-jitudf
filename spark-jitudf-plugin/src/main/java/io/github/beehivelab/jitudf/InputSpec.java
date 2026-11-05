/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import java.io.Serializable;
import java.util.Objects;

/**
 * What one of the UDF's arguments is: a single number per row, or a list of them.
 *
 * <p>
 * The two shapes reach the kernel very differently. A scalar column is one buffer of fixed-width
 * values, one per row, so one device view is enough. A cuDF {@code LIST} column is two buffers --
 * offsets, with {@code numRows + 1} entries, and a flattened child holding every row's elements end
 * to end -- so it becomes two views and the kernel reaches row {@code r}'s elements between
 * {@code offsets[r]} and {@code offsets[r+1]}. There is no per-row {@code float[]} anywhere, and
 * nothing is gathered into one: the kernel indexes libcudf's own layout.
 * </p>
 */
public final class InputSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    public enum Shape {
        /** One fixed-width value per row. */
        SCALAR,
        /** A variable-length list per row: cuDF {@code LIST}, Spark {@code ArrayType}. */
        LIST
    }

    private final Shape shape;
    private final PrimitiveKind kind;

    private InputSpec(Shape shape, PrimitiveKind kind) {
        this.shape = shape;
        this.kind = kind;
    }

    public static InputSpec scalar(PrimitiveKind kind) {
        return new InputSpec(Shape.SCALAR, kind);
    }

    public static InputSpec list(PrimitiveKind elementKind) {
        return new InputSpec(Shape.LIST, elementKind);
    }

    public Shape shape() {
        return shape;
    }

    public boolean isList() {
        return shape == Shape.LIST;
    }

    /** The value kind for a scalar, or the element kind for a list. */
    public PrimitiveKind kind() {
        return kind;
    }

    /** How many kernel parameters this argument becomes: one, or offsets plus values. */
    public int kernelArity() {
        return shape == Shape.LIST ? 2 : 1;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof InputSpec spec && spec.shape == shape && spec.kind == kind;
    }

    @Override
    public int hashCode() {
        return Objects.hash(shape, kind);
    }

    @Override
    public String toString() {
        return shape == Shape.LIST ? "LIST<" + kind + ">" : kind.toString();
    }
}
