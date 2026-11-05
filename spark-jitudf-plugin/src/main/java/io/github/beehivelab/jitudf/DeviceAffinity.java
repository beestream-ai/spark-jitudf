/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); see LICENSE.
 */
package io.github.beehivelab.jitudf;

import ai.rapids.cudf.Cuda;

import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Ties a kernel to the same GPU the cuDF buffers it borrows live on.
 *
 * <p>
 * A device pointer is only meaningful on the device that produced it. Spark gives each executor a
 * GPU, RAPIDS sets that GPU as the CUDA current device, and RMM allocates there; TornadoVM, left
 * alone, uses its own default device, which is device 0 of its first backend. On the single-GPU
 * machine this work was measured on those are the same device and nothing goes wrong, which is
 * exactly why the mismatch has to be refused rather than assumed away: on a two-GPU host an
 * executor assigned GPU 1 would hand a GPU 1 pointer to a kernel launched on GPU 0, and the result
 * would not be an error -- it would be a wrong answer, or a fault far from its cause.
 * </p>
 *
 * <p>
 * So this class makes the mapping explicit: the plan runs on the TornadoVM device whose CUDA
 * ordinal is the ordinal cuDF is using on this thread, and if there is no such device the UDF is
 * refused. Nothing here selects a device for cuDF; it only follows the one already chosen.
 * </p>
 */
final class DeviceAffinity {

    private DeviceAffinity() {
    }

    /** The CUDA device cuDF and RMM are using on this thread. */
    static int currentCudaOrdinal() {
        return Cuda.getDevice();
    }

    /**
     * The TornadoVM device for {@code ordinal}.
     *
     * <p>
     * TornadoVM's CUDA backend enumerates its devices with {@code cuDeviceGet(i)}, so its device
     * {@code i} is CUDA ordinal {@code i}. That correspondence is checked here rather than trusted:
     * an ordinal the backend does not have is a refusal, not a fallback to the default device.
     * </p>
     */
    static TornadoDevice forCudaOrdinal(int ordinal) {
        var runtime = TornadoRuntimeProvider.getTornadoRuntime();
        for (int i = 0; i < runtime.getNumBackends(); i++) {
            TornadoBackend backend = runtime.getBackend(i);
            if (backend.getBackendType() != TornadoVMBackendType.CUDA) {
                continue;
            }
            if (ordinal < 0 || ordinal >= backend.getNumDevices()) {
                throw new IllegalStateException("cuDF is on CUDA device " + ordinal + " but TornadoVM's CUDA backend has only " + backend.getNumDevices()
                        + " device(s); a kernel launched on another device cannot read this one's pointers");
            }
            return backend.getDevice(ordinal);
        }
        throw new IllegalStateException("TornadoVM has no CUDA backend on this classpath, so a cuDF device pointer cannot be given to a kernel");
    }
}
