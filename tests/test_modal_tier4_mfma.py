"""Tier-4 MFMA GPU validation on AMD MI300X.

Single-wavefront (64 threads) MFMA test: C = A @ B for 16x16 fp16 tiles.
A = B = identity ⇒ expect C = identity.

Layout per CDNA3 mfma_f32_16x16x16f16:
  lane   = threadIdx.x (0..63)
  row16  = lane % 16
  k_base = (lane / 16) * 4
  A lane: A[row16, k_base..k_base+3]    (4 fp16, row-major)
  B lane: B[k_base..k_base+3, row16]    (4 fp16)
  C lane: C[k_base..k_base+3, row16]    (4 fp32 result)
"""

import struct
import sys

import zse_compiler as zse


@zse.kernel
def mfma_identity_kernel(a_in: "fp16_tensor", b_in: "fp16_tensor", c_out: "fp32_tensor"):
    lane = zse.thread_id(0)
    row16 = lane % 16
    k_base = (lane // 16) * 4

    a = zse.local_array(4, zse.float16)
    b = zse.local_array(4, zse.float16)
    c = zse.local_array(4, zse.float32)

    a[0] = a_in[row16 * 16 + (k_base + 0)]
    a[1] = a_in[row16 * 16 + (k_base + 1)]
    a[2] = a_in[row16 * 16 + (k_base + 2)]
    a[3] = a_in[row16 * 16 + (k_base + 3)]

    b[0] = b_in[(k_base + 0) * 16 + row16]
    b[1] = b_in[(k_base + 1) * 16 + row16]
    b[2] = b_in[(k_base + 2) * 16 + row16]
    b[3] = b_in[(k_base + 3) * 16 + row16]

    c[0] = 0.0
    c[1] = 0.0
    c[2] = 0.0
    c[3] = 0.0

    zse.mfma_f32_16x16x16_f16(a, b, c)

    c_out[(k_base + 0) * 16 + row16] = c[0]
    c_out[(k_base + 1) * 16 + row16] = c[1]
    c_out[(k_base + 2) * 16 + row16] = c[2]
    c_out[(k_base + 3) * 16 + row16] = c[3]


def fp16_bits(f):
    if f == 0.0:
        return 0x0000
    if f == 1.0:
        return 0x3C00
    raise ValueError("only 0/1 needed")


def main():
    print("=== Tier-4 MFMA GPU validation on MI300X ===", flush=True)

    print("\n--- Generated HIP ---", flush=True)
    print(mfma_identity_kernel.source(backend="rocm"), flush=True)
    print("---------------------\n", flush=True)

    print("Compiling via HIPRTC...", flush=True)
    mfma_identity_kernel.compile(backend="rocm")
    print("Compile OK", flush=True)

    N = 16
    mem = zse.GPUMemory(backend="rocm")
    a_dev = mem.allocate((N * N,), zse.float16)
    b_dev = mem.allocate((N * N,), zse.float16)
    c_dev = mem.allocate((N * N,), zse.float32)

    a_host = bytearray(N * N * 2)
    b_host = bytearray(N * N * 2)
    for i in range(N):
        for j in range(N):
            v = fp16_bits(1.0 if i == j else 0.0)
            struct.pack_into("<H", a_host, (i * N + j) * 2, v)
            struct.pack_into("<H", b_host, (i * N + j) * 2, v)

    mem.copy_host_to_device(bytes(a_host), a_dev)
    mem.copy_host_to_device(bytes(b_host), b_dev)
    mem.copy_host_to_device(bytes(N * N * 4), c_dev)

    print("Launching: grid=(1,1,1), block=(64,1,1)...", flush=True)
    mfma_identity_kernel.launch(
        grid=(1, 1, 1),
        block=(64, 1, 1),
        args=(a_dev, b_dev, c_dev),
        backend="rocm",
    )

    out_bytes = mem.copy_device_to_host(c_dev)
    c_host = list(struct.unpack(f"<{N * N}f", out_bytes))

    print("\n--- Result matrix ---", flush=True)
    for i in range(N):
        print(" ".join(f"{c_host[i * N + j]:.1f}" for j in range(N)), flush=True)

    n_wrong = 0
    for i in range(N):
        for j in range(N):
            expected = 1.0 if i == j else 0.0
            actual = c_host[i * N + j]
            if abs(actual - expected) >= 1e-3:
                n_wrong += 1
                if n_wrong <= 5:
                    print(f"  MISMATCH C[{i},{j}] expect={expected} got={actual}", flush=True)

    n_correct = N * N - n_wrong
    print(f"\nResult: {n_correct}/{N * N} correct, {n_wrong} wrong", flush=True)

    mem.free(a_dev)
    mem.free(b_dev)
    mem.free(c_dev)

    if n_wrong == 0:
        print("=== MFMA TIER-4 GPU VALIDATION: PASSED ===", flush=True)
        sys.exit(0)
    print("=== MFMA TIER-4 GPU VALIDATION: FAILED ===", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
