"""GPU smoke test for zse.unpack_int4 / zse.unpack_uint4 intrinsics.

Verifies the Tier-2 / Tier-2.5 primitives actually execute correctly on a
real A100 (not just generate plausible-looking source strings).

Pipeline: @zse.kernel using the intrinsics → CUDA C → NVRTC → cubin → GPU
launch → readback → compare with Python reference.

Run: modal run tests/test_modal_unpack_int4.py
"""

import modal
import sys

app = modal.App("zse-unpack-int4-gpu-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
)


# ---------- Python references ----------

def ref_unpack_int4(packed: int) -> list:
    """Sign-extended nibbles, range [-8, 7]."""
    out = []
    for i in range(8):
        n = (packed >> (i * 4)) & 0xF
        if n & 0x8:
            n -= 0x10
        out.append(n)
    return out


def ref_unpack_uint4(packed: int) -> list:
    """Unsigned nibbles, range [0, 15]."""
    return [(packed >> (i * 4)) & 0xF for i in range(8)]


@app.function(gpu="A100", image=zse_image, timeout=300)
def test_unpack_on_gpu():
    sys.path.insert(0, "/root/zse-compiler")
    import struct
    import zse_compiler as zse

    print("=" * 70)
    print("ZSE unpack_int4 / unpack_uint4 — GPU EXECUTION TEST")
    print("=" * 70)

    backend = zse.detect_backend()
    devices = zse.get_devices(backend)
    print(f"[GPU] {devices[0].name}  sm_{devices[0].compute_capability}  "
          f"{devices[0].vram_total_gb:.1f}GB")
    assert backend == "cuda"

    # ---------- Kernels ----------
    # int32_tensor maps to int* — we use the bits as uint32 inside the intrinsic.

    @zse.kernel
    def k_unpack_uint4(packed: "int32_tensor", out: "int32_tensor", N: int):
        tid = zse.global_id(0)
        if tid >= N:
            return
        zse.unpack_uint4(packed[tid], out, tid * 8)

    @zse.kernel
    def k_unpack_int4(packed: "int32_tensor", out: "int32_tensor", N: int):
        tid = zse.global_id(0)
        if tid >= N:
            return
        zse.unpack_int4(packed[tid], out, tid * 8)

    # ---------- Compile + show the actually-generated CUDA ----------
    print("\n[CODEGEN] Generated CUDA for unpack_uint4 kernel:")
    src_uu4 = k_unpack_uint4.source("cuda")
    print(src_uu4)

    print("\n[CODEGEN] Generated CUDA for unpack_int4 kernel:")
    src_u4 = k_unpack_int4.source("cuda")
    print(src_u4)

    print("\n[COMPILE] NVRTC compiling both kernels...")
    k_unpack_uint4.compile("cuda")
    k_unpack_int4.compile("cuda")
    print("    ✓ Both compiled")

    # ---------- Test data ----------
    test_words = [
        0x00000000,  # all zero
        0xFFFFFFFF,  # all 0xF (uint4: 15, int4: -1)
        0x76543210,  # 0..7 (no sign-bit set, both should agree)
        0xFEDCBA98,  # 8..15 (uint4) / -8..-1 (int4)
        0x12345678,
        0xDEADBEEF,
        0xCAFEBABE,
        0x89ABCDEF,
        0x0F0F0F0F,
        0xF0F0F0F0,
        0xAAAAAAAA,
        0x55555555,
    ]
    N = len(test_words)

    # Pack as int32 little-endian (struct 'i' = signed 32-bit, but the bit
    # pattern is preserved — kernel reinterprets via (unsigned int) cast).
    def to_i32(u32):
        return u32 - 0x100000000 if u32 >= 0x80000000 else u32

    packed_host = struct.pack(f"<{N}i", *[to_i32(w) for w in test_words])
    out_zero = bytes(N * 8 * 4)  # N * 8 nibbles * 4 bytes/int

    mem = zse.GPUMemory(backend="cuda")

    # ---------- Run uint4 ----------
    print(f"\n[RUN uint4] N={N} packed words → {N*8} unpacked nibbles")
    p_uu4 = mem.allocate((N,), zse.int32)
    o_uu4 = mem.allocate((N * 8,), zse.int32)
    mem.copy_host_to_device(packed_host, p_uu4)
    mem.copy_host_to_device(out_zero, o_uu4)

    block = 256
    grid = (N + block - 1) // block
    k_unpack_uint4.launch(grid=(grid,), block=(block,),
                          args=(p_uu4, o_uu4, N), backend="cuda")

    got_uu4 = list(struct.unpack(f"<{N*8}i", mem.copy_device_to_host(o_uu4)))

    fail_uu4 = 0
    for i, w in enumerate(test_words):
        expect = ref_unpack_uint4(w)
        actual = got_uu4[i * 8:(i + 1) * 8]
        ok = expect == actual
        marker = "✓" if ok else "✗"
        print(f"  {marker} 0x{w:08X}  exp={expect}  got={actual}")
        if not ok:
            fail_uu4 += 1

    # ---------- Run int4 ----------
    print(f"\n[RUN int4]  N={N} packed words → {N*8} unpacked nibbles")
    p_u4 = mem.allocate((N,), zse.int32)
    o_u4 = mem.allocate((N * 8,), zse.int32)
    mem.copy_host_to_device(packed_host, p_u4)
    mem.copy_host_to_device(out_zero, o_u4)

    k_unpack_int4.launch(grid=(grid,), block=(block,),
                         args=(p_u4, o_u4, N), backend="cuda")

    got_u4 = list(struct.unpack(f"<{N*8}i", mem.copy_device_to_host(o_u4)))

    fail_u4 = 0
    for i, w in enumerate(test_words):
        expect = ref_unpack_int4(w)
        actual = got_u4[i * 8:(i + 1) * 8]
        ok = expect == actual
        marker = "✓" if ok else "✗"
        print(f"  {marker} 0x{w:08X}  exp={expect}  got={actual}")
        if not ok:
            fail_u4 += 1

    # ---------- Cleanup ----------
    mem.free(p_uu4); mem.free(o_uu4)
    mem.free(p_u4);  mem.free(o_u4)

    print("\n" + "=" * 70)
    print(f"RESULT: unpack_uint4 failures = {fail_uu4} / {N}")
    print(f"RESULT: unpack_int4  failures = {fail_u4} / {N}")
    print("=" * 70)

    assert fail_uu4 == 0, f"{fail_uu4} unpack_uint4 mismatches"
    assert fail_u4 == 0, f"{fail_u4} unpack_int4 mismatches"
    print("✅ ALL CORRECT — intrinsics execute on real GPU and match Python reference")
    return {"uint4_fail": fail_uu4, "int4_fail": fail_u4, "n_tested": N}


@app.local_entrypoint()
def main():
    result = test_unpack_on_gpu.remote()
    print(f"\nModal result: {result}")
