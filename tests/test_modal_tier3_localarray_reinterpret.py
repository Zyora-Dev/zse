"""GPU smoke test for Tier-3 primitives: zse.local_array + zse.reinterpret.

Builds an INT4-style mini-dequant kernel that exercises all three new
features end-to-end on a real A100:

    weights: uint8 buffer of packed nibbles
    qp = zse.reinterpret(weights, zse.uint32)   # vectorized u32 loads
    buf = zse.local_array(8, zse.int32)         # per-thread register scratch
    zse.unpack_uint4(qp[tid], buf, 0)           # 8 unsigned nibbles
    out[tid] = sum(buf[0..7])

Compares GPU output against Python reference. Run:
    modal run tests/test_modal_tier3_localarray_reinterpret.py
"""

import modal
import sys

app = modal.App("zse-tier3-localarray-reinterpret-gpu")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=300)
def test_tier3_on_gpu():
    sys.path.insert(0, "/root/zse-compiler")
    import struct
    import zse_compiler as zse

    print("=" * 72)
    print("ZSE Tier-3 — local_array + reinterpret on REAL A100")
    print("=" * 72)

    devs = zse.get_devices(zse.detect_backend())
    print(f"[GPU] {devs[0].name}  sm_{devs[0].compute_capability}  "
          f"{devs[0].vram_total_gb:.1f}GB")

    # ---------- The kernel exercising all 3 primitives ----------
    @zse.kernel
    def int4_minisum(weights: "uint8_tensor", out: "int32_tensor", N: int):
        tid = zse.global_id(0)
        if tid >= N:
            return
        # Tier-3 (Fix 2): reinterpret packed bytes as uint32 — one 32-bit load
        # instead of four 8-bit loads.
        qp = zse.reinterpret(weights, zse.uint32)
        packed = qp[tid]
        # Tier-3 (Fix 1): per-thread register scratch.
        buf = zse.local_array(8, zse.int32)
        # Tier-2.5: unpack 8 unsigned 4-bit nibbles into the scratch.
        zse.unpack_uint4(packed, buf, 0)
        # Sum the 8 nibbles (range [0, 8*15] = [0, 120]).
        s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
        out[tid] = s

    print("\n[CODEGEN] CUDA source actually sent to NVRTC:")
    src = int4_minisum.source("cuda")
    print(src)

    # Sanity checks on the generated code
    assert "unsigned int* qp = " in src,    "reinterpret LHS missing"
    assert "((unsigned int*)(weights))" in src, "reinterpret cast missing"
    assert "int buf[8];" in src,           "local_array missing"
    assert "_zse_uu4_" in src,             "unpack_uint4 lowering missing"
    print("    ✓ All 3 primitives present in generated source")

    print("\n[COMPILE] NVRTC compiling...")
    int4_minisum.compile("cuda")
    print("    ✓ NVRTC compile OK")

    # ---------- Test data ----------
    test_words = [
        0x00000000,  # all zero          → sum 0
        0xFFFFFFFF,  # all 15            → sum 120
        0x76543210,  # 0..7              → sum 28
        0xFEDCBA98,  # 8..15             → sum 92
        0x12345678,  # 8,7,6,5,4,3,2,1   → sum 36
        0x0F0F0F0F,  # alt 15,0          → sum 60
        0xF0F0F0F0,  # alt 0,15          → sum 60
        0xAAAAAAAA,  # all 10            → sum 80
        0x55555555,  # all 5             → sum 40
        0xDEADBEEF,  # 15,14,14,11,13,10,14,13 → sum 104
        0xCAFEBABE,  # 14,11,10,11,14,15,10,12 → sum 97
        0x89ABCDEF,  # 15,14,13,12,11,10,9,8   → sum 92
    ]
    N = len(test_words)

    # Pack into a flat uint8 buffer (little-endian within each u32).
    packed_bytes = bytearray()
    for w in test_words:
        for i in range(4):
            packed_bytes.append((w >> (i * 8)) & 0xFF)

    out_zero = bytes(N * 4)
    mem = zse.GPUMemory(backend="cuda")

    weights = mem.allocate((N * 4,), zse.uint8)
    out_buf = mem.allocate((N,), zse.int32)
    mem.copy_host_to_device(bytes(packed_bytes), weights)
    mem.copy_host_to_device(out_zero, out_buf)

    block = 256
    grid = (N + block - 1) // block
    int4_minisum.launch(grid=(grid,), block=(block,),
                        args=(weights, out_buf, N), backend="cuda")

    got = list(struct.unpack(f"<{N}i", mem.copy_device_to_host(out_buf)))

    # Python reference
    def ref(w):
        return sum((w >> (i * 4)) & 0xF for i in range(8))

    print(f"\n[VERIFY] N={N} packed words, summing 8 unsigned nibbles each:")
    fail = 0
    for i, w in enumerate(test_words):
        expect = ref(w)
        actual = got[i]
        ok = expect == actual
        marker = "✓" if ok else "✗"
        print(f"  {marker} 0x{w:08X}  expect_sum={expect:3d}  gpu_sum={actual:3d}")
        if not ok:
            fail += 1

    mem.free(weights)
    mem.free(out_buf)

    print("\n" + "=" * 72)
    print(f"RESULT: {fail} mismatches out of {N}")
    print("=" * 72)
    assert fail == 0, f"{fail} mismatches — Tier-3 primitives misbehaving on GPU"
    print("✅ ALL CORRECT — local_array + reinterpret + unpack_uint4 work on A100")
    return {"failures": fail, "n": N}


@app.local_entrypoint()
def main():
    result = test_tier3_on_gpu.remote()
    print(f"\nModal result: {result}")
