"""End-to-end Metal test: detect → codegen → compile → dispatch → readback.

Runs on Apple Silicon / Intel Mac with discrete GPU.
"""
import ctypes
import struct
import zse_compiler as zse
from zse_compiler.runtime.device import detect_backend, get_devices
from zse_compiler.runtime.metal_dispatch import MetalRuntime


# --- 1. Detection ---
backend = detect_backend()
devices = get_devices("metal")
assert backend == "metal", f"Expected metal, got {backend}"
assert devices, "No Metal devices found"
d = devices[0]
print(f"[1/5] Detected: backend={backend}, name={d.name!r}, vram={d.vram_total_gb:.2f} GB")


# --- 2. Kernel codegen ---
@zse.kernel
def vector_add(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor, n: int):
    tid = zse.global_id(0)
    if tid < n:
        out[tid] = a[tid] + b[tid]


src = vector_add.source("metal")
assert "kernel void vector_add" in src
print(f"[2/5] Codegen: {len(src)} chars of MSL generated")


# --- 3. Compile ---
rt = MetalRuntime()
pipeline = rt.compile_msl(src, "vector_add")
assert pipeline, "compile_msl returned NULL"
print(f"[3/5] Compiled MSL → Metal pipeline (device: {rt.device_name!r}, vram={rt.device_memory_mb} MB)")


# --- 4. Alloc + populate buffers ---
N = 1024
nbytes = N * 4  # float32

buf_a = rt.alloc_buffer(nbytes)
buf_b = rt.alloc_buffer(nbytes)
buf_out = rt.alloc_buffer(nbytes)
buf_n = rt.alloc_buffer(4)  # constant int& n

# Write inputs via unified-memory pointer
def write_floats(buf, values):
    ptr = rt.buffer_contents(buf)
    arr = (ctypes.c_float * len(values)).from_address(ptr)
    for i, v in enumerate(values):
        arr[i] = v

def write_int(buf, value):
    ptr = rt.buffer_contents(buf)
    ctypes.c_int32.from_address(ptr).value = value

def read_floats(buf, n):
    ptr = rt.buffer_contents(buf)
    arr = (ctypes.c_float * n).from_address(ptr)
    return [arr[i] for i in range(n)]

write_floats(buf_a, [float(i) for i in range(N)])
write_floats(buf_b, [float(i) * 2.0 for i in range(N)])
write_floats(buf_out, [0.0] * N)
write_int(buf_n, N)

print(f"[4/5] Allocated 4 buffers (3×{nbytes}B + 4B scalar) in unified memory")


# --- 5. Dispatch + readback ---
BLOCK = 256
GRID = (N + BLOCK - 1) // BLOCK
gpu_ms = rt.dispatch(
    pipeline,
    [buf_a, buf_b, buf_out, buf_n],
    grid=(GRID, 1, 1),
    block=(BLOCK, 1, 1),
)

result = read_floats(buf_out, N)
expected = [float(i) + float(i) * 2.0 for i in range(N)]

mismatches = sum(1 for r, e in zip(result, expected) if abs(r - e) > 1e-5)
print(f"[5/5] Dispatched grid={GRID}×{BLOCK}, GPU time={gpu_ms:.3f} ms, "
      f"mismatches={mismatches}/{N}")

assert mismatches == 0, f"{mismatches} elements wrong; first few: {result[:5]} vs {expected[:5]}"

print()
print("✅ METAL END-TO-END PASS")
print(f"   {N} float32 elements computed on {rt.device_name}")
print(f"   result[0..4] = {result[:5]}")
print(f"   expected      = {expected[:5]}")

# Cleanup
rt.free_buffer(buf_a)
rt.free_buffer(buf_b)
rt.free_buffer(buf_out)
rt.free_buffer(buf_n)
