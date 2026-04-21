"""ROCm/HIP Code Generator — IR → HIP C source code.

Key differences from CUDA:
- Wavefront size is 64 (not 32) on most AMD GPUs
- Shuffle intrinsics: __shfl_down → __shfl_down (HIP provides CUDA-compat wrappers)
- No _sync variants needed (HIP wavefront is always implicitly synced)
"""

from zse_compiler.codegen.base import BaseCodegen
from zse_compiler.ir.nodes import (
    IRFunction, IRParam, IRSharedMemDecl,
    IRAtomicAdd, IRAtomicMax, IRAtomicMin, IRAtomicCAS, IRMathFunc,
    IRWarpShuffle, IRWarpVote, IRWarpReduce, IRBlockReduce,
    IRLoadFloat4, IRStoreFloat4, IRLoadHalf2, IRStoreHalf2,
    IRTileLoad, IRTileStore, IRConst,
)
from zse_compiler.types.dtypes import DTYPE_MAP


AXIS_MAP = {0: "x", 1: "y", 2: "z"}

MATH_FUNCS = {
    "exp": "expf",
    "log": "logf",
    "sqrt": "sqrtf",
    "rsqrt": "rsqrtf",
    "max_val": "fmaxf",
    "min_val": "fminf",
    "fma": "fmaf",
    "pow": "powf",
    "cos": "cosf",
    "sin": "sinf",
}

REDUCE_OPS = {
    "sum": lambda a, b: f"({a} + {b})",
    "max": lambda a, b: f"fmaxf({a}, {b})",
    "min": lambda a, b: f"fminf({a}, {b})",
}

# AMD wavefront is 64 on RDNA/CDNA — need extra reduction steps
WARP_SIZE = 64
SHUFFLE_OFFSETS = [32, 16, 8, 4, 2, 1]


class ROCmCodegen(BaseCodegen):
    """Generates HIP C kernel source code from ZSE IR."""

    def _emit_header(self) -> str:
        return '#include <hip/hip_runtime.h>\n\nextern "C" {'

    def _emit_footer(self) -> str:
        return "}"

    def _emit_function_signature(self, func: IRFunction) -> str:
        params = ", ".join(self._param_to_str(p) for p in func.params)
        return f"__global__ void {func.name}({params})"

    def _param_to_str(self, param: IRParam) -> str:
        if param.dtype in ("tensor", "Tensor"):
            return f"float* __restrict__ {param.name}"
        elif param.dtype in ("half_tensor", "fp16_tensor"):
            return f"half* __restrict__ {param.name}"
        elif param.dtype == "uint8_tensor":
            return f"unsigned char* __restrict__ {param.name}"
        elif param.dtype == "int8_tensor":
            return f"signed char* __restrict__ {param.name}"
        elif param.dtype == "int32_tensor":
            return f"int* __restrict__ {param.name}"
        elif param.dtype in DTYPE_MAP:
            return f"{DTYPE_MAP[param.dtype].hip_type}* __restrict__ {param.name}"
        elif param.dtype == "int":
            return f"int {param.name}"
        elif param.dtype == "float":
            return f"float {param.name}"
        return f"float* __restrict__ {param.name}"

    def _emit_thread_idx(self, axis: int) -> str:
        return f"threadIdx.{AXIS_MAP[axis]}"

    def _emit_block_idx(self, axis: int) -> str:
        return f"blockIdx.{AXIS_MAP[axis]}"

    def _emit_block_dim(self, axis: int) -> str:
        return f"blockDim.{AXIS_MAP[axis]}"

    def _emit_grid_dim(self, axis: int) -> str:
        return f"gridDim.{AXIS_MAP[axis]}"

    def _emit_barrier(self) -> str:
        return "__syncthreads();"

    def _emit_shared_mem_decl(self, node: IRSharedMemDecl) -> str:
        self._shared_mem_counter += 1
        name = node.name or f"smem_{self._shared_mem_counter}"
        node.name = name
        dtype = DTYPE_MAP.get(node.dtype)
        hip_type = dtype.hip_type if dtype else "float"
        size = " * ".join(str(s) for s in node.shape)
        return f"__shared__ {hip_type} {name}[{size}];"

    # --- Atomics ---

    def _emit_atomic_add(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicAdd(&{ptr}, {val});"

    def _emit_atomic_add_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicAdd(&{ptr}, {val})"

    def _emit_atomic_max(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicMax(&{ptr}, {val});"

    def _emit_atomic_max_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicMax(&{ptr}, {val})"

    def _emit_atomic_min(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicMin(&{ptr}, {val});"

    def _emit_atomic_min_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomicMin(&{ptr}, {val})"

    def _emit_atomic_cas_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        cmp = self._emit_expr(node.compare)
        val = self._emit_expr(node.value)
        return f"atomicCAS(&{ptr}, {cmp}, {val})"

    def _emit_math_func(self, node: IRMathFunc) -> str:
        func_name = MATH_FUNCS.get(node.name, node.name)
        args = ", ".join(self._emit_expr(a) for a in node.args)
        return f"{func_name}({args})"

    # --- Warp (Wavefront) primitives ---

    def _emit_lane_id(self) -> str:
        return f"(threadIdx.x % {WARP_SIZE})"

    def _emit_warp_id(self) -> str:
        return f"(threadIdx.x / {WARP_SIZE})"

    def _emit_warp_shuffle(self, node: IRWarpShuffle) -> str:
        val = self._emit_expr(node.value)
        offset = self._emit_expr(node.offset)
        width = self._emit_expr(node.width)
        # HIP provides CUDA-compatible wrappers
        variant_map = {
            "down": f"__shfl_down({val}, {offset}, {width})",
            "up":   f"__shfl_up({val}, {offset}, {width})",
            "xor":  f"__shfl_xor({val}, {offset}, {width})",
            "idx":  f"__shfl({val}, {offset}, {width})",
        }
        return variant_map[node.variant]

    def _emit_warp_vote(self, node: IRWarpVote) -> str:
        pred = self._emit_expr(node.predicate)
        vote_map = {
            "ballot": f"__ballot({pred})",
            "all":    f"__all({pred})",
            "any":    f"__any({pred})",
        }
        return vote_map[node.variant]

    def _emit_warp_reduce(self, node: IRWarpReduce) -> str:
        val = self._emit_expr(node.value)
        combine = REDUCE_OPS[node.op]
        var = f"_zse_wr_{id(node) % 10000}"
        lines = [f"[&]() {{ float {var} = {val};"]
        for offset in SHUFFLE_OFFSETS:  # 32,16,8,4,2,1 for AMD wavefront64
            shfl = f"__shfl_xor({var}, {offset}, {WARP_SIZE})"
            lines.append(f" {var} = {combine(var, shfl)};")
        lines.append(f" return {var}; }}()")
        return "".join(lines)

    def _emit_block_reduce(self, node: IRBlockReduce) -> str:
        val = self._emit_expr(node.value)
        combine = REDUCE_OPS[node.op]
        identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[node.op]
        var = f"_zse_br_{id(node) % 10000}"
        nwarps = 16  # max 1024 threads / 64 wavefront = 16 wavefronts

        lines = []
        lines.append(f"[&]() {{")
        lines.append(f"  __shared__ float _zse_bsmem[{nwarps}];")
        lines.append(f"  float {var} = {val};")
        for offset in SHUFFLE_OFFSETS:
            shfl = f"__shfl_xor({var}, {offset}, {WARP_SIZE})"
            lines.append(f"  {var} = {combine(var, shfl)};")
        lines.append(f"  int _zse_lid = threadIdx.x % {WARP_SIZE};")
        lines.append(f"  int _zse_wid = threadIdx.x / {WARP_SIZE};")
        lines.append(f"  if (_zse_lid == 0) _zse_bsmem[_zse_wid] = {var};")
        lines.append(f"  __syncthreads();")
        lines.append(f"  int _zse_nwarps = (blockDim.x + {WARP_SIZE - 1}) / {WARP_SIZE};")
        lines.append(f"  {var} = (_zse_lid < _zse_nwarps) ? _zse_bsmem[_zse_lid] : {identity};")
        for offset in SHUFFLE_OFFSETS:
            shfl = f"__shfl_xor({var}, {offset}, {WARP_SIZE})"
            lines.append(f"  {var} = {combine(var, shfl)};")
        lines.append(f"  return {var};")
        lines.append(f"}}()")
        return "\n".join(lines)

    # --- Vectorized memory ---

    def _emit_load_float4(self, node: IRLoadFloat4) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        return f"reinterpret_cast<float4*>({tensor})[{idx}]"

    def _emit_store_float4(self, node: IRStoreFloat4) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        vals = [self._emit_expr(v) for v in node.values]
        return f"reinterpret_cast<float4*>({tensor})[{idx}] = make_float4({', '.join(vals)});"

    def _emit_store_float4_expr(self, node: IRStoreFloat4) -> str:
        return self._emit_store_float4(node).rstrip(";")

    def _emit_load_half2(self, node: IRLoadHalf2) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        return f"reinterpret_cast<half2*>({tensor})[{idx}]"

    def _emit_store_half2(self, node: IRStoreHalf2) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        vals = [self._emit_expr(v) for v in node.values]
        return f"reinterpret_cast<half2*>({tensor})[{idx}] = make_half2({', '.join(vals)});"

    # --- Tiling ---

    def _emit_tile_load(self, node: IRTileLoad) -> str:
        tensor = self._emit_expr(node.tensor)
        tr = self._emit_expr(node.tile_row)
        tc = self._emit_expr(node.tile_col)
        ts = self._emit_expr(node.tile_size)
        sbuf = self._emit_expr(node.shared_buf) if node.shared_buf else "_zse_tile"
        lines = []
        lines.append(f"{sbuf}[threadIdx.y * {ts} + threadIdx.x] = "
                      f"{tensor}[({tr} + threadIdx.y) * {ts} + ({tc} + threadIdx.x)];")
        lines.append(f"__syncthreads();")
        return "\n".join(self._indented(l) for l in lines)

    def _emit_tile_store(self, node: IRTileStore) -> str:
        tensor = self._emit_expr(node.tensor)
        tr = self._emit_expr(node.tile_row)
        tc = self._emit_expr(node.tile_col)
        ts = self._emit_expr(node.tile_size)
        sbuf = self._emit_expr(node.shared_buf)
        lines = []
        lines.append(f"{tensor}[({tr} + threadIdx.y) * {ts} + ({tc} + threadIdx.x)] = "
                      f"{sbuf}[threadIdx.y * {ts} + threadIdx.x];")
        lines.append(f"__syncthreads();")
        return "\n".join(self._indented(l) for l in lines)

    # --- FP16 conversion ---

    def _emit_half_to_float(self, node) -> str:
        val = self._emit_expr(node.value)
        return f"__half2float({val})"

    def _emit_float_to_half(self, node) -> str:
        val = self._emit_expr(node.value)
        return f"__float2half({val})"

    # --- Dynamic Shared Memory ---

    def _emit_dynamic_shared_mem_decl(self, node) -> str:
        self._shared_mem_counter += 1
        name = node.name or f"_zse_dsmem_{self._shared_mem_counter}"
        node.name = name
        from zse_compiler.types.dtypes import DTYPE_MAP as DM
        dtype = DM.get(node.dtype)
        hip_type = dtype.hip_type if dtype else "float"
        return f"extern __shared__ {hip_type} {name}[];"
