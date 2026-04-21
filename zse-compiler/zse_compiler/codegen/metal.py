"""Metal Shading Language Code Generator — IR → MSL source code.

Metal differences from CUDA/HIP:
- SIMD groups (warps) use simd_shuffle_* intrinsics
- Shared memory = threadgroup address space
- No __global__ — uses kernel function attribute
- Thread indexing via function parameters with [[ ]] attributes
- SIMD width is 32 on Apple GPUs
- Atomics use atomic_fetch_*_explicit with memory_order
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


MATH_FUNCS = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "max_val": "max",
    "min_val": "min",
    "fma": "fma",
    "pow": "pow",
    "cos": "cos",
    "sin": "sin",
}

REDUCE_OPS = {
    "sum": lambda a, b: f"({a} + {b})",
    "max": lambda a, b: f"max({a}, {b})",
    "min": lambda a, b: f"min({a}, {b})",
}

SIMD_WIDTH = 32  # Apple GPU SIMD width


class MetalCodegen(BaseCodegen):
    """Generates Metal Shading Language kernel source code from ZSE IR."""

    def _emit_header(self) -> str:
        return (
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
        )

    def _emit_function_signature(self, func: IRFunction) -> str:
        params = []
        for p in func.params:
            params.append(self._param_to_str(p))

        # Metal thread indexing via function parameters
        params.append("uint3 _zse_tid [[thread_position_in_threadgroup]]")
        params.append("uint3 _zse_bid [[threadgroup_position_in_grid]]")
        params.append("uint3 _zse_bdim [[threads_per_threadgroup]]")
        params.append("uint3 _zse_gdim [[threadgroups_per_grid]]")
        params.append("uint _zse_simd_lane [[thread_index_in_simdgroup]]")
        params.append("uint _zse_simd_id [[simdgroup_index_in_threadgroup]]")

        params_str = ",\n    ".join(params)
        return f"kernel void {func.name}(\n    {params_str})"

    def _param_to_str(self, param: IRParam) -> str:
        if param.dtype in ("tensor", "Tensor"):
            return f"device float* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype in ("half_tensor", "fp16_tensor"):
            return f"device half* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype == "uint8_tensor":
            return f"device uchar* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype == "int8_tensor":
            return f"device char* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype == "int32_tensor":
            return f"device int* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype in DTYPE_MAP:
            metal_type = DTYPE_MAP[param.dtype].metal_type
            return f"device {metal_type}* {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype == "int":
            return f"constant int& {param.name} [[buffer({self._get_buffer_index(param)})]]"
        elif param.dtype == "float":
            return f"constant float& {param.name} [[buffer({self._get_buffer_index(param)})]]"
        return f"device float* {param.name} [[buffer({self._get_buffer_index(param)})]]"

    def _get_buffer_index(self, param: IRParam) -> int:
        if not hasattr(self, '_buffer_counter'):
            self._buffer_counter = 0
        idx = self._buffer_counter
        self._buffer_counter += 1
        return idx

    def generate(self, func) -> str:
        self._buffer_counter = 0
        return super().generate(func)

    def _emit_thread_idx(self, axis: int) -> str:
        components = {0: "_zse_tid.x", 1: "_zse_tid.y", 2: "_zse_tid.z"}
        return components[axis]

    def _emit_block_idx(self, axis: int) -> str:
        components = {0: "_zse_bid.x", 1: "_zse_bid.y", 2: "_zse_bid.z"}
        return components[axis]

    def _emit_block_dim(self, axis: int) -> str:
        components = {0: "_zse_bdim.x", 1: "_zse_bdim.y", 2: "_zse_bdim.z"}
        return components[axis]

    def _emit_grid_dim(self, axis: int) -> str:
        components = {0: "_zse_gdim.x", 1: "_zse_gdim.y", 2: "_zse_gdim.z"}
        return components[axis]

    def _emit_barrier(self) -> str:
        return "threadgroup_barrier(mem_flags::mem_threadgroup);"

    def _emit_shared_mem_decl(self, node: IRSharedMemDecl) -> str:
        self._shared_mem_counter += 1
        name = node.name or f"smem_{self._shared_mem_counter}"
        node.name = name
        dtype = DTYPE_MAP.get(node.dtype)
        metal_type = dtype.metal_type if dtype else "float"
        size = " * ".join(str(s) for s in node.shape)
        return f"threadgroup {metal_type} {name}[{size}];"

    # --- Atomics ---

    def _emit_atomic_add(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_add_explicit((device atomic_float*)&{ptr}, {val}, memory_order_relaxed);"

    def _emit_atomic_add_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_add_explicit((device atomic_float*)&{ptr}, {val}, memory_order_relaxed)"

    def _emit_atomic_max(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_max_explicit((device atomic_int*)&{ptr}, as_type<int>({val}), memory_order_relaxed);"

    def _emit_atomic_max_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_max_explicit((device atomic_int*)&{ptr}, as_type<int>({val}), memory_order_relaxed)"

    def _emit_atomic_min(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_min_explicit((device atomic_int*)&{ptr}, as_type<int>({val}), memory_order_relaxed);"

    def _emit_atomic_min_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        val = self._emit_expr(node.value)
        return f"atomic_fetch_min_explicit((device atomic_int*)&{ptr}, as_type<int>({val}), memory_order_relaxed)"

    def _emit_atomic_cas_expr(self, node) -> str:
        ptr = self._emit_expr(node.ptr)
        cmp = self._emit_expr(node.compare)
        val = self._emit_expr(node.value)
        return f"atomic_compare_exchange_weak_explicit((device atomic_int*)&{ptr}, &{cmp}, {val}, memory_order_relaxed, memory_order_relaxed)"

    def _emit_math_func(self, node: IRMathFunc) -> str:
        func_name = MATH_FUNCS.get(node.name, node.name)
        args = ", ".join(self._emit_expr(a) for a in node.args)
        return f"{func_name}({args})"

    # --- Warp (SIMD group) primitives ---

    def _emit_lane_id(self) -> str:
        return "_zse_simd_lane"

    def _emit_warp_id(self) -> str:
        return "_zse_simd_id"

    def _emit_warp_shuffle(self, node: IRWarpShuffle) -> str:
        val = self._emit_expr(node.value)
        offset = self._emit_expr(node.offset)
        variant_map = {
            "down": f"simd_shuffle_down({val}, {offset})",
            "up":   f"simd_shuffle_up({val}, {offset})",
            "xor":  f"simd_shuffle_xor({val}, {offset})",
            "idx":  f"simd_shuffle({val}, {offset})",
        }
        return variant_map[node.variant]

    def _emit_warp_vote(self, node: IRWarpVote) -> str:
        pred = self._emit_expr(node.predicate)
        vote_map = {
            "ballot": f"simd_ballot({pred})",
            "all":    f"simd_all({pred})",
            "any":    f"simd_any({pred})",
        }
        return vote_map[node.variant]

    def _emit_warp_reduce(self, node: IRWarpReduce) -> str:
        val = self._emit_expr(node.value)
        # Metal has built-in SIMD reductions
        reduce_map = {
            "sum": f"simd_sum({val})",
            "max": f"simd_max({val})",
            "min": f"simd_min({val})",
        }
        return reduce_map[node.op]

    def _emit_block_reduce(self, node: IRBlockReduce) -> str:
        val = self._emit_expr(node.value)
        combine = REDUCE_OPS[node.op]
        identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[node.op]
        var = f"_zse_br_{id(node) % 10000}"

        # Metal: use simd_sum/max/min for stage 1, then threadgroup memory
        simd_reduce = {"sum": "simd_sum", "max": "simd_max", "min": "simd_min"}[node.op]

        lines = []
        lines.append(f"[&]() {{")
        lines.append(f"  threadgroup float _zse_bsmem[32];")
        lines.append(f"  float {var} = {simd_reduce}({val});")
        lines.append(f"  if (_zse_simd_lane == 0) _zse_bsmem[_zse_simd_id] = {var};")
        lines.append(f"  threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"  uint _zse_nsimd = (_zse_bdim.x + {SIMD_WIDTH - 1}) / {SIMD_WIDTH};")
        lines.append(f"  {var} = (_zse_simd_lane < _zse_nsimd) ? _zse_bsmem[_zse_simd_lane] : {identity};")
        lines.append(f"  {var} = {simd_reduce}({var});")
        lines.append(f"  return {var};")
        lines.append(f"}}()")
        return "\n".join(lines)

    # --- Vectorized memory ---

    def _emit_load_float4(self, node: IRLoadFloat4) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        return f"reinterpret_cast<device float4*>({tensor})[{idx}]"

    def _emit_store_float4(self, node: IRStoreFloat4) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        vals = [self._emit_expr(v) for v in node.values]
        return f"reinterpret_cast<device float4*>({tensor})[{idx}] = float4({', '.join(vals)});"

    def _emit_store_float4_expr(self, node: IRStoreFloat4) -> str:
        return self._emit_store_float4(node).rstrip(";")

    def _emit_load_half2(self, node: IRLoadHalf2) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        return f"reinterpret_cast<device half2*>({tensor})[{idx}]"

    def _emit_store_half2(self, node: IRStoreHalf2) -> str:
        tensor = self._emit_expr(node.tensor)
        idx = self._emit_expr(node.index)
        vals = [self._emit_expr(v) for v in node.values]
        return f"reinterpret_cast<device half2*>({tensor})[{idx}] = half2({', '.join(vals)});"

    # --- Tiling ---

    def _emit_tile_load(self, node: IRTileLoad) -> str:
        tensor = self._emit_expr(node.tensor)
        tr = self._emit_expr(node.tile_row)
        tc = self._emit_expr(node.tile_col)
        ts = self._emit_expr(node.tile_size)
        sbuf = self._emit_expr(node.shared_buf) if node.shared_buf else "_zse_tile"
        lines = []
        lines.append(f"{sbuf}[_zse_tid.y * {ts} + _zse_tid.x] = "
                      f"{tensor}[({tr} + _zse_tid.y) * {ts} + ({tc} + _zse_tid.x)];")
        lines.append(f"threadgroup_barrier(mem_flags::mem_threadgroup);")
        return "\n".join(self._indented(l) for l in lines)

    def _emit_tile_store(self, node: IRTileStore) -> str:
        tensor = self._emit_expr(node.tensor)
        tr = self._emit_expr(node.tile_row)
        tc = self._emit_expr(node.tile_col)
        ts = self._emit_expr(node.tile_size)
        sbuf = self._emit_expr(node.shared_buf)
        lines = []
        lines.append(f"{tensor}[({tr} + _zse_tid.y) * {ts} + ({tc} + _zse_tid.x)] = "
                      f"{sbuf}[_zse_tid.y * {ts} + _zse_tid.x];")
        lines.append(f"threadgroup_barrier(mem_flags::mem_threadgroup);")
        return "\n".join(self._indented(l) for l in lines)

    def _default_var_type(self) -> str:
        """Metal doesn't support 'auto' — use float as default."""
        return "float"

    # --- FP16 conversion ---

    def _emit_half_to_float(self, node) -> str:
        val = self._emit_expr(node.value)
        return f"float({val})"

    def _emit_float_to_half(self, node) -> str:
        val = self._emit_expr(node.value)
        return f"half({val})"

    # --- Dynamic Shared Memory ---

    def _emit_dynamic_shared_mem_decl(self, node) -> str:
        self._shared_mem_counter += 1
        name = node.name or f"_zse_dsmem_{self._shared_mem_counter}"
        node.name = name
        return f"threadgroup float {name}[1]; // Metal: use threadgroup buffer for dynamic size"
