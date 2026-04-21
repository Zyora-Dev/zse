"""CUDA Code Generator — IR → CUDA C source code."""

from zse_compiler.codegen.base import BaseCodegen
from zse_compiler.ir.nodes import (
    IRFunction, IRParam, IRSharedMemDecl,
    IRAtomicAdd, IRAtomicMax, IRAtomicMin, IRAtomicCAS, IRMathFunc,
    IRWarpShuffle, IRWarpVote, IRWarpReduce, IRBlockReduce,
    IRLoadFloat4, IRStoreFloat4, IRLoadHalf2, IRStoreHalf2,
    IRTileLoad, IRTileStore, IRConst,
    IRWmmaLoadA, IRWmmaLoadB, IRWmmaFill, IRWmmaMMA, IRWmmaStore,
    IRHalfToFloat, IRFloatToHalf, IRDynamicSharedMemDecl,
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
    "fma": "__fmaf_rn",
    "pow": "powf",
    "cos": "cosf",
    "sin": "sinf",
}

# Warp reduce: op → combine expression
REDUCE_OPS = {
    "sum": lambda a, b: f"({a} + {b})",
    "max": lambda a, b: f"fmaxf({a}, {b})",
    "min": lambda a, b: f"fminf({a}, {b})",
}


class CUDACodegen(BaseCodegen):
    """Generates CUDA C kernel source code from ZSE IR."""

    def __init__(self):
        super().__init__()
        self._uses_wmma = False
        self._uses_half = False

    def generate(self, func) -> str:
        """Override to detect WMMA and half usage before generating."""
        from zse_compiler.ir.nodes import IRWmmaLoadA, IRWmmaLoadB, IRWmmaFill, IRWmmaMMA, IRWmmaStore
        self._uses_wmma = self._check_uses_wmma(func.body)
        self._uses_half = self._check_uses_half(func)
        return super().generate(func)

    def _check_uses_wmma(self, stmts) -> bool:
        from zse_compiler.ir.nodes import IRWmmaLoadA, IRWmmaLoadB, IRWmmaFill, IRWmmaMMA, IRWmmaStore
        for s in stmts:
            if isinstance(s, (IRWmmaLoadA, IRWmmaLoadB, IRWmmaFill, IRWmmaMMA, IRWmmaStore)):
                return True
            if hasattr(s, 'body') and isinstance(getattr(s, 'body'), list):
                if self._check_uses_wmma(s.body):
                    return True
            if hasattr(s, 'then_body'):
                if self._check_uses_wmma(s.then_body):
                    return True
        return False

    def _check_uses_half(self, func) -> bool:
        """Check if any param uses half_tensor or body uses half_to_float/float_to_half."""
        for p in func.params:
            if p.dtype in ("half_tensor", "fp16_tensor", "uint8_tensor", "int8_tensor"):
                return True
        return self._check_half_in_stmts(func.body)

    def _check_half_in_stmts(self, stmts) -> bool:
        for s in stmts:
            if isinstance(s, (IRHalfToFloat, IRFloatToHalf)):
                return True
            if hasattr(s, 'value') and isinstance(getattr(s, 'value'), (IRHalfToFloat, IRFloatToHalf)):
                return True
            if hasattr(s, 'body') and isinstance(getattr(s, 'body'), list):
                if self._check_half_in_stmts(s.body):
                    return True
            if hasattr(s, 'then_body'):
                if self._check_half_in_stmts(s.then_body):
                    return True
        return False

    def _emit_header(self) -> str:
        parts = []
        if self._uses_half:
            parts.append('#include <cuda_fp16.h>')
        if self._uses_wmma:
            parts.append('#include <mma.h>')
            parts.append('using namespace nvcuda;')
        parts.append('extern "C" {')
        return '\n'.join(parts)

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
            return f"{DTYPE_MAP[param.dtype].cuda_type}* __restrict__ {param.name}"
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
        cuda_type = dtype.cuda_type if dtype else "float"
        size = " * ".join(str(s) for s in node.shape)
        return f"__shared__ {cuda_type} {name}[{size}];"

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

    # --- Warp primitives ---

    def _emit_lane_id(self) -> str:
        return "(threadIdx.x & 31)"

    def _emit_warp_id(self) -> str:
        return "(threadIdx.x >> 5)"

    def _emit_warp_shuffle(self, node: IRWarpShuffle) -> str:
        val = self._emit_expr(node.value)
        offset = self._emit_expr(node.offset)
        width = self._emit_expr(node.width)
        variant_map = {
            "down": f"__shfl_down_sync(0xffffffff, {val}, {offset}, {width})",
            "up":   f"__shfl_up_sync(0xffffffff, {val}, {offset}, {width})",
            "xor":  f"__shfl_xor_sync(0xffffffff, {val}, {offset}, {width})",
            "idx":  f"__shfl_sync(0xffffffff, {val}, {offset}, {width})",
        }
        return variant_map[node.variant]

    def _emit_warp_vote(self, node: IRWarpVote) -> str:
        pred = self._emit_expr(node.predicate)
        vote_map = {
            "ballot": f"__ballot_sync(0xffffffff, {pred})",
            "all":    f"__all_sync(0xffffffff, {pred})",
            "any":    f"__any_sync(0xffffffff, {pred})",
        }
        return vote_map[node.variant]

    def _emit_warp_reduce(self, node: IRWarpReduce) -> str:
        """Emit butterfly warp reduction — fully unrolled, no shared memory needed.

        Generates:
            float val = <input>;
            val = val + __shfl_xor_sync(0xffffffff, val, 16, 32);  // or fmaxf etc
            val = val + __shfl_xor_sync(0xffffffff, val, 8, 32);
            val = val + __shfl_xor_sync(0xffffffff, val, 4, 32);
            val = val + __shfl_xor_sync(0xffffffff, val, 2, 32);
            val = val + __shfl_xor_sync(0xffffffff, val, 1, 32);
        """
        val = self._emit_expr(node.value)
        combine = REDUCE_OPS[node.op]

        # Generate inline butterfly reduction using a helper variable
        var_name = f"_zse_wr_{id(node) % 10000}"
        lines = [f"[&]() {{ float {var_name} = {val};"]
        for offset in [16, 8, 4, 2, 1]:
            shfl = f"__shfl_xor_sync(0xffffffff, {var_name}, {offset}, 32)"
            lines.append(f" {var_name} = {combine(var_name, shfl)};")
        lines.append(f" return {var_name}; }}()")
        return "".join(lines)

    def _emit_block_reduce(self, node: IRBlockReduce) -> str:
        """Emit block-level reduction: warp reduce → shared mem → cross-warp reduce.

        Two-stage:
            1. Each warp reduces its own values via butterfly shuffle
            2. Lane 0 of each warp writes to shared memory
            3. First warp reduces the partial sums from all warps
        """
        val = self._emit_expr(node.value)
        combine = REDUCE_OPS[node.op]
        identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[node.op]
        var = f"_zse_br_{id(node) % 10000}"

        lines = []
        lines.append(f"[&]() {{")
        lines.append(f"  __shared__ float _zse_bsmem[32];")  # max 32 warps per block
        lines.append(f"  float {var} = {val};")

        # Stage 1: warp-level butterfly reduce
        for offset in [16, 8, 4, 2, 1]:
            shfl = f"__shfl_xor_sync(0xffffffff, {var}, {offset}, 32)"
            lines.append(f"  {var} = {combine(var, shfl)};")

        # Stage 2: lane 0 of each warp writes to shared memory
        lines.append(f"  int _zse_lid = threadIdx.x & 31;")
        lines.append(f"  int _zse_wid = threadIdx.x >> 5;")
        lines.append(f"  if (_zse_lid == 0) _zse_bsmem[_zse_wid] = {var};")
        lines.append(f"  __syncthreads();")

        # Stage 3: first warp reduces across warps
        lines.append(f"  int _zse_nwarps = (blockDim.x + 31) >> 5;")
        lines.append(f"  {var} = (_zse_lid < _zse_nwarps) ? _zse_bsmem[_zse_lid] : {identity};")
        for offset in [16, 8, 4, 2, 1]:
            shfl = f"__shfl_xor_sync(0xffffffff, {var}, {offset}, 32)"
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
        """Emit cooperative tile load: each thread loads one element."""
        tensor = self._emit_expr(node.tensor)
        tr = self._emit_expr(node.tile_row)
        tc = self._emit_expr(node.tile_col)
        ts = self._emit_expr(node.tile_size)
        sbuf = self._emit_expr(node.shared_buf) if node.shared_buf else "_zse_tile"

        lines = []
        lines.append(f"// Cooperative tile load {ts}x{ts}")
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

    # --- WMMA / Tensor Core ---

    _wmma_counter = 0

    def _emit_wmma_load_a(self, node) -> str:
        CUDACodegen._wmma_counter += 1
        name = node.frag_name or f"_frag_a_{CUDACodegen._wmma_counter}"
        node.frag_name = name
        tensor = self._emit_expr(node.tensor)
        row = self._emit_expr(node.row)
        col = self._emit_expr(node.col)
        stride = self._emit_expr(node.stride)
        return (f"wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> {name};\n"
                + self._indented(f"wmma::load_matrix_sync({name}, (half*)({tensor} + {row} * {stride} + {col}), {stride});"))

    def _emit_wmma_load_b(self, node) -> str:
        CUDACodegen._wmma_counter += 1
        name = node.frag_name or f"_frag_b_{CUDACodegen._wmma_counter}"
        node.frag_name = name
        tensor = self._emit_expr(node.tensor)
        row = self._emit_expr(node.row)
        col = self._emit_expr(node.col)
        stride = self._emit_expr(node.stride)
        return (f"wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> {name};\n"
                + self._indented(f"wmma::load_matrix_sync({name}, (half*)({tensor} + {row} * {stride} + {col}), {stride});"))

    def _emit_wmma_fill(self, node) -> str:
        CUDACodegen._wmma_counter += 1
        name = node.frag_name or f"_frag_c_{CUDACodegen._wmma_counter}"
        node.frag_name = name
        val = self._emit_expr(node.value)
        return (f"wmma::fragment<wmma::accumulator, 16, 16, 16, float> {name};\n"
                + self._indented(f"wmma::fill_fragment({name}, {val});"))

    def _emit_wmma_mma(self, node) -> str:
        a = self._emit_expr(node.a_frag)
        b = self._emit_expr(node.b_frag)
        c = self._emit_expr(node.c_frag)
        return f"wmma::mma_sync({c}, {a}, {b}, {c});"

    def _emit_wmma_store(self, node) -> str:
        tensor = self._emit_expr(node.tensor)
        row = self._emit_expr(node.row)
        col = self._emit_expr(node.col)
        stride = self._emit_expr(node.stride)
        frag = self._emit_expr(node.frag)
        return f"wmma::store_matrix_sync({tensor} + {row} * {stride} + {col}, {frag}, {stride}, wmma::mem_row_major);"

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
        dtype = DTYPE_MAP.get(node.dtype)
        cuda_type = dtype.cuda_type if dtype else "float"
        return f"extern __shared__ {cuda_type} {name}[];"
