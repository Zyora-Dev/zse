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
    IRUnpackInt4, IRUnpackUint4,
    IRMfmaOp,
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

    def __init__(self):
        super().__init__()
        self._block_reduce_counter = 0
        self._unpack_int4_counter = 0
        self._unpack_uint4_counter = 0
        self._mfma_counter = 0

    def generate(self, func):
        self._block_reduce_counter = 0
        self._unpack_int4_counter = 0
        self._unpack_uint4_counter = 0
        self._mfma_counter = 0
        return super().generate(func)

    def _emit_header(self) -> str:
        return '#include <hip/hip_runtime.h>\n\nextern "C" {'

    def _emit_footer(self) -> str:
        return "}"

    def _emit_function_signature(self, func: IRFunction) -> str:
        params = ", ".join(self._param_to_str(p) for p in func.params)
        return f"__global__ void {func.name}({params})"

    def _param_to_str(self, param: IRParam) -> str:
        if param.dtype in ("tensor", "Tensor", "fp32_tensor", "float32_tensor"):
            return f"float* __restrict__ {param.name}"
        elif param.dtype in ("half_tensor", "fp16_tensor", "float16_tensor"):
            return f"half* __restrict__ {param.name}"
        elif param.dtype in ("bf16_tensor", "bfloat16_tensor"):
            return f"hip_bfloat16* __restrict__ {param.name}"
        elif param.dtype == "uint8_tensor":
            return f"unsigned char* __restrict__ {param.name}"
        elif param.dtype == "int8_tensor":
            return f"signed char* __restrict__ {param.name}"
        elif param.dtype == "uint16_tensor":
            return f"unsigned short* __restrict__ {param.name}"
        elif param.dtype == "int16_tensor":
            return f"short* __restrict__ {param.name}"
        elif param.dtype == "int32_tensor":
            return f"int* __restrict__ {param.name}"
        elif param.dtype == "uint32_tensor":
            return f"unsigned int* __restrict__ {param.name}"
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
        self._block_reduce_counter += 1
        cid = self._block_reduce_counter
        var = f"_zse_br_{cid}"
        bsmem = f"_zse_bsmem_{cid}"
        lid = f"_zse_lid_{cid}"
        wid = f"_zse_wid_{cid}"
        nwarps_v = f"_zse_nwarps_{cid}"
        nwarps = 16  # max 1024 threads / 64 wavefront = 16 wavefronts

        lines = []
        lines.append(f"[&]() {{")
        lines.append(f"  __shared__ float {bsmem}[{nwarps}];")
        lines.append(f"  float {var} = {val};")
        for offset in SHUFFLE_OFFSETS:
            shfl = f"__shfl_xor({var}, {offset}, {WARP_SIZE})"
            lines.append(f"  {var} = {combine(var, shfl)};")
        lines.append(f"  int {lid} = threadIdx.x % {WARP_SIZE};")
        lines.append(f"  int {wid} = threadIdx.x / {WARP_SIZE};")
        lines.append(f"  if ({lid} == 0) {bsmem}[{wid}] = {var};")
        lines.append(f"  __syncthreads();")
        lines.append(f"  int {nwarps_v} = (blockDim.x + {WARP_SIZE - 1}) / {WARP_SIZE};")
        lines.append(f"  {var} = ({lid} < {nwarps_v}) ? {bsmem}[{lid}] : {identity};")
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
        if node.bound_row is not None and node.bound_col is not None:
            br = self._emit_expr(node.bound_row)
            bc = self._emit_expr(node.bound_col)
            lines.append(f"{{ int _zse_gr = ({tr}) + threadIdx.y;")
            lines.append(f"  int _zse_gc = ({tc}) + threadIdx.x;")
            lines.append(f"  {sbuf}[threadIdx.y * {ts} + threadIdx.x] = "
                          f"(_zse_gr < ({br}) && _zse_gc < ({bc})) ? "
                          f"{tensor}[_zse_gr * ({bc}) + _zse_gc] : 0.0f; }}")
        else:
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
        if node.bound_row is not None and node.bound_col is not None:
            br = self._emit_expr(node.bound_row)
            bc = self._emit_expr(node.bound_col)
            lines.append(f"{{ int _zse_gr = ({tr}) + threadIdx.y;")
            lines.append(f"  int _zse_gc = ({tc}) + threadIdx.x;")
            lines.append(f"  if (_zse_gr < ({br}) && _zse_gc < ({bc})) "
                          f"{tensor}[_zse_gr * ({bc}) + _zse_gc] = "
                          f"{sbuf}[threadIdx.y * {ts} + threadIdx.x]; }}")
        else:
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

    # --- INT4 nibble unpack (Tier-2 primitive) ---

    def _emit_unpack_int4(self, node) -> str:
        self._unpack_int4_counter += 1
        cid = self._unpack_int4_counter
        packed = self._emit_expr(node.packed)
        out_buf = self._emit_expr(node.out_buf)
        base = self._emit_expr(node.base_idx)
        p = f"_zse_u4_p_{cid}"
        o = f"_zse_u4_o_{cid}"
        # Unrolled shift+mask+sign-extend. HIPRTC lowers to v_bfe_i32.
        return (
            f"{{ unsigned int {p} = (unsigned int)({packed}); int {o} = (int)({base});\n"
            + self._indented(f"  #pragma unroll\n")
            + self._indented(f"  for (int _zse_u4_i_{cid} = 0; _zse_u4_i_{cid} < 8; ++_zse_u4_i_{cid}) {{\n")
            + self._indented(f"    int _zse_u4_n_{cid} = (int)(({p} >> (_zse_u4_i_{cid} * 4)) & 0xF);\n")
            + self._indented(f"    _zse_u4_n_{cid} = (_zse_u4_n_{cid} ^ 0x8) - 0x8;\n")
            + self._indented(f"    ({out_buf})[{o} + _zse_u4_i_{cid}] = _zse_u4_n_{cid};\n")
            + self._indented(f"  }} }}")
        )

    def _emit_unpack_uint4(self, node) -> str:
        self._unpack_uint4_counter += 1
        cid = self._unpack_uint4_counter
        packed = self._emit_expr(node.packed)
        out_buf = self._emit_expr(node.out_buf)
        base = self._emit_expr(node.base_idx)
        p = f"_zse_uu4_p_{cid}"
        o = f"_zse_uu4_o_{cid}"
        # Unrolled shift+mask, no sign-extend. HIPRTC lowers to v_bfe_u32.
        return (
            f"{{ unsigned int {p} = (unsigned int)({packed}); int {o} = (int)({base});\n"
            + self._indented(f"  #pragma unroll\n")
            + self._indented(f"  for (int _zse_uu4_i_{cid} = 0; _zse_uu4_i_{cid} < 8; ++_zse_uu4_i_{cid}) {{\n")
            + self._indented(f"    int _zse_uu4_n_{cid} = (int)(({p} >> (_zse_uu4_i_{cid} * 4)) & 0xF);\n")
            + self._indented(f"    ({out_buf})[{o} + _zse_uu4_i_{cid}] = _zse_uu4_n_{cid};\n")
            + self._indented(f"  }} }}")
        )

    # --- Dynamic Shared Memory ---

    def _emit_dynamic_shared_mem_decl(self, node) -> str:
        self._shared_mem_counter += 1
        name = node.name or f"_zse_dsmem_{self._shared_mem_counter}"
        node.name = name
        from zse_compiler.types.dtypes import DTYPE_MAP as DM
        dtype = DM.get(node.dtype)
        hip_type = dtype.hip_type if dtype else "float"
        return f"extern __shared__ {hip_type} {name}[];"

    # --- AMD CDNA MFMA matrix cores (Tier-4) ---

    # shape -> (builtin name, A/B lane width, C lane width)
    _MFMA_SPECS = {
        "16x16x16_f16": ("__builtin_amdgcn_mfma_f32_16x16x16f16", 4, 4),
        "32x32x8_f16":  ("__builtin_amdgcn_mfma_f32_32x32x8f16",  4, 16),
    }

    def _emit_mfma_op(self, node) -> str:
        spec = self._MFMA_SPECS.get(node.shape)
        if spec is None:
            raise NotImplementedError(f"Unsupported MFMA shape: {node.shape}")
        builtin, ab_w, c_w = spec
        self._mfma_counter += 1
        cid = self._mfma_counter
        a = self._emit_expr(node.a_buf)
        b = self._emit_expr(node.b_buf)
        c = self._emit_expr(node.c_buf)
        av = f"_zse_mfma_a_{cid}"
        bv = f"_zse_mfma_b_{cid}"
        cv = f"_zse_mfma_c_{cid}"
        # HIP `half` is `__half` (struct wrapper); ext_vector of __fp16 needs the raw
        # 16-bit value. Round-trip via __half2float -> (__fp16) cast is the portable
        # bridge and the optimizer folds it to a no-op move.
        a_init = ", ".join(f"(__fp16)__half2float(({a})[{i}])" for i in range(ab_w))
        b_init = ", ".join(f"(__fp16)__half2float(({b})[{i}])" for i in range(ab_w))
        c_init = ", ".join(f"({c})[{i}]" for i in range(c_w))
        # Block scope keeps multiple MFMA calls per kernel from colliding.
        # __fp16 (== _Float16) is HIP's fp16; ext_vector_type makes the SIMD vector.
        lines = [
            "{",
            f"  typedef __fp16 _zse_mfma_h{ab_w}_t_{cid} __attribute__((ext_vector_type({ab_w})));",
            f"  typedef float _zse_mfma_f{c_w}_t_{cid} __attribute__((ext_vector_type({c_w})));",
            f"  _zse_mfma_h{ab_w}_t_{cid} {av} = {{ {a_init} }};",
            f"  _zse_mfma_h{ab_w}_t_{cid} {bv} = {{ {b_init} }};",
            f"  _zse_mfma_f{c_w}_t_{cid} {cv} = {{ {c_init} }};",
            f"  {cv} = {builtin}({av}, {bv}, {cv}, 0, 0, 0);",
            f"  #pragma unroll",
            f"  for (int _zse_mfma_i_{cid} = 0; _zse_mfma_i_{cid} < {c_w}; ++_zse_mfma_i_{cid}) {{",
            f"    ({c})[_zse_mfma_i_{cid}] = {cv}[_zse_mfma_i_{cid}];",
            f"  }}",
            f"}}",
        ]
        return "\n".join(self._indented(l) if i > 0 else l for i, l in enumerate(lines))
