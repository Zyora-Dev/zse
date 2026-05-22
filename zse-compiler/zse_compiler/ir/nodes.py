"""ZSE IR Nodes — Backend-agnostic intermediate representation.

Every Python AST construct maps to one or more IR nodes.
Code generators walk these nodes to emit CUDA C / HIP C / MSL.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union


# --- Base ---

@dataclass
class IRNode:
    """Base class for all IR nodes."""
    pass


# --- Module / Function ---

@dataclass
class IRParam(IRNode):
    name: str
    dtype: str  # "tensor", "int", "float", etc.


@dataclass
class IRFunction(IRNode):
    name: str
    params: List[IRParam]
    body: List[IRNode]
    shared_mem: List['IRSharedMemDecl'] = field(default_factory=list)


@dataclass
class IRModule(IRNode):
    """Top-level container for multiple kernel functions."""
    functions: List[IRFunction] = field(default_factory=list)


# --- Expressions ---

@dataclass
class IRConst(IRNode):
    value: Any  # int, float


@dataclass
class IRVar(IRNode):
    name: str


@dataclass
class IRBinOp(IRNode):
    op: str  # "+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "<", "<=", ">", ">=", "==", "!=", "&&", "||"
    left: IRNode
    right: IRNode


@dataclass
class IRUnaryOp(IRNode):
    op: str  # "-", "!", "~"
    operand: IRNode


@dataclass
class IRCast(IRNode):
    dtype: str  # "int", "float"
    operand: IRNode


# --- Memory Access ---

@dataclass
class IRLoad(IRNode):
    """Load from tensor: a[i] or a[i, j]"""
    tensor: IRNode  # IRVar pointing to a tensor param
    index: List[IRNode]  # list of index expressions


@dataclass
class IRStore(IRNode):
    """Store to tensor: a[i] = value"""
    tensor: IRNode
    index: List[IRNode]
    value: IRNode


@dataclass
class IRIndex(IRNode):
    """Multi-dimensional index."""
    indices: List[IRNode]


# --- Assignments ---

@dataclass
class IRAssign(IRNode):
    """Variable assignment: x = expr"""
    name: str
    value: IRNode
    dtype: Optional[str] = None  # Type hint if annotated


# --- GPU Thread Primitives ---

@dataclass
class IRThreadIdx(IRNode):
    """threadIdx.x/y/z"""
    axis: int = 0  # 0=x, 1=y, 2=z


@dataclass
class IRBlockIdx(IRNode):
    """blockIdx.x/y/z"""
    axis: int = 0


@dataclass
class IRBlockDim(IRNode):
    """blockDim.x/y/z"""
    axis: int = 0


@dataclass
class IRGridDim(IRNode):
    """gridDim.x/y/z"""
    axis: int = 0


@dataclass
class IRGlobalId(IRNode):
    """blockIdx * blockDim + threadIdx"""
    axis: int = 0


# --- Shared Memory ---

@dataclass
class IRSharedMemDecl(IRNode):
    """Shared memory declaration."""
    shape: Tuple[int, ...]
    dtype: str = "float32"
    name: str = ""  # Assigned during codegen


# --- Synchronization ---

@dataclass
class IRBarrier(IRNode):
    """Thread barrier — __syncthreads()"""
    pass


# --- Atomics ---

@dataclass
class IRAtomicAdd(IRNode):
    ptr: IRNode
    value: IRNode


@dataclass
class IRAtomicMax(IRNode):
    ptr: IRNode
    value: IRNode


@dataclass
class IRAtomicMin(IRNode):
    ptr: IRNode
    value: IRNode


@dataclass
class IRAtomicCAS(IRNode):
    ptr: IRNode
    compare: IRNode
    value: IRNode


# --- Warp Primitives ---

@dataclass
class IRWarpShuffle(IRNode):
    """Warp shuffle — read value from another lane.
    variant: "down", "up", "xor", "idx"
    """
    variant: str  # "down", "up", "xor", "idx"
    value: IRNode
    offset: IRNode  # offset/lane_mask/src_lane depending on variant
    width: IRNode = field(default_factory=lambda: IRConst(value=32))


@dataclass
class IRWarpVote(IRNode):
    """Warp voting — ballot, all, any.
    variant: "ballot", "all", "any"
    """
    variant: str
    predicate: IRNode


@dataclass
class IRLaneId(IRNode):
    """Lane index within current warp (0-31)."""
    pass


@dataclass
class IRWarpId(IRNode):
    """Warp index within current block."""
    pass


@dataclass
class IRWarpReduce(IRNode):
    """Warp-level reduction — expands to butterfly shuffle loop.
    op: "sum", "max", "min"
    """
    op: str  # "sum", "max", "min"
    value: IRNode


@dataclass
class IRBlockReduce(IRNode):
    """Block-level reduction — warp reduce → shared mem → cross-warp reduce.
    op: "sum", "max", "min"
    """
    op: str
    value: IRNode


# --- Vectorized Memory Access ---

@dataclass
class IRLoadFloat4(IRNode):
    """Load 4 consecutive floats as 128-bit transaction."""
    tensor: IRNode
    index: IRNode


@dataclass
class IRStoreFloat4(IRNode):
    """Store 4 consecutive floats as 128-bit transaction."""
    tensor: IRNode
    index: IRNode
    values: List[IRNode] = field(default_factory=list)  # [v0, v1, v2, v3]


@dataclass
class IRLoadHalf2(IRNode):
    """Load 2 consecutive halfs as 32-bit transaction."""
    tensor: IRNode
    index: IRNode


@dataclass
class IRStoreHalf2(IRNode):
    """Store 2 consecutive halfs as 32-bit transaction."""
    tensor: IRNode
    index: IRNode
    values: List[IRNode] = field(default_factory=list)  # [v0, v1]


# --- Tiling ---

@dataclass
class IRTileLoad(IRNode):
    """Cooperative tile load from global to shared memory.

    If bound_row / bound_col are provided, threads whose target element
    lies outside the bounds skip the load and write 0.0f into the shared
    tile slot — this is required for matmul tail tiles where the problem
    size isn't a tile multiple.
    """
    tensor: IRNode
    tile_row: IRNode
    tile_col: IRNode
    tile_size: IRNode
    shared_buf: Optional[IRNode] = None
    bound_row: Optional[IRNode] = None
    bound_col: Optional[IRNode] = None


@dataclass
class IRTileStore(IRNode):
    """Store tile from shared memory to global memory.

    If bound_row / bound_col are provided, threads whose target element
    lies outside the bounds skip the store.
    """
    shared_buf: IRNode
    tensor: IRNode
    tile_row: IRNode
    tile_col: IRNode
    tile_size: IRNode
    bound_row: Optional[IRNode] = None
    bound_col: Optional[IRNode] = None


# --- Tensor Core / WMMA ---

@dataclass
class IRWmmaLoadA(IRNode):
    """Load matrix tile into WMMA fragment A."""
    tensor: IRNode
    row: IRNode
    col: IRNode
    stride: IRNode
    frag_name: str = ""  # assigned during codegen


@dataclass
class IRWmmaLoadB(IRNode):
    """Load matrix tile into WMMA fragment B."""
    tensor: IRNode
    row: IRNode
    col: IRNode
    stride: IRNode
    frag_name: str = ""


@dataclass
class IRWmmaFill(IRNode):
    """Initialize WMMA accumulator fragment."""
    value: IRNode
    frag_name: str = ""


@dataclass
class IRWmmaMMA(IRNode):
    """WMMA matrix multiply-accumulate: C = A * B + C"""
    a_frag: IRNode
    b_frag: IRNode
    c_frag: IRNode


@dataclass
class IRWmmaStore(IRNode):
    """Store WMMA accumulator to global memory."""
    tensor: IRNode
    row: IRNode
    col: IRNode
    stride: IRNode
    frag: IRNode


# --- AMD CDNA MFMA (Matrix Fused Multiply-Add) — Tier-4 ---

@dataclass
class IRMfmaOp(IRNode):
    """AMD CDNA matrix multiply-accumulate.

    Operands are existing ``local_array`` buffers (per-lane fragments). In ROCm
    codegen the buffers are copied into ``ext_vector_type`` vectors, the
    ``__builtin_amdgcn_mfma_*`` builtin is issued, then the result is written
    back into the C buffer in place.

    Shape is one of: "16x16x16_f16", "32x32x8_f16". Determines:
      - A,B lane fragment width (always 4 fp16 for these two shapes)
      - C lane fragment width (4 fp32 for 16x16x16, 16 fp32 for 32x32x8)
      - Which CDNA builtin is emitted
    """
    a_buf: IRNode      # local_array(4, float16) — A fragment per lane
    b_buf: IRNode      # local_array(4, float16) — B fragment per lane
    c_buf: IRNode      # local_array(4 or 16, float32) — C accumulator per lane (in/out)
    shape: str         # "16x16x16_f16" or "32x32x8_f16"


# --- Math Functions ---

@dataclass
class IRMathFunc(IRNode):
    """Intrinsic math function: exp, log, sqrt, rsqrt, max, min, fma"""
    name: str  # "exp", "log", "sqrt", "rsqrt", "max_val", "min_val", "fma"
    args: List[IRNode] = field(default_factory=list)


# --- Control Flow ---

@dataclass
class IRIf(IRNode):
    condition: IRNode
    then_body: List[IRNode]
    else_body: List[IRNode] = field(default_factory=list)
    is_ternary: bool = False


@dataclass
class IRFor(IRNode):
    var: str
    start: IRNode
    stop: IRNode
    step: IRNode
    body: List[IRNode] = field(default_factory=list)


@dataclass
class IRWhile(IRNode):
    condition: IRNode
    body: List[IRNode] = field(default_factory=list)


@dataclass
class IRReturn(IRNode):
    value: Optional[IRNode] = None


# --- FP16 Conversion ---

@dataclass
class IRHalfToFloat(IRNode):
    """Convert half → float: __half2float(x) in CUDA."""
    value: IRNode


@dataclass
class IRFloatToHalf(IRNode):
    """Convert float → half: __float2half(x) in CUDA."""
    value: IRNode


# --- Dynamic Shared Memory ---

@dataclass
class IRDynamicSharedMemDecl(IRNode):
    """Dynamic shared memory: extern __shared__ float smem[];
    Size is specified at kernel launch time via shared_mem_bytes.
    """
    dtype: str = "float32"
    name: str = ""


# --- INT4 nibble unpack (Tier-2 primitive for hand-tuned dequant kernels) ---

@dataclass
class IRUnpackInt4(IRNode):
    """Unpack 8 sign-extended 4-bit nibbles from a packed uint32 into out_buf.

    Writes 8 consecutive sign-extended int values to out_buf[base_idx..base_idx+8].
    Nibble i comes from bits [4*i .. 4*i+3] of packed (little-endian within u32).

    Lowered to an unrolled shift+mask+sign-extend sequence that NVRTC / HIPRTC
    optimize to native bit-field-extract instructions (`bfe.s32` on PTX,
    `v_bfe_i32` on AMDGCN). Metal uses the scalar form.

    This is the foundational primitive for fast INT4 dequant matmuls — avoids
    8 separate Python-DSL shift/mask expressions and gives the backend
    compiler a tight, recognizable pattern.
    """
    packed: IRNode      # uint32 source
    out_buf: IRNode     # destination buffer (shared or local)
    base_idx: IRNode    # starting index in out_buf


@dataclass
class IRUnpackUint4(IRNode):
    """Unpack 8 unsigned 4-bit nibbles (0..15) from a packed uint32 into out_buf.

    Identical to IRUnpackInt4 but without sign-extension — used for the
    asymmetric INT4 quant format where each nibble is in [0, 15] and the
    zero point is stored separately (as in our .zse INT4 weight layout).

    Lowering: unrolled (packed >> (4*i)) & 0xF. NVRTC -> bfe.u32,
    HIPRTC -> v_bfe_u32, Metal -> scalar shift+mask.
    """
    packed: IRNode
    out_buf: IRNode
    base_idx: IRNode


# --- Local register array (Tier-3 primitive — per-thread scratch in registers) ---

@dataclass
class IRLocalArrayDecl(IRNode):
    """Per-thread local-scope array. CUDA/HIP: `T name[size];` (stack —
    promoted to registers by ptxas/llvm when size is small and accesses
    are unrolled). Metal: `thread T name[size];`.

    Use for hand-tuned dequant kernels that need a per-thread N-nibble
    scratch buffer before doing the matmul accumulate.
    """
    name: str = ""
    size: int = 0
    dtype: str = "float32"


# --- Pointer reinterpret cast (Tier-3 primitive — vectorized weight loads) ---

@dataclass
class IRReinterpret(IRNode):
    """Reinterpret a pointer as a different element type — emits a C-style
    cast like `((unsigned int*)(weight_bytes))`. Used to pull 4 packed
    uint8 weights as one uint32 load (vectorized memory).

    Operand must be an existing tensor / pointer expression. Result is an
    expression of the new pointer type; assign to a local variable, then
    index normally: `qp = zse.reinterpret(w, zse.uint32); v = qp[i]`.
    """
    operand: IRNode
    dtype: str = "uint32"
