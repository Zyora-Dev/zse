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
    """Cooperative tile load from global to shared memory."""
    tensor: IRNode
    tile_row: IRNode
    tile_col: IRNode
    tile_size: IRNode
    shared_buf: Optional[IRNode] = None


@dataclass
class IRTileStore(IRNode):
    """Store tile from shared memory to global memory."""
    shared_buf: IRNode
    tensor: IRNode
    tile_row: IRNode
    tile_col: IRNode
    tile_size: IRNode


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
