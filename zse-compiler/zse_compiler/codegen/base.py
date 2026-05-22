"""Base code generator — shared logic for all backends."""

from typing import List, Dict
from zse_compiler.ir.nodes import (
    IRNode, IRFunction, IRParam,
    IRBinOp, IRUnaryOp, IRLoad, IRStore, IRConst,
    IRVar, IRThreadIdx, IRBlockIdx, IRBlockDim, IRGridDim, IRGlobalId,
    IRSharedMemDecl, IRBarrier,
    IRAtomicAdd, IRAtomicMax, IRAtomicMin, IRAtomicCAS, IRMathFunc,
    IRIf, IRFor, IRWhile, IRReturn, IRCast, IRAssign,
    # Warp primitives
    IRWarpShuffle, IRWarpVote, IRLaneId, IRWarpId,
    IRWarpReduce, IRBlockReduce,
    # Vectorized memory
    IRLoadFloat4, IRStoreFloat4, IRLoadHalf2, IRStoreHalf2,
    # Tiling
    IRTileLoad, IRTileStore,
    # WMMA
    IRWmmaLoadA, IRWmmaLoadB, IRWmmaFill, IRWmmaMMA, IRWmmaStore,
    # CDNA MFMA (Tier-4)
    IRMfmaOp,
    # FP16
    IRHalfToFloat, IRFloatToHalf,
    # Dynamic shared memory
    IRDynamicSharedMemDecl,
    # INT4 unpack
    IRUnpackInt4,
    IRUnpackUint4,
    # Local register array + pointer reinterpret
    IRLocalArrayDecl,
    IRReinterpret,
)
from zse_compiler.ir.type_inference import infer_types


class BaseCodegen:
    """Base code generator — subclassed per backend."""

    def __init__(self):
        self._indent = 0
        self._shared_mem_counter = 0
        self._local_vars = set()
        self._var_types: Dict[str, str] = {}  # inferred types
        self._needs_block_reduce_smem = False

    def generate(self, func: IRFunction) -> str:
        """Generate complete source code for a kernel function."""
        self._local_vars = set()
        self._shared_mem_counter = 0
        self._needs_block_reduce_smem = False
        self._var_types = infer_types(func)

        header = self._emit_header()
        signature = self._emit_function_signature(func)
        body = self._emit_body(func.body)
        footer = self._emit_footer()

        return f"{header}\n{signature} {{\n{body}\n}}\n{footer}"

    def _emit_header(self) -> str:
        return ""

    def _emit_footer(self) -> str:
        return ""

    def _emit_function_signature(self, func: IRFunction) -> str:
        raise NotImplementedError

    def _emit_body(self, stmts: List[IRNode]) -> str:
        self._indent += 1
        lines = []
        for stmt in stmts:
            code = self._emit_node(stmt)
            if code:
                lines.append(self._indented(code))
        self._indent -= 1
        return "\n".join(lines)

    def _emit_node(self, node: IRNode) -> str:
        if isinstance(node, IRAssign):
            return self._emit_assign(node)
        elif isinstance(node, IRStore):
            return self._emit_store(node)
        elif isinstance(node, IRStoreFloat4):
            return self._emit_store_float4(node)
        elif isinstance(node, IRStoreHalf2):
            return self._emit_store_half2(node)
        elif isinstance(node, IRIf):
            return self._emit_if(node)
        elif isinstance(node, IRFor):
            return self._emit_for(node)
        elif isinstance(node, IRWhile):
            return self._emit_while(node)
        elif isinstance(node, IRReturn):
            return self._emit_return(node)
        elif isinstance(node, IRBarrier):
            return self._emit_barrier()
        elif isinstance(node, IRSharedMemDecl):
            return self._emit_shared_mem_decl(node)
        elif isinstance(node, IRDynamicSharedMemDecl):
            return self._emit_dynamic_shared_mem_decl(node)
        elif isinstance(node, IRAtomicAdd):
            return self._emit_atomic_add(node)
        elif isinstance(node, IRAtomicMax):
            return self._emit_atomic_max(node)
        elif isinstance(node, IRAtomicMin):
            return self._emit_atomic_min(node)
        elif isinstance(node, IRTileLoad):
            return self._emit_tile_load(node)
        elif isinstance(node, IRTileStore):
            return self._emit_tile_store(node)
        elif isinstance(node, IRWmmaLoadA):
            return self._emit_wmma_load_a(node)
        elif isinstance(node, IRWmmaLoadB):
            return self._emit_wmma_load_b(node)
        elif isinstance(node, IRWmmaFill):
            return self._emit_wmma_fill(node)
        elif isinstance(node, IRWmmaMMA):
            return self._emit_wmma_mma(node)
        elif isinstance(node, IRWmmaStore):
            return self._emit_wmma_store(node)
        elif isinstance(node, IRMfmaOp):
            return self._emit_mfma_op(node)
        elif isinstance(node, IRUnpackInt4):
            return self._emit_unpack_int4(node)
        elif isinstance(node, IRUnpackUint4):
            return self._emit_unpack_uint4(node)
        elif isinstance(node, IRLocalArrayDecl):
            return self._emit_local_array_decl(node)
        else:
            # Expression statement
            return self._emit_expr(node) + ";"

    def _emit_expr(self, node: IRNode) -> str:
        if isinstance(node, IRConst):
            return self._emit_const(node)
        elif isinstance(node, IRVar):
            return node.name
        elif isinstance(node, IRBinOp):
            left = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            return f"({left} {node.op} {right})"
        elif isinstance(node, IRUnaryOp):
            operand = self._emit_expr(node.operand)
            return f"({node.op}{operand})"
        elif isinstance(node, IRCast):
            operand = self._emit_expr(node.operand)
            return self._emit_cast(node.dtype, operand)
        elif isinstance(node, IRLoad):
            return self._emit_load(node)
        elif isinstance(node, IRThreadIdx):
            return self._emit_thread_idx(node.axis)
        elif isinstance(node, IRBlockIdx):
            return self._emit_block_idx(node.axis)
        elif isinstance(node, IRBlockDim):
            return self._emit_block_dim(node.axis)
        elif isinstance(node, IRGridDim):
            return self._emit_grid_dim(node.axis)
        elif isinstance(node, IRGlobalId):
            return f"({self._emit_block_idx(node.axis)} * {self._emit_block_dim(node.axis)} + {self._emit_thread_idx(node.axis)})"
        elif isinstance(node, IRMathFunc):
            return self._emit_math_func(node)
        elif isinstance(node, IRAtomicAdd):
            return self._emit_atomic_add_expr(node)
        elif isinstance(node, IRAtomicMax):
            return self._emit_atomic_max_expr(node)
        elif isinstance(node, IRAtomicMin):
            return self._emit_atomic_min_expr(node)
        elif isinstance(node, IRAtomicCAS):
            return self._emit_atomic_cas_expr(node)
        elif isinstance(node, IRIf) and node.is_ternary:
            cond = self._emit_expr(node.condition)
            t = self._emit_expr(node.then_body[0])
            f = self._emit_expr(node.else_body[0])
            return f"({cond} ? {t} : {f})"
        # Warp primitives
        elif isinstance(node, IRLaneId):
            return self._emit_lane_id()
        elif isinstance(node, IRWarpId):
            return self._emit_warp_id()
        elif isinstance(node, IRWarpShuffle):
            return self._emit_warp_shuffle(node)
        elif isinstance(node, IRWarpVote):
            return self._emit_warp_vote(node)
        elif isinstance(node, IRWarpReduce):
            return self._emit_warp_reduce(node)
        elif isinstance(node, IRBlockReduce):
            return self._emit_block_reduce(node)
        # Vectorized memory
        elif isinstance(node, IRLoadFloat4):
            return self._emit_load_float4(node)
        elif isinstance(node, IRLoadHalf2):
            return self._emit_load_half2(node)
        elif isinstance(node, IRStoreFloat4):
            return self._emit_store_float4_expr(node)
        # FP16 conversion
        elif isinstance(node, IRHalfToFloat):
            return self._emit_half_to_float(node)
        elif isinstance(node, IRFloatToHalf):
            return self._emit_float_to_half(node)
        elif isinstance(node, IRDynamicSharedMemDecl):
            return node.name or "_zse_dsmem"
        elif isinstance(node, IRReinterpret):
            return self._emit_reinterpret(node)
        else:
            raise ValueError(f"Cannot emit expression for {type(node).__name__}")

    def _emit_const(self, node: IRConst) -> str:
        if isinstance(node.value, float):
            return f"{node.value}f"
        return str(node.value)

    def _emit_assign(self, node: IRAssign) -> str:
        value = self._emit_expr(node.value)
        if node.name not in self._local_vars:
            self._local_vars.add(node.name)
            # Special case: RHS is a pointer reinterpret — emit explicit pointer type
            # so Metal (which lacks `auto`) and code clarity both work.
            if isinstance(node.value, IRReinterpret):
                ptr_type = self._reinterpret_lhs_type(node.value.dtype)
                return f"{ptr_type} {node.name} = {value};"
            # Use inferred type if available, fall back to annotation or auto
            inferred = self._var_types.get(node.name)
            if inferred:
                dtype = self._map_type(inferred)
            elif node.dtype:
                dtype = self._map_type(node.dtype)
            else:
                dtype = self._default_var_type()
            return f"{dtype} {node.name} = {value};"
        return f"{node.name} = {value};"

    def _reinterpret_lhs_type(self, dtype: str) -> str:
        """LHS pointer type for `auto`-less backends. Override in Metal."""
        from zse_compiler.types.dtypes import DTYPE_MAP
        if dtype in DTYPE_MAP:
            return f"{DTYPE_MAP[dtype].cuda_type}*"
        return "auto"

    def _emit_local_array_decl(self, node) -> str:
        """Per-thread local-scope array — stack-allocated; CUDA/HIP ptxas
        promotes small fully-unrolled arrays to registers."""
        from zse_compiler.types.dtypes import DTYPE_MAP
        ctype = DTYPE_MAP[node.dtype].cuda_type if node.dtype in DTYPE_MAP else "float"
        self._local_vars.add(node.name)
        return f"{ctype} {node.name}[{node.size}];"

    def _emit_reinterpret(self, node) -> str:
        """Pointer reinterpret cast — `((T*)(operand))`. Override in Metal
        to preserve `device` address space."""
        from zse_compiler.types.dtypes import DTYPE_MAP
        ctype = DTYPE_MAP[node.dtype].cuda_type if node.dtype in DTYPE_MAP else "float"
        operand = self._emit_expr(node.operand)
        return f"(({ctype}*)({operand}))"

    def _map_type(self, t: str) -> str:
        """Map abstract type to C type. Override in backends for specifics."""
        mapping = {
            "int": "int",
            "uint": "unsigned int",
            "float": "float",
            "float4": "float4",
            "half2": "half2",
        }
        if t in mapping:
            return mapping[t]
        # Fall through to DTYPE_MAP for explicit dtype names like
        # "int32", "uint32", "int16", "uint16", "int8", "uint8", "float16", "bfloat16".
        from zse_compiler.types.dtypes import DTYPE_MAP
        if t in DTYPE_MAP:
            return DTYPE_MAP[t].cuda_type
        return self._default_var_type()

    def _default_var_type(self) -> str:
        """Default type for untyped variables. CUDA/HIP use auto, Metal uses float."""
        return "auto"

    def _emit_load(self, node: IRLoad) -> str:
        tensor = self._emit_expr(node.tensor)
        indices = [self._emit_expr(i) for i in node.index]
        if len(indices) == 1:
            return f"{tensor}[{indices[0]}]"
        return f"{tensor}[{' + '.join(indices)}]"

    def _emit_store(self, node: IRStore) -> str:
        tensor = self._emit_expr(node.tensor)
        indices = [self._emit_expr(i) for i in node.index]
        value = self._emit_expr(node.value)
        idx = indices[0] if len(indices) == 1 else " + ".join(indices)
        return f"{tensor}[{idx}] = {value};"

    def _emit_if(self, node: IRIf) -> str:
        cond = self._emit_expr(node.condition)
        lines = [f"if ({cond}) {{"]
        self._indent += 1
        for s in node.then_body:
            lines.append(self._indented(self._emit_node(s)))
        self._indent -= 1
        if node.else_body:
            lines.append(self._indented("} else {", -0))
            self._indent += 1
            for s in node.else_body:
                lines.append(self._indented(self._emit_node(s)))
            self._indent -= 1
        lines.append(self._indented("}", -0))
        return "\n".join(lines)

    def _emit_for(self, node: IRFor) -> str:
        start = self._emit_expr(node.start)
        stop = self._emit_expr(node.stop)
        step = self._emit_expr(node.step)
        var = node.var
        self._local_vars.add(var)

        lines = [f"for (int {var} = {start}; {var} < {stop}; {var} += {step}) {{"]
        self._indent += 1
        for s in node.body:
            lines.append(self._indented(self._emit_node(s)))
        self._indent -= 1
        lines.append(self._indented("}", -0))
        return "\n".join(lines)

    def _emit_while(self, node: IRWhile) -> str:
        cond = self._emit_expr(node.condition)
        lines = [f"while ({cond}) {{"]
        self._indent += 1
        for s in node.body:
            lines.append(self._indented(self._emit_node(s)))
        self._indent -= 1
        lines.append(self._indented("}", -0))
        return "\n".join(lines)

    def _emit_return(self, node: IRReturn) -> str:
        if node.value:
            return f"return {self._emit_expr(node.value)};"
        return "return;"

    def _emit_cast(self, dtype: str, operand: str) -> str:
        type_map = {"int": "int", "float": "float"}
        return f"(({type_map.get(dtype, dtype)}){operand})"

    def _indented(self, text: str, extra: int = 0) -> str:
        return "    " * (self._indent + extra) + text

    # --- Backend-specific (must override) ---

    def _emit_thread_idx(self, axis: int) -> str:
        raise NotImplementedError

    def _emit_block_idx(self, axis: int) -> str:
        raise NotImplementedError

    def _emit_block_dim(self, axis: int) -> str:
        raise NotImplementedError

    def _emit_grid_dim(self, axis: int) -> str:
        raise NotImplementedError

    def _emit_barrier(self) -> str:
        raise NotImplementedError

    def _emit_shared_mem_decl(self, node: IRSharedMemDecl) -> str:
        raise NotImplementedError

    def _emit_atomic_add(self, node: IRAtomicAdd) -> str:
        raise NotImplementedError

    def _emit_atomic_add_expr(self, node: IRAtomicAdd) -> str:
        raise NotImplementedError

    def _emit_atomic_max(self, node) -> str:
        raise NotImplementedError

    def _emit_atomic_max_expr(self, node) -> str:
        raise NotImplementedError

    def _emit_atomic_min(self, node) -> str:
        raise NotImplementedError

    def _emit_atomic_min_expr(self, node) -> str:
        raise NotImplementedError

    def _emit_atomic_cas_expr(self, node) -> str:
        raise NotImplementedError

    def _emit_math_func(self, node: IRMathFunc) -> str:
        raise NotImplementedError

    def _emit_param_to_str(self, param: IRParam) -> str:
        raise NotImplementedError

    # --- Warp primitives (must override) ---

    def _emit_lane_id(self) -> str:
        raise NotImplementedError

    def _emit_warp_id(self) -> str:
        raise NotImplementedError

    def _emit_warp_shuffle(self, node: IRWarpShuffle) -> str:
        raise NotImplementedError

    def _emit_warp_vote(self, node: IRWarpVote) -> str:
        raise NotImplementedError

    def _emit_warp_reduce(self, node: IRWarpReduce) -> str:
        raise NotImplementedError

    def _emit_block_reduce(self, node: IRBlockReduce) -> str:
        raise NotImplementedError

    # --- Vectorized memory (must override) ---

    def _emit_load_float4(self, node: IRLoadFloat4) -> str:
        raise NotImplementedError

    def _emit_store_float4(self, node: IRStoreFloat4) -> str:
        raise NotImplementedError

    def _emit_store_float4_expr(self, node: IRStoreFloat4) -> str:
        raise NotImplementedError

    def _emit_load_half2(self, node: IRLoadHalf2) -> str:
        raise NotImplementedError

    def _emit_store_half2(self, node: IRStoreHalf2) -> str:
        raise NotImplementedError

    # --- Tiling (must override) ---

    def _emit_tile_load(self, node: IRTileLoad) -> str:
        raise NotImplementedError

    def _emit_tile_store(self, node: IRTileStore) -> str:
        raise NotImplementedError

    # --- WMMA / Tensor Core (must override) ---

    def _emit_wmma_load_a(self, node) -> str:
        raise NotImplementedError

    def _emit_wmma_load_b(self, node) -> str:
        raise NotImplementedError

    def _emit_wmma_fill(self, node) -> str:
        raise NotImplementedError

    def _emit_wmma_mma(self, node) -> str:
        raise NotImplementedError

    def _emit_wmma_store(self, node) -> str:
        raise NotImplementedError

    # --- AMD CDNA MFMA (must override in backends that support it) ---

    def _emit_mfma_op(self, node) -> str:
        raise NotImplementedError(
            f"MFMA ({node.shape}) is an AMD CDNA matrix-core intrinsic; "
            f"only the ROCm backend supports it. Use wmma_* primitives for NVIDIA / Metal."
        )

    # --- FP16 Conversion (must override) ---

    def _emit_half_to_float(self, node) -> str:
        raise NotImplementedError

    def _emit_float_to_half(self, node) -> str:
        raise NotImplementedError

    # --- Dynamic Shared Memory (must override) ---

    def _emit_dynamic_shared_mem_decl(self, node) -> str:
        raise NotImplementedError
