"""ZSE AST Parser — Reads Python @zse.kernel functions and converts to ZSE IR.

Flow: Python source → ast.parse → walk AST → emit IR nodes

We parse the function's AST and map:
- Variable assignments → IR Store
- Tensor indexing (a[i]) → IR Load/Store
- Arithmetic ops → IR BinOp
- zse.thread_id() calls → IR ThreadIdx
- zse.shared_memory() → IR SharedMemDecl
- if/for/while → IR control flow
- zse.syncthreads() → IR Barrier
"""

import ast
import inspect
import textwrap
from typing import Dict, List, Optional, Any

from zse_compiler.ir.nodes import (
    IRModule, IRFunction, IRParam,
    IRBinOp, IRUnaryOp, IRLoad, IRStore, IRConst,
    IRVar, IRIndex, IRThreadIdx, IRBlockIdx, IRBlockDim, IRGridDim, IRGlobalId,
    IRSharedMemDecl, IRBarrier,
    IRAtomicAdd, IRAtomicMax, IRAtomicMin, IRAtomicCAS, IRMathFunc,
    IRIf, IRFor, IRWhile, IRReturn,
    IRCast, IRAssign,
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
    IRLocalArrayDecl,
    IRReinterpret,
)
from zse_compiler.types.primitives import PRIMITIVE_FUNCTIONS
from zse_compiler.types.dtypes import DType, DTYPE_MAP


class KernelParser:
    """Parses a Python function into ZSE IR."""

    def __init__(self):
        self._var_types: Dict[str, str] = {}  # variable name → inferred type
        self._params: List[IRParam] = []
        self._shared_mem_decls: List[IRSharedMemDecl] = []

    def parse(self, func) -> IRFunction:
        """Parse a decorated Python function into IR."""
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Find the function definition (skip decorators)
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is None:
            raise SyntaxError(f"No function definition found in {func.__name__}")

        # Parse parameters
        self._params = self._parse_params(func_def)

        # Parse body statements
        body = []
        for stmt in func_def.body:
            ir_nodes = self._parse_stmt(stmt)
            if isinstance(ir_nodes, list):
                body.extend(ir_nodes)
            elif ir_nodes is not None:
                body.append(ir_nodes)

        return IRFunction(
            name=func_def.name,
            params=self._params,
            body=body,
            shared_mem=self._shared_mem_decls,
        )

    def _parse_params(self, func_def: ast.FunctionDef) -> List[IRParam]:
        """Extract function parameters with type annotations."""
        params = []
        for arg in func_def.args.args:
            name = arg.arg
            dtype_hint = "tensor"  # default

            if arg.annotation:
                dtype_hint = self._resolve_annotation(arg.annotation)

            params.append(IRParam(name=name, dtype=dtype_hint))
            self._var_types[name] = dtype_hint

        return params

    def _resolve_annotation(self, node: ast.expr) -> str:
        """Resolve type annotation to a string."""
        if isinstance(node, ast.Attribute):
            # zse.Tensor, zse.float32, etc.
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "tensor"

    def _parse_stmt(self, node: ast.stmt):
        """Parse a statement into IR node(s)."""
        if isinstance(node, ast.Assign):
            return self._parse_assign(node)
        elif isinstance(node, ast.AugAssign):
            return self._parse_aug_assign(node)
        elif isinstance(node, ast.Expr):
            return self._parse_expr_stmt(node)
        elif isinstance(node, ast.If):
            return self._parse_if(node)
        elif isinstance(node, ast.For):
            return self._parse_for(node)
        elif isinstance(node, ast.While):
            return self._parse_while(node)
        elif isinstance(node, ast.Return):
            return self._parse_return(node)
        elif isinstance(node, ast.Pass):
            return None  # pass → no-op
        elif isinstance(node, ast.AnnAssign):
            return self._parse_ann_assign(node)
        else:
            raise SyntaxError(f"Unsupported statement: {ast.dump(node)}")

    def _parse_assign(self, node: ast.Assign):
        """Parse assignment: x = expr or a[i] = expr"""
        if len(node.targets) != 1:
            raise SyntaxError("Multiple assignment targets not supported")

        target = node.targets[0]
        value = self._parse_expr(node.value)

        if isinstance(target, ast.Subscript):
            # a[i] = expr → Store
            tensor = self._parse_expr(target.value)
            index = self._parse_subscript_index(target)
            return IRStore(tensor=tensor, index=index, value=value)
        elif isinstance(target, ast.Name):
            # Special case: dynamic shared memory → set name on decl, return directly
            if isinstance(value, IRDynamicSharedMemDecl):
                value.name = target.id
                return value
            # Special case: static shared memory → set name on decl, return directly
            if isinstance(value, IRSharedMemDecl):
                value.name = target.id
                self._var_types[target.id] = "ptr"
                return value
            # Special case: local register array → set name on decl, return directly
            if isinstance(value, IRLocalArrayDecl):
                value.name = target.id
                self._var_types[target.id] = "ptr"
                return value
            # Special case: pointer reinterpret → mark var as pointer for type emit
            if isinstance(value, IRReinterpret):
                self._var_types[target.id] = "ptr"
                return IRAssign(name=target.id, value=value)
            # x = expr → Assign
            self._var_types[target.id] = "auto"
            return IRAssign(name=target.id, value=value)
        else:
            raise SyntaxError(f"Unsupported assignment target: {ast.dump(target)}")

    def _parse_aug_assign(self, node: ast.AugAssign):
        """Parse augmented assignment: x += expr"""
        target_expr = self._parse_expr(node.target)
        value = self._parse_expr(node.value)
        op = self._binop_str(node.op)

        if isinstance(node.target, ast.Subscript):
            tensor = self._parse_expr(node.target.value)
            index = self._parse_subscript_index(node.target)
            combined = IRBinOp(op=op, left=IRLoad(tensor=tensor, index=index), right=value)
            return IRStore(tensor=tensor, index=index, value=combined)
        elif isinstance(node.target, ast.Name):
            combined = IRBinOp(op=op, left=IRVar(name=node.target.id), right=value)
            return IRAssign(name=node.target.id, value=combined)
        else:
            raise SyntaxError(f"Unsupported aug assign target")

    def _parse_expr_stmt(self, node: ast.Expr):
        """Parse expression statement (function calls like syncthreads())."""
        return self._parse_expr(node.value)

    def _parse_expr(self, node: ast.expr):
        """Parse an expression into an IR node."""
        if isinstance(node, ast.Constant):
            return IRConst(value=node.value)

        elif isinstance(node, ast.Name):
            return IRVar(name=node.id)

        elif isinstance(node, ast.BinOp):
            left = self._parse_expr(node.left)
            right = self._parse_expr(node.right)
            op = self._binop_str(node.op)
            return IRBinOp(op=op, left=left, right=right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._parse_expr(node.operand)
            op = self._unaryop_str(node.op)
            return IRUnaryOp(op=op, operand=operand)

        elif isinstance(node, ast.Compare):
            # Simple comparison: a < b
            if len(node.ops) != 1:
                raise SyntaxError("Chained comparisons not supported in kernels")
            left = self._parse_expr(node.left)
            right = self._parse_expr(node.comparators[0])
            op = self._cmpop_str(node.ops[0])
            return IRBinOp(op=op, left=left, right=right)

        elif isinstance(node, ast.Subscript):
            tensor = self._parse_expr(node.value)
            index = self._parse_subscript_index(node)
            return IRLoad(tensor=tensor, index=index)

        elif isinstance(node, ast.Call):
            return self._parse_call(node)

        elif isinstance(node, ast.IfExp):
            # Ternary: a if cond else b
            cond = self._parse_expr(node.test)
            true_val = self._parse_expr(node.body)
            false_val = self._parse_expr(node.orelse)
            return IRIf(condition=cond, then_body=[true_val], else_body=[false_val], is_ternary=True)

        elif isinstance(node, ast.BoolOp):
            # and / or
            op = "&&" if isinstance(node.op, ast.And) else "||"
            left = self._parse_expr(node.values[0])
            for val in node.values[1:]:
                right = self._parse_expr(val)
                left = IRBinOp(op=op, left=left, right=right)
            return left

        else:
            raise SyntaxError(f"Unsupported expression: {ast.dump(node)}")

    def _parse_call(self, node: ast.Call):
        """Parse function calls — distinguish ZSE primitives from regular calls."""
        func_name = self._get_call_name(node)

        if func_name in ("thread_id", "zse.thread_id"):
            axis = self._get_axis_arg(node)
            return IRThreadIdx(axis=axis)

        elif func_name in ("block_id", "zse.block_id"):
            axis = self._get_axis_arg(node)
            return IRBlockIdx(axis=axis)

        elif func_name in ("block_dim", "zse.block_dim"):
            axis = self._get_axis_arg(node)
            return IRBlockDim(axis=axis)

        elif func_name in ("grid_dim", "zse.grid_dim"):
            axis = self._get_axis_arg(node)
            return IRGridDim(axis=axis)

        elif func_name in ("global_id", "zse.global_id"):
            axis = self._get_axis_arg(node)
            return IRGlobalId(axis=axis)

        elif func_name in ("syncthreads", "zse.syncthreads"):
            return IRBarrier()

        elif func_name in ("shared_memory", "zse.shared_memory"):
            return self._parse_shared_memory(node)

        elif func_name in ("atomic_add", "zse.atomic_add"):
            if len(node.args) != 2:
                raise SyntaxError("atomic_add requires 2 arguments: (ptr, value)")
            ptr = self._parse_expr(node.args[0])
            val = self._parse_expr(node.args[1])
            return IRAtomicAdd(ptr=ptr, value=val)

        elif func_name in ("atomic_max", "zse.atomic_max"):
            ptr = self._parse_expr(node.args[0])
            val = self._parse_expr(node.args[1])
            return IRAtomicMax(ptr=ptr, value=val)

        elif func_name in ("atomic_min", "zse.atomic_min"):
            ptr = self._parse_expr(node.args[0])
            val = self._parse_expr(node.args[1])
            return IRAtomicMin(ptr=ptr, value=val)

        elif func_name in ("atomic_cas", "zse.atomic_cas"):
            ptr = self._parse_expr(node.args[0])
            cmp = self._parse_expr(node.args[1])
            val = self._parse_expr(node.args[2])
            return IRAtomicCAS(ptr=ptr, compare=cmp, value=val)

        # --- Warp primitives ---

        elif func_name in ("lane_id", "zse.lane_id"):
            return IRLaneId()

        elif func_name in ("warp_id", "zse.warp_id"):
            return IRWarpId()

        elif func_name in ("warp_shuffle_down", "zse.warp_shuffle_down"):
            val = self._parse_expr(node.args[0])
            offset = self._parse_expr(node.args[1])
            width = self._parse_expr(node.args[2]) if len(node.args) > 2 else IRConst(value=32)
            return IRWarpShuffle(variant="down", value=val, offset=offset, width=width)

        elif func_name in ("warp_shuffle_up", "zse.warp_shuffle_up"):
            val = self._parse_expr(node.args[0])
            offset = self._parse_expr(node.args[1])
            width = self._parse_expr(node.args[2]) if len(node.args) > 2 else IRConst(value=32)
            return IRWarpShuffle(variant="up", value=val, offset=offset, width=width)

        elif func_name in ("warp_shuffle_xor", "zse.warp_shuffle_xor"):
            val = self._parse_expr(node.args[0])
            mask = self._parse_expr(node.args[1])
            width = self._parse_expr(node.args[2]) if len(node.args) > 2 else IRConst(value=32)
            return IRWarpShuffle(variant="xor", value=val, offset=mask, width=width)

        elif func_name in ("warp_shuffle", "zse.warp_shuffle"):
            val = self._parse_expr(node.args[0])
            src = self._parse_expr(node.args[1])
            width = self._parse_expr(node.args[2]) if len(node.args) > 2 else IRConst(value=32)
            return IRWarpShuffle(variant="idx", value=val, offset=src, width=width)

        elif func_name in ("warp_ballot", "zse.warp_ballot"):
            pred = self._parse_expr(node.args[0])
            return IRWarpVote(variant="ballot", predicate=pred)

        elif func_name in ("warp_all", "zse.warp_all"):
            pred = self._parse_expr(node.args[0])
            return IRWarpVote(variant="all", predicate=pred)

        elif func_name in ("warp_any", "zse.warp_any"):
            pred = self._parse_expr(node.args[0])
            return IRWarpVote(variant="any", predicate=pred)

        # --- Warp / Block reductions ---

        elif func_name in ("warp_reduce_sum", "zse.warp_reduce_sum"):
            val = self._parse_expr(node.args[0])
            return IRWarpReduce(op="sum", value=val)

        elif func_name in ("warp_reduce_max", "zse.warp_reduce_max"):
            val = self._parse_expr(node.args[0])
            return IRWarpReduce(op="max", value=val)

        elif func_name in ("warp_reduce_min", "zse.warp_reduce_min"):
            val = self._parse_expr(node.args[0])
            return IRWarpReduce(op="min", value=val)

        elif func_name in ("block_reduce_sum", "zse.block_reduce_sum"):
            val = self._parse_expr(node.args[0])
            return IRBlockReduce(op="sum", value=val)

        elif func_name in ("block_reduce_max", "zse.block_reduce_max"):
            val = self._parse_expr(node.args[0])
            return IRBlockReduce(op="max", value=val)

        elif func_name in ("block_reduce_min", "zse.block_reduce_min"):
            val = self._parse_expr(node.args[0])
            return IRBlockReduce(op="min", value=val)

        # --- Vectorized memory ---

        elif func_name in ("load_float4", "zse.load_float4"):
            tensor = self._parse_expr(node.args[0])
            idx = self._parse_expr(node.args[1])
            return IRLoadFloat4(tensor=tensor, index=idx)

        elif func_name in ("store_float4", "zse.store_float4"):
            tensor = self._parse_expr(node.args[0])
            idx = self._parse_expr(node.args[1])
            values = [self._parse_expr(node.args[i]) for i in range(2, 6)]
            return IRStoreFloat4(tensor=tensor, index=idx, values=values)

        elif func_name in ("load_half2", "zse.load_half2"):
            tensor = self._parse_expr(node.args[0])
            idx = self._parse_expr(node.args[1])
            return IRLoadHalf2(tensor=tensor, index=idx)

        elif func_name in ("store_half2", "zse.store_half2"):
            tensor = self._parse_expr(node.args[0])
            idx = self._parse_expr(node.args[1])
            values = [self._parse_expr(node.args[i]) for i in range(2, 4)]
            return IRStoreHalf2(tensor=tensor, index=idx, values=values)

        # --- Tiling ---

        elif func_name in ("tile_load", "zse.tile_load"):
            tensor = self._parse_expr(node.args[0])
            tr = self._parse_expr(node.args[1])
            tc = self._parse_expr(node.args[2])
            ts = self._parse_expr(node.args[3])
            sbuf = self._parse_expr(node.args[4]) if len(node.args) > 4 else None
            br = self._parse_expr(node.args[5]) if len(node.args) > 5 else None
            bc = self._parse_expr(node.args[6]) if len(node.args) > 6 else None
            return IRTileLoad(tensor=tensor, tile_row=tr, tile_col=tc,
                              tile_size=ts, shared_buf=sbuf,
                              bound_row=br, bound_col=bc)

        elif func_name in ("tile_store", "zse.tile_store"):
            sbuf = self._parse_expr(node.args[0])
            tensor = self._parse_expr(node.args[1])
            tr = self._parse_expr(node.args[2])
            tc = self._parse_expr(node.args[3])
            ts = self._parse_expr(node.args[4])
            br = self._parse_expr(node.args[5]) if len(node.args) > 5 else None
            bc = self._parse_expr(node.args[6]) if len(node.args) > 6 else None
            return IRTileStore(shared_buf=sbuf, tensor=tensor, tile_row=tr,
                               tile_col=tc, tile_size=ts,
                               bound_row=br, bound_col=bc)

        # --- WMMA / Tensor Core ---

        elif func_name in ("wmma_load_a", "zse.wmma_load_a"):
            tensor = self._parse_expr(node.args[0])
            row = self._parse_expr(node.args[1])
            col = self._parse_expr(node.args[2])
            stride = self._parse_expr(node.args[3])
            return IRWmmaLoadA(tensor=tensor, row=row, col=col, stride=stride)

        elif func_name in ("wmma_load_b", "zse.wmma_load_b"):
            tensor = self._parse_expr(node.args[0])
            row = self._parse_expr(node.args[1])
            col = self._parse_expr(node.args[2])
            stride = self._parse_expr(node.args[3])
            return IRWmmaLoadB(tensor=tensor, row=row, col=col, stride=stride)

        elif func_name in ("wmma_fill", "zse.wmma_fill"):
            val = self._parse_expr(node.args[0]) if node.args else IRConst(value=0.0)
            return IRWmmaFill(value=val)

        elif func_name in ("wmma_mma", "zse.wmma_mma"):
            a = self._parse_expr(node.args[0])
            b = self._parse_expr(node.args[1])
            c = self._parse_expr(node.args[2])
            return IRWmmaMMA(a_frag=a, b_frag=b, c_frag=c)

        elif func_name in ("wmma_store", "zse.wmma_store"):
            tensor = self._parse_expr(node.args[0])
            row = self._parse_expr(node.args[1])
            col = self._parse_expr(node.args[2])
            stride = self._parse_expr(node.args[3])
            frag = self._parse_expr(node.args[4]) if len(node.args) > 4 else IRVar(name="_frag_c")
            return IRWmmaStore(tensor=tensor, row=row, col=col, stride=stride, frag=frag)

        # --- AMD CDNA MFMA (Tier-4) ---

        elif func_name in ("mfma_f32_16x16x16_f16", "zse.mfma_f32_16x16x16_f16"):
            if len(node.args) != 3:
                raise SyntaxError("mfma_f32_16x16x16_f16 requires 3 args: (a_buf, b_buf, c_buf)")
            a = self._parse_expr(node.args[0])
            b = self._parse_expr(node.args[1])
            c = self._parse_expr(node.args[2])
            return IRMfmaOp(a_buf=a, b_buf=b, c_buf=c, shape="16x16x16_f16")

        elif func_name in ("mfma_f32_32x32x8_f16", "zse.mfma_f32_32x32x8_f16"):
            if len(node.args) != 3:
                raise SyntaxError("mfma_f32_32x32x8_f16 requires 3 args: (a_buf, b_buf, c_buf)")
            a = self._parse_expr(node.args[0])
            b = self._parse_expr(node.args[1])
            c = self._parse_expr(node.args[2])
            return IRMfmaOp(a_buf=a, b_buf=b, c_buf=c, shape="32x32x8_f16")

        # --- INT4 nibble unpack (Tier-2) ---

        elif func_name in ("unpack_int4", "zse.unpack_int4"):
            if len(node.args) != 3:
                raise SyntaxError("unpack_int4 requires 3 args: (packed_u32, out_buf, base_idx)")
            packed = self._parse_expr(node.args[0])
            out_buf = self._parse_expr(node.args[1])
            base_idx = self._parse_expr(node.args[2])
            return IRUnpackInt4(packed=packed, out_buf=out_buf, base_idx=base_idx)

        elif func_name in ("unpack_uint4", "zse.unpack_uint4"):
            if len(node.args) != 3:
                raise SyntaxError("unpack_uint4 requires 3 args: (packed_u32, out_buf, base_idx)")
            packed = self._parse_expr(node.args[0])
            out_buf = self._parse_expr(node.args[1])
            base_idx = self._parse_expr(node.args[2])
            return IRUnpackUint4(packed=packed, out_buf=out_buf, base_idx=base_idx)

        elif func_name.split(".")[-1] in ("exp", "log", "sqrt", "rsqrt", "max_val", "min_val", "fma", "pow", "cos", "sin"):
            math_name = func_name.split(".")[-1]
            args = [self._parse_expr(a) for a in node.args]
            return IRMathFunc(name=math_name, args=args)

        # Python builtins min/max → map to min_val/max_val
        elif func_name == "min":
            args = [self._parse_expr(a) for a in node.args]
            return IRMathFunc(name="min_val", args=args)

        elif func_name == "max":
            args = [self._parse_expr(a) for a in node.args]
            return IRMathFunc(name="max_val", args=args)

        # FP16 conversion
        elif func_name in ("half_to_float", "zse.half_to_float"):
            val = self._parse_expr(node.args[0])
            return IRHalfToFloat(value=val)

        elif func_name in ("float_to_half", "zse.float_to_half"):
            val = self._parse_expr(node.args[0])
            return IRFloatToHalf(value=val)

        # Dynamic shared memory
        elif func_name in ("dynamic_shared_memory", "zse.dynamic_shared_memory"):
            return self._parse_dynamic_shared_memory(node)

        # Local register array — buf = zse.local_array(8, zse.int32)
        elif func_name in ("local_array", "zse.local_array"):
            if len(node.args) < 1:
                raise SyntaxError("local_array requires at least 1 arg: (size, dtype=)")
            size_node = node.args[0]
            if not isinstance(size_node, ast.Constant) or not isinstance(size_node.value, int):
                raise SyntaxError("local_array size must be an integer literal")
            dtype = "float32"
            if len(node.args) >= 2:
                dtype = self._resolve_annotation(node.args[1])
            return IRLocalArrayDecl(name="", size=size_node.value, dtype=dtype)

        # Pointer reinterpret — qp = zse.reinterpret(weights, zse.uint32)
        elif func_name in ("reinterpret", "zse.reinterpret"):
            if len(node.args) != 2:
                raise SyntaxError("reinterpret requires 2 args: (pointer, dtype)")
            operand = self._parse_expr(node.args[0])
            dtype = self._resolve_annotation(node.args[1])
            return IRReinterpret(operand=operand, dtype=dtype)

        elif func_name == "range":
            # Special — handled by for-loop parser
            return node  # Pass through

        elif func_name == "int" or func_name == "float":
            # Type cast
            if len(node.args) != 1:
                raise SyntaxError(f"{func_name}() takes exactly 1 argument")
            operand = self._parse_expr(node.args[0])
            return IRCast(dtype=func_name, operand=operand)

        else:
            raise SyntaxError(f"Unknown function call in kernel: {func_name}")

    def _parse_shared_memory(self, node: ast.Call):
        """Parse shared_memory(shape, dtype) declaration."""
        if len(node.args) < 1:
            raise SyntaxError("shared_memory requires at least shape argument")

        shape_node = node.args[0]
        if isinstance(shape_node, ast.Tuple):
            shape = tuple(
                elt.value if isinstance(elt, ast.Constant) else 0
                for elt in shape_node.elts
            )
        elif isinstance(shape_node, ast.Constant):
            shape = (shape_node.value,)
        else:
            raise SyntaxError("shared_memory shape must be a tuple literal or int")

        dtype = "float32"
        if len(node.args) >= 2:
            dtype = self._resolve_annotation(node.args[1])

        decl = IRSharedMemDecl(shape=shape, dtype=dtype)
        self._shared_mem_decls.append(decl)
        return decl

    def _parse_dynamic_shared_memory(self, node: ast.Call):
        """Parse dynamic_shared_memory(dtype) — size determined at launch."""
        dtype = "float32"
        if node.args:
            dtype = self._resolve_annotation(node.args[0])
        decl = IRDynamicSharedMemDecl(dtype=dtype)
        self._shared_mem_decls.append(decl)
        return decl

    def _parse_if(self, node: ast.If):
        cond = self._parse_expr(node.test)
        then_body = []
        for s in node.body:
            r = self._parse_stmt(s)
            if isinstance(r, list):
                then_body.extend(r)
            elif r is not None:
                then_body.append(r)

        else_body = []
        for s in node.orelse:
            r = self._parse_stmt(s)
            if isinstance(r, list):
                else_body.extend(r)
            elif r is not None:
                else_body.append(r)

        return IRIf(condition=cond, then_body=then_body, else_body=else_body)

    def _parse_for(self, node: ast.For):
        """Parse for loop: for i in range(...)"""
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("For loop target must be a simple variable")

        var_name = node.target.id
        self._var_types[var_name] = "int"

        # Parse range() call
        if isinstance(node.iter, ast.Call):
            call_name = self._get_call_name(node.iter)
            if call_name == "range":
                start, stop, step = self._parse_range_args(node.iter)
            else:
                raise SyntaxError("For loop iterable must be range()")
        else:
            raise SyntaxError("For loop iterable must be range()")

        body = []
        for s in node.body:
            r = self._parse_stmt(s)
            if isinstance(r, list):
                body.extend(r)
            elif r is not None:
                body.append(r)

        return IRFor(var=var_name, start=start, stop=stop, step=step, body=body)

    def _parse_while(self, node: ast.While):
        cond = self._parse_expr(node.test)
        body = []
        for s in node.body:
            r = self._parse_stmt(s)
            if isinstance(r, list):
                body.extend(r)
            elif r is not None:
                body.append(r)
        return IRWhile(condition=cond, body=body)

    def _parse_return(self, node: ast.Return):
        value = self._parse_expr(node.value) if node.value else None
        return IRReturn(value=value)

    def _parse_ann_assign(self, node: ast.AnnAssign):
        """Parse annotated assignment: x: int = expr"""
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("Annotated assignment target must be a variable")
        name = node.target.id
        dtype = self._resolve_annotation(node.annotation)
        self._var_types[name] = dtype
        if node.value:
            value = self._parse_expr(node.value)
            return IRAssign(name=name, value=value, dtype=dtype)
        return None

    # --- Helpers ---

    def _parse_subscript_index(self, node: ast.Subscript):
        """Parse subscript index — handles single index and tuple index."""
        sl = node.slice
        if isinstance(sl, ast.Tuple):
            return [self._parse_expr(elt) for elt in sl.elts]
        else:
            return [self._parse_expr(sl)]

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return ""

    def _get_axis_arg(self, node: ast.Call) -> int:
        if node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
        return 0

    def _parse_range_args(self, node: ast.Call):
        """Parse range(stop), range(start, stop), range(start, stop, step)."""
        args = [self._parse_expr(a) for a in node.args]
        if len(args) == 1:
            return IRConst(value=0), args[0], IRConst(value=1)
        elif len(args) == 2:
            return args[0], args[1], IRConst(value=1)
        elif len(args) == 3:
            return args[0], args[1], args[2]
        else:
            raise SyntaxError("range() takes 1-3 arguments")

    @staticmethod
    def _binop_str(op) -> str:
        mapping = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "/", ast.Mod: "%",
            ast.LShift: "<<", ast.RShift: ">>",
            ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
        }
        return mapping.get(type(op), "+")

    @staticmethod
    def _unaryop_str(op) -> str:
        mapping = {ast.USub: "-", ast.Not: "!", ast.Invert: "~"}
        return mapping.get(type(op), "-")

    @staticmethod
    def _cmpop_str(op) -> str:
        mapping = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }
        return mapping.get(type(op), "==")
