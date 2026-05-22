"""ZSE Type Inference — Infer types for local variables in kernel IR.

Walks the IR and assigns types to variables based on:
- thread_id, block_id, lane_id, warp_id → int
- Arithmetic with int → int (unless float operand)
- Tensor loads → float (or tensor dtype)
- Math functions → float
- Explicit annotations → as specified
"""

from zse_compiler.ir.nodes import (
    IRNode, IRFunction, IRAssign, IRFor,
    IRConst, IRVar, IRBinOp, IRUnaryOp, IRCast,
    IRThreadIdx, IRBlockIdx, IRBlockDim, IRGridDim, IRGlobalId,
    IRLaneId, IRWarpId,
    IRLoad, IRMathFunc, IRWarpShuffle, IRWarpVote,
    IRWarpReduce, IRBlockReduce,
    IRLoadFloat4, IRLoadHalf2,
    IRLocalArrayDecl, IRReinterpret,
)
from typing import Dict


def infer_types(func: IRFunction) -> Dict[str, str]:
    """Infer types for all local variables in a kernel function.

    Returns: dict of variable_name → type_string ("int", "float", "uint", "float4", "half2")
    """
    types: Dict[str, str] = {}

    # Parameters
    for p in func.params:
        if p.dtype in ("tensor", "Tensor"):
            types[p.name] = "tensor"
        else:
            types[p.name] = p.dtype

    # Walk body
    _infer_body(func.body, types)

    return types


def _infer_body(stmts: list, types: Dict[str, str]):
    for stmt in stmts:
        if isinstance(stmt, IRAssign):
            if stmt.dtype:
                types[stmt.name] = stmt.dtype
            else:
                # Special-case: RHS is a pointer reinterpret — record as ptr:<elem>
                if isinstance(stmt.value, IRReinterpret):
                    types[stmt.name] = f"ptr:{stmt.value.dtype}"
                else:
                    types[stmt.name] = _infer_expr_type(stmt.value, types)
        elif isinstance(stmt, IRLocalArrayDecl):
            # Stack array — record element type as ptr:<elem> so indexed loads infer correctly
            types[stmt.name] = f"ptr:{stmt.dtype}"
        elif isinstance(stmt, IRFor):
            types[stmt.var] = "int"
            _infer_body(stmt.body, types)
        elif hasattr(stmt, 'then_body'):
            _infer_body(stmt.then_body, types)
            if hasattr(stmt, 'else_body'):
                _infer_body(stmt.else_body, types)
        elif hasattr(stmt, 'body') and isinstance(getattr(stmt, 'body'), list):
            _infer_body(stmt.body, types)


def _infer_expr_type(node: IRNode, types: Dict[str, str]) -> str:
    """Infer the type of an expression."""
    if isinstance(node, IRConst):
        if isinstance(node.value, float):
            return "float"
        elif isinstance(node.value, int):
            return "int"
        return "float"

    elif isinstance(node, IRVar):
        return types.get(node.name, "float")

    elif isinstance(node, (IRThreadIdx, IRBlockIdx, IRBlockDim, IRGridDim, IRGlobalId)):
        return "int"

    elif isinstance(node, (IRLaneId, IRWarpId)):
        return "int"

    elif isinstance(node, IRBinOp):
        left_t = _infer_expr_type(node.left, types)
        right_t = _infer_expr_type(node.right, types)
        # Comparison ops always return int (bool)
        if node.op in ("<", "<=", ">", ">=", "==", "!=", "&&", "||"):
            return "int"
        # If either operand is float, result is float
        if left_t == "float" or right_t == "float":
            return "float"
        return "int"

    elif isinstance(node, IRUnaryOp):
        return _infer_expr_type(node.operand, types)

    elif isinstance(node, IRCast):
        return node.dtype

    elif isinstance(node, IRLoad):
        # If we're loading from a known pointer variable (reinterpret or local_array),
        # return the element type so downstream arithmetic is correctly typed.
        if isinstance(node.tensor, IRVar):
            t = types.get(node.tensor.name, "")
            if t.startswith("ptr:"):
                return t[4:]
        return "float"  # Tensor loads are float by default

    elif isinstance(node, IRReinterpret):
        # As an rvalue, a reinterpret evaluates to a pointer of the chosen dtype.
        return f"ptr:{node.dtype}"

    elif isinstance(node, (IRMathFunc, IRWarpReduce, IRBlockReduce)):
        return "float"

    elif isinstance(node, IRWarpShuffle):
        return _infer_expr_type(node.value, types)

    elif isinstance(node, IRWarpVote):
        if node.variant == "ballot":
            return "uint"
        return "int"  # bool-like

    elif isinstance(node, IRLoadFloat4):
        return "float4"

    elif isinstance(node, IRLoadHalf2):
        return "half2"

    return "float"  # Default
