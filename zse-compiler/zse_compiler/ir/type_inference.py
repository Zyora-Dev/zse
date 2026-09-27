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
    IRLocalArrayDecl, IRReinterpret, IRSharedMemDecl, IRDynamicSharedMemDecl,
    IRIf, IRWhile, IRReturn, IRHalfToFloat, IRFloatToHalf,
)
from dataclasses import fields
from typing import Dict
from zse_compiler.types.dtypes import DTYPE_MAP


def _normalize_type(dtype: str) -> str:
    aliases = {"float32": "float", "int32": "int", "uint32": "uint", "half": "float16"}
    if dtype in DTYPE_MAP:
        dtype = DTYPE_MAP[dtype].name
    return aliases.get(dtype, dtype)


def _promote_type(dtype: str) -> str:
    dtype = _normalize_type(dtype)
    if dtype in ("int8", "uint8", "int16", "uint16", "int4", "uint4"):
        return "int"
    if dtype in ("float16", "bfloat16"):
        return "float"
    return dtype


def _join_types(left: str, right: str) -> str:
    if left == right:
        return left
    left, right = _promote_type(left), _promote_type(right)
    if left == right:
        return left
    if "float" in (left, right):
        return "float"
    if {left, right} <= {"int", "uint"}:
        return "uint"
    raise ValueError(f"Incompatible kernel variable types: {left} and {right}")


def infer_types(func: IRFunction) -> Dict[str, str]:
    """Infer types for all local variables in a kernel function.

    Returns: dict of variable_name → type_string ("int", "float", "uint", "float4", "half2")
    """
    types: Dict[str, str] = {}

    # Parameters
    for p in func.params:
        if p.dtype in ("tensor", "Tensor"):
            types[p.name] = "ptr:float"
        elif p.dtype.endswith("_tensor"):
            types[p.name] = f"ptr:{_normalize_type(p.dtype[:-7])}"
        elif p.dtype in DTYPE_MAP:
            types[p.name] = f"ptr:{_normalize_type(p.dtype)}"
        else:
            types[p.name] = _normalize_type(p.dtype)

    parameter_types = dict(types)
    while True:
        previous = dict(types)
        _infer_body(func.body, types)
        types.update(parameter_types)
        if types == previous:
            break

    return types


def _infer_body(stmts: list, types: Dict[str, str]):
    for stmt in stmts:
        if isinstance(stmt, IRAssign):
            if stmt.dtype:
                inferred = _normalize_type(stmt.dtype)
            else:
                inferred = _infer_expr_type(stmt.value, types)
            if stmt.name in types and not stmt.dtype:
                inferred = _join_types(types[stmt.name], inferred)
            types[stmt.name] = inferred
        elif isinstance(stmt, (IRLocalArrayDecl, IRSharedMemDecl, IRDynamicSharedMemDecl)):
            types[stmt.name] = f"ptr:{_normalize_type(stmt.dtype)}"
        elif isinstance(stmt, (IRFor, IRWhile)):
            loop_types = dict(types)
            if isinstance(stmt, IRFor):
                loop_types[stmt.var] = "int"
            _infer_body(stmt.body, loop_types)
            for name in types:
                types[name] = _join_types(types[name], loop_types[name])
        elif hasattr(stmt, 'then_body'):
            then_types, else_types = dict(types), dict(types)
            _infer_body(stmt.then_body, then_types)
            _infer_body(stmt.else_body, else_types)
            for name in then_types.keys() | else_types.keys():
                if name in then_types and name in else_types:
                    types[name] = _join_types(then_types[name], else_types[name])
                else:
                    types[name] = then_types[name] if name in then_types else else_types[name]
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
        if node.op in ("<<", ">>"):
            return _promote_type(left_t)
        return _join_types(_promote_type(left_t), _promote_type(right_t))

    elif isinstance(node, IRUnaryOp):
        if node.op == "!":
            return "int"
        return _promote_type(_infer_expr_type(node.operand, types))

    elif isinstance(node, IRCast):
        return _normalize_type(node.dtype)

    elif isinstance(node, IRLoad):
        # If we're loading from a known pointer variable (reinterpret or local_array),
        # return the element type so downstream arithmetic is correctly typed.
        if isinstance(node.tensor, IRVar):
            t = types.get(node.tensor.name, "")
            if t.startswith("ptr:"):
                return _promote_type(t[4:])
        return "float"  # Tensor loads are float by default

    elif isinstance(node, IRReinterpret):
        # As an rvalue, a reinterpret evaluates to a pointer of the chosen dtype.
        return f"ptr:{_normalize_type(node.dtype)}"

    elif isinstance(node, IRIf) and node.is_ternary:
        return _join_types(
            _infer_expr_type(node.then_body[0], types),
            _infer_expr_type(node.else_body[0], types),
        )

    elif isinstance(node, IRHalfToFloat):
        return "float"

    elif isinstance(node, IRFloatToHalf):
        return "float16"

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


def validate_assignments(func: IRFunction) -> None:
    """Reject reads not initialized on every reachable incoming path."""
    _validate_body(func.body, {param.name for param in func.params})


def _validate_reads(node, assigned):
    if isinstance(node, IRVar):
        if node.name not in assigned:
            raise ValueError(f"Kernel variable '{node.name}' may be read before assignment")
    elif isinstance(node, IRNode):
        for field in fields(node):
            _validate_reads(getattr(node, field.name), assigned)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _validate_reads(item, assigned)


def _validate_body(stmts, assigned):
    assigned = set(assigned)
    outer_assigned = set(assigned)
    block_arrays = set()
    for stmt in stmts:
        if isinstance(stmt, IRAssign):
            _validate_reads(stmt.value, assigned)
            assigned.add(stmt.name)
        elif isinstance(stmt, (IRLocalArrayDecl, IRSharedMemDecl, IRDynamicSharedMemDecl)):
            _validate_reads(stmt, assigned)
            assigned.add(stmt.name)
            block_arrays.add(stmt.name)
        elif isinstance(stmt, IRIf) and not stmt.is_ternary:
            _validate_reads(stmt.condition, assigned)
            then_assigned, then_returns = _validate_body(stmt.then_body, assigned)
            else_assigned, else_returns = _validate_body(stmt.else_body, assigned)
            if then_returns and else_returns:
                return assigned, True
            if then_returns:
                assigned = else_assigned
            elif else_returns:
                assigned = then_assigned
            else:
                assigned = then_assigned & else_assigned
        elif isinstance(stmt, IRFor):
            for expression in (stmt.start, stmt.stop, stmt.step):
                _validate_reads(expression, assigned)
            _validate_body(stmt.body, assigned | {stmt.var})
        elif isinstance(stmt, IRWhile):
            _validate_reads(stmt.condition, assigned)
            _validate_body(stmt.body, assigned)
        else:
            _validate_reads(stmt, assigned)
            if isinstance(stmt, IRReturn):
                return assigned, True
    return assigned - (block_arrays - outer_assigned), False
