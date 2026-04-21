"""ZSE Kernel Validator — validates kernel functions before compilation."""

import ast
import inspect
import textwrap
from typing import List


# Python features NOT allowed in GPU kernels
FORBIDDEN_BUILTINS = {
    "print", "input", "open", "eval", "exec", "compile",
    "globals", "locals", "vars", "dir", "help",
    "import", "__import__", "isinstance", "issubclass",
    "list", "dict", "set", "tuple", "str", "bytes",
    "map", "filter", "zip", "enumerate", "sorted", "reversed",
    "iter", "next", "len",  # len could be supported later
}

ALLOWED_CALLS = {
    "range", "int", "float",
    # ZSE thread primitives
    "thread_id", "block_id", "block_dim", "grid_dim", "global_id",
    "lane_id", "warp_id",
    # Memory + sync
    "shared_memory", "dynamic_shared_memory", "syncthreads",
    # Atomics
    "atomic_add", "atomic_max", "atomic_min", "atomic_cas",
    # Math
    "exp", "log", "sqrt", "rsqrt", "max_val", "min_val", "fma",
    "pow", "cos", "sin", "min", "max",
    # FP16 conversion
    "half_to_float", "float_to_half",
    # Warp primitives
    "warp_shuffle_down", "warp_shuffle_up", "warp_shuffle_xor", "warp_shuffle",
    "warp_ballot", "warp_all", "warp_any",
    # Reductions
    "warp_reduce_sum", "warp_reduce_max", "warp_reduce_min",
    "block_reduce_sum", "block_reduce_max", "block_reduce_min",
    # Vectorized memory
    "load_float4", "store_float4", "load_half2", "store_half2",
    # Tiling
    "tile_load", "tile_store",
}

# Also allow all the above with "zse." prefix
ALLOWED_CALLS = ALLOWED_CALLS | {f"zse.{name}" for name in ALLOWED_CALLS if name not in ("range", "int", "float")}


class KernelValidationError(Exception):
    """Raised when a kernel function contains invalid constructs."""
    pass


def validate_kernel(func) -> List[str]:
    """Validate a function is suitable for GPU kernel compilation.

    Returns list of warnings (empty = valid).
    Raises KernelValidationError for fatal issues.
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    errors = []
    warnings = []

    for node in ast.walk(tree):
        # No closures / nested functions
        if isinstance(node, ast.FunctionDef) and node.name != func.__name__:
            errors.append(f"Nested functions not allowed in kernels: '{node.name}'")

        # No classes
        if isinstance(node, ast.ClassDef):
            errors.append(f"Class definitions not allowed in kernels: '{node.name}'")

        # No imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append("Import statements not allowed in kernels")

        # No try/except
        if isinstance(node, (ast.Try, ast.ExceptHandler)):
            errors.append("Try/except not allowed in kernels (no exceptions on GPU)")

        # No yield / async
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            errors.append("Generators not allowed in kernels")
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
            errors.append("Async constructs not allowed in kernels")

        # No with statements
        if isinstance(node, ast.With):
            errors.append("With statements not allowed in kernels")

        # No list/dict/set comprehensions
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            errors.append("Comprehensions not allowed in kernels")

        # Validate function calls
        if isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name in FORBIDDEN_BUILTINS:
                errors.append(f"Built-in '{call_name}' not allowed in kernels")
            elif call_name and call_name not in ALLOWED_CALLS:
                warnings.append(f"Unknown function call '{call_name}' — may not compile")

        # No string operations
        if isinstance(node, ast.JoinedStr):  # f-strings
            errors.append("String formatting not allowed in kernels")

        # No global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed in kernels")

    if errors:
        msg = f"Kernel '{func.__name__}' validation failed:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        raise KernelValidationError(msg)

    return warnings


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
        return node.func.attr
    return ""
