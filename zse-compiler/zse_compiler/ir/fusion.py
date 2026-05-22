"""ZSE Kernel Fusion — Combine multiple element-wise kernel functions into a single kernel.

Without fusion:
    kernel1: read A from global mem → compute → write B to global mem
    kernel2: read B from global mem → compute → write C to global mem
    = 2 kernel launches, 4 global memory trips

With fusion:
    fused:   read A from global mem → compute1 → compute2 → write C to global mem
    = 1 kernel launch, 2 global memory trips

This is critical for LLM inference where chains like:
    matmul → add_bias → silu → multiply
are common and each kernel launch wastes memory bandwidth.

Usage:
    @zse.kernel
    def add_bias(x: zse.Tensor, bias: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        out[idx] = x[idx] + bias[idx]

    @zse.kernel
    def silu(x: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        val = x[idx]
        out[idx] = val * (1.0 / (1.0 + zse.exp(-val)))

    fused = zse.fuse([add_bias, silu], name="add_bias_silu")
    print(fused.source("cuda"))
"""

from typing import List, Optional
from zse_compiler.ir.nodes import (
    IRFunction, IRParam, IRAssign, IRStore, IRLoad, IRVar,
    IRGlobalId, IRNode, IRConst,
)
from zse_compiler.ast_parser.parser import KernelParser
from zse_compiler.ast_parser.validator import validate_kernel


class FusionPass:
    """Fuses multiple element-wise kernels into a single kernel.

    Strategy:
    1. Parse each kernel to IR
    2. Identify the "output" tensor of kernel N and "input" tensor of kernel N+1
    3. Inline kernel N+1's body, replacing input loads with the computed value
    4. Emit single fused kernel
    """

    def fuse(self, kernel_funcs: list, name: Optional[str] = None,
             chain: Optional[List[str]] = None) -> IRFunction:
        """Fuse a list of KernelFunction objects into a single IR function.

        Requirements:
        - Each kernel must be element-wise (indexed by global_id)
        - Output of kernel N must be input of kernel N+1

        Args:
            kernel_funcs: kernels to fuse in order.
            name: optional name for the fused kernel.
            chain: optional list of length len(kernel_funcs)-1 giving the
                tensor-param NAME in kernel_funcs[i+1] that receives the
                output of kernel_funcs[i]. Use this when a kernel has
                multiple input tensors (e.g. residual_add(x, res, out))
                — otherwise the fuser will raise rather than guess.
        """
        if len(kernel_funcs) < 2:
            raise ValueError("Need at least 2 kernels to fuse")

        irs = [kf.ir for kf in kernel_funcs]
        if chain is not None and len(chain) != len(kernel_funcs) - 1:
            raise ValueError(
                f"chain must have length {len(kernel_funcs) - 1} "
                f"(one entry per junction), got {len(chain)}"
            )

        # Start with first kernel's params and body
        fused_params = list(irs[0].params)
        fused_body = []

        # Track the "output variable" — the tensor being written to
        prev_output_var = None
        prev_output_expr = None

        for i, ir in enumerate(irs):
            if i == 0:
                # First kernel: keep all statements, identify output
                for stmt in ir.body:
                    if isinstance(stmt, IRStore):
                        # This is the output — capture the value expression
                        prev_output_var = self._get_tensor_name(stmt.tensor)
                        prev_output_expr = stmt.value
                        # Don't emit the store — it's intermediate
                        # But save the value in a local var for chaining
                        fused_body.append(IRAssign(
                            name=f"_fused_{i}",
                            value=stmt.value,
                        ))
                    else:
                        fused_body.append(stmt)
            else:
                # Subsequent kernels: inline body, replace input loads with prev output
                hint = chain[i - 1] if chain is not None else None
                input_tensor = self._find_input_tensor(ir, prev_output_var, irs[i-1], hint=hint, kernel_name=kernel_funcs[i].name)

                for stmt in ir.body:
                    if isinstance(stmt, IRAssign):
                        # Check if the value loads from the chained input
                        new_value = self._replace_loads(
                            stmt.value, input_tensor, f"_fused_{i-1}"
                        )
                        if isinstance(stmt, IRStore):
                            # Skip assigning idx again if it's the same global_id pattern
                            pass

                        # Skip re-declaring idx if it's the same global_id
                        if self._is_global_id_assign(stmt):
                            continue

                        fused_body.append(IRAssign(
                            name=stmt.name if not self._is_val_assign(stmt, input_tensor) else stmt.name,
                            value=new_value,
                            dtype=stmt.dtype,
                        ))
                    elif isinstance(stmt, IRStore):
                        new_value = self._replace_loads(
                            stmt.value, input_tensor, f"_fused_{i-1}"
                        )
                        if i == len(irs) - 1:
                            # Last kernel: emit the final store
                            fused_body.append(IRStore(
                                tensor=stmt.tensor,
                                index=stmt.index,
                                value=new_value,
                            ))
                        else:
                            # Intermediate: capture as local
                            prev_output_var = self._get_tensor_name(stmt.tensor)
                            prev_output_expr = new_value
                            fused_body.append(IRAssign(
                                name=f"_fused_{i}",
                                value=new_value,
                            ))

                # Add new params (excluding the chained input/output)
                for p in ir.params:
                    if p.name != input_tensor and not self._param_exists(fused_params, p.name):
                        # Don't add intermediate output tensors
                        if p.name != prev_output_var:
                            fused_params.append(p)

        # Build fused function
        fused_name = name or "_".join(kf.name for kf in kernel_funcs)

        return IRFunction(
            name=fused_name,
            params=self._dedupe_params(fused_params),
            body=fused_body,
        )

    def _get_tensor_name(self, node: IRNode) -> str:
        if isinstance(node, IRVar):
            return node.name
        return ""

    def _find_input_tensor(self, ir: IRFunction, prev_output: str, prev_ir: IRFunction,
                           hint: Optional[str] = None, kernel_name: str = "") -> str:
        """Find which parameter of this kernel receives the previous kernel's output.

        Resolution order:
          1. Explicit chain hint from the user (always wins).
          2. If exactly one tensor param is read-only (never appears as an IRStore
             target), that is unambiguously the input.
          3. Otherwise → raise: the user must disambiguate via chain=[...].
        """
        tensor_params = [p for p in ir.params if p.dtype in ("tensor", "Tensor",
                         "half_tensor", "fp16_tensor", "uint8_tensor", "int8_tensor",
                         "int32_tensor")]
        param_names = {p.name for p in tensor_params}

        # 1. Honor explicit hint
        if hint is not None:
            if hint not in param_names:
                raise ValueError(
                    f"chain entry '{hint}' is not a tensor parameter of kernel "
                    f"'{kernel_name}' (have: {sorted(param_names)})"
                )
            return hint

        # 2. Collect write-target tensors
        written = set()
        self._collect_store_targets(ir.body, written)
        read_only = [p.name for p in tensor_params if p.name not in written]

        if len(read_only) == 1:
            return read_only[0]
        if len(read_only) == 0:
            # No read-only tensor — caller must specify
            raise ValueError(
                f"Cannot fuse into kernel '{kernel_name}': no read-only tensor "
                f"parameter found. Pass chain=[...] to disambiguate."
            )
        # Multi-input kernel (e.g. residual_add(x, res, out)): refuse to guess.
        raise ValueError(
            f"Cannot fuse into kernel '{kernel_name}': it has multiple input "
            f"tensors {sorted(read_only)}. Pass chain=['<param_name>', ...] "
            f"to specify which one receives the previous kernel's output."
        )

    def _collect_store_targets(self, stmts, out: set) -> None:
        """Recursively gather tensor names that appear as IRStore.tensor anywhere."""
        for s in stmts:
            if isinstance(s, IRStore):
                name = self._get_tensor_name(s.tensor)
                if name:
                    out.add(name)
            # Recurse into nested bodies (if/for/while)
            for attr in ("body", "then_body", "else_body"):
                sub = getattr(s, attr, None)
                if isinstance(sub, list):
                    self._collect_store_targets(sub, out)

    def _replace_loads(self, node: IRNode, tensor_name: str, replacement_var: str) -> IRNode:
        """Replace loads from tensor_name with a reference to replacement_var."""
        if isinstance(node, IRLoad):
            if isinstance(node.tensor, IRVar) and node.tensor.name == tensor_name:
                return IRVar(name=replacement_var)
        # Recursively handle binary ops etc.
        if hasattr(node, 'left') and hasattr(node, 'right'):
            from copy import copy
            new_node = copy(node)
            new_node.left = self._replace_loads(node.left, tensor_name, replacement_var)
            new_node.right = self._replace_loads(node.right, tensor_name, replacement_var)
            return new_node
        if hasattr(node, 'operand'):
            from copy import copy
            new_node = copy(node)
            new_node.operand = self._replace_loads(node.operand, tensor_name, replacement_var)
            return new_node
        if hasattr(node, 'args') and isinstance(getattr(node, 'args'), list):
            from copy import copy
            new_node = copy(node)
            new_node.args = [self._replace_loads(a, tensor_name, replacement_var) for a in node.args]
            return new_node
        return node

    def _is_global_id_assign(self, stmt) -> bool:
        if isinstance(stmt, IRAssign) and isinstance(stmt.value, IRGlobalId):
            return True
        return False

    def _is_val_assign(self, stmt, input_tensor: str) -> bool:
        if isinstance(stmt, IRAssign) and isinstance(stmt.value, IRLoad):
            return isinstance(stmt.value.tensor, IRVar) and stmt.value.tensor.name == input_tensor
        return False

    def _param_exists(self, params: List[IRParam], name: str) -> bool:
        return any(p.name == name for p in params)

    def _dedupe_params(self, params: List[IRParam]) -> List[IRParam]:
        seen = set()
        result = []
        for p in params:
            if p.name not in seen:
                seen.add(p.name)
                result.append(p)
        return result
