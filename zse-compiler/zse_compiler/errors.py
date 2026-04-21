"""ZSE Source Mapping — Map generated code back to original Python source.

When compilation fails, we point the developer to the exact line in their
Python kernel that caused the error.
"""

import inspect
import textwrap
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SourceLocation:
    """Points to a location in the original Python source."""
    file: str
    line: int
    col: int = 0
    text: str = ""

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}"
        if self.text:
            loc += f"\n  {self.text.strip()}"
        return loc


class KernelCompileError(Exception):
    """Raised when kernel compilation fails with source location info."""

    def __init__(self, message: str, location: Optional[SourceLocation] = None,
                 backend: Optional[str] = None, generated_code: Optional[str] = None):
        self.location = location
        self.backend = backend
        self.generated_code = generated_code

        parts = [f"\n[ZSE Kernel Compile Error]"]
        if backend:
            parts.append(f"Backend: {backend}")
        parts.append(f"Error: {message}")
        if location:
            parts.append(f"Source: {location}")
        if generated_code:
            # Show relevant portion of generated code
            lines = generated_code.split('\n')
            if len(lines) <= 20:
                parts.append(f"Generated code:\n{''.join(f'  {i+1:3d}| {l}' + chr(10) for i, l in enumerate(lines))}")
        super().__init__("\n".join(parts))


class KernelValidationError(Exception):
    """Raised when kernel validation fails with clear diagnostics."""

    def __init__(self, func_name: str, errors: List[str], warnings: List[str] = None):
        parts = [f"\n[ZSE Kernel Validation Error] '{func_name}'"]
        if errors:
            parts.append("Errors:")
            for e in errors:
                parts.append(f"  ✗ {e}")
        if warnings:
            parts.append("Warnings:")
            for w in warnings:
                parts.append(f"  ⚠ {w}")
        parts.append("\nKernels can only use ZSE primitives (zse.thread_id, zse.block_id, etc.)")
        parts.append("Run `help(zse.kernel)` for supported constructs.")
        super().__init__("\n".join(parts))


def get_kernel_source_lines(func) -> List[str]:
    """Get the source lines of a kernel function."""
    try:
        source = inspect.getsource(func)
        return textwrap.dedent(source).split('\n')
    except OSError:
        return []


def format_parse_error(func, node_line: int, message: str) -> str:
    """Format a parse error with source context."""
    lines = get_kernel_source_lines(func)
    if not lines or node_line <= 0:
        return f"Parse error in '{func.__name__}': {message}"

    parts = [f"\n[ZSE Parse Error] in '{func.__name__}', line {node_line}:"]
    # Show context: 2 lines before, error line, 2 lines after
    start = max(0, node_line - 3)
    end = min(len(lines), node_line + 2)
    for i in range(start, end):
        marker = " >> " if i == node_line - 1 else "    "
        parts.append(f"  {marker}{i+1:3d}| {lines[i]}")
    parts.append(f"  Error: {message}")
    return "\n".join(parts)
