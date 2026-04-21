"""ZSE Tensor — lightweight GPU tensor with zero PyTorch dependency.

Used both as:
1. Type annotation in @zse.kernel functions (compile time)
2. Actual data container at runtime (wraps raw GPU pointer)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from zse_compiler.types.dtypes import DType, float32


@dataclass
class Tensor:
    """ZSE Tensor — raw GPU memory + shape metadata.

    At compile time: used as type hint for kernel parameters.
    At runtime: wraps a GPU buffer pointer with shape/stride info.
    """
    shape: Tuple[int, ...] = ()
    dtype: DType = field(default_factory=lambda: float32)
    strides: Optional[Tuple[int, ...]] = None
    _data_ptr: int = 0  # Raw GPU memory pointer (set by runtime)
    _device: str = ""   # "cuda:0", "rocm:0", "metal:0"
    _nbytes: int = 0

    def __post_init__(self):
        if self.shape and self.strides is None:
            # Row-major (C-contiguous) strides
            self.strides = self._compute_strides(self.shape)
        if self.shape:
            self._nbytes = self._compute_nbytes()

    def _compute_strides(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))

    def _compute_nbytes(self) -> int:
        numel = 1
        for d in self.shape:
            numel *= d
        if self.dtype.size_bits < 8:
            # Packed types (int4): 2 values per byte
            return (numel * self.dtype.size_bits + 7) // 8
        return numel * self.dtype.size_bytes()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    @property
    def device(self) -> str:
        return self._device

    def is_contiguous(self) -> bool:
        if not self.strides:
            return True
        expected = self._compute_strides(self.shape)
        return self.strides == expected

    def reshape(self, *new_shape: int) -> Tensor:
        """Returns a new Tensor descriptor with reshaped dimensions."""
        # Validate total elements match
        new_numel = 1
        for d in new_shape:
            new_numel *= d
        if new_numel != self.numel:
            raise ValueError(f"Cannot reshape {self.shape} ({self.numel} elements) to {new_shape} ({new_numel} elements)")
        return Tensor(
            shape=new_shape,
            dtype=self.dtype,
            _data_ptr=self._data_ptr,
            _device=self._device,
        )

    def __repr__(self) -> str:
        return f"zse.Tensor(shape={self.shape}, dtype={self.dtype.name}, device='{self._device}')"


def empty(shape: Tuple[int, ...], dtype: DType = float32) -> Tensor:
    """Create a Tensor descriptor (no allocation — runtime allocates on GPU)."""
    return Tensor(shape=shape, dtype=dtype)


def zeros(shape: Tuple[int, ...], dtype: DType = float32) -> Tensor:
    """Create a zero-initialized Tensor descriptor."""
    t = Tensor(shape=shape, dtype=dtype)
    # Runtime will handle zero-fill on GPU
    return t
