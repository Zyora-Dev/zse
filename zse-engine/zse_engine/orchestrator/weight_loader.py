"""ZSE Weight Loader — Upload .zse model weights to GPU memory.

Uses ZSELoader's mmap for zero-copy transfer: mmap → cuMemcpyHtoD.
No intermediate Python copy, no deserialization overhead.

For a 7B INT4 model (~3.5GB), this takes <2s on PCIe 4.0 (vs vLLM's 5-10s
which includes PyTorch tensor creation + safetensors deserialization).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Callable

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16, int4, int8, uint4

from zse_engine.format.loader import ZSELoader
from zse_engine.format.weight_index import WeightEntry


@dataclass
class GPUWeight:
    """A model weight tensor on GPU."""
    name: str
    shape: Tuple[int, ...]
    dtype: str                    # "int4", "int8", "float16"
    data_ptr: int                 # GPU pointer to packed weight data
    data_nbytes: int
    scales_ptr: int = 0           # GPU pointer to quantization scales
    scales_nbytes: int = 0
    zeros_ptr: int = 0            # GPU pointer to quantization zero-points
    zeros_nbytes: int = 0
    group_size: int = 128
    num_elements: int = 0

    @property
    def total_gpu_bytes(self) -> int:
        return self.data_nbytes + self.scales_nbytes + self.zeros_nbytes


class WeightStore:
    """Collection of all model weights on GPU.

    Provides name-based lookup for the forward pass.
    """

    def __init__(self):
        self._weights: Dict[str, GPUWeight] = {}
        self._total_bytes = 0

    def add(self, weight: GPUWeight):
        self._weights[weight.name] = weight
        self._total_bytes += weight.total_gpu_bytes

    def get(self, name: str) -> GPUWeight:
        """Get a weight by name. Raises KeyError if not found."""
        return self._weights[name]

    def find(self, name: str) -> Optional[GPUWeight]:
        """Get a weight by name, or None."""
        return self._weights.get(name)

    def has(self, name: str) -> bool:
        return name in self._weights

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def num_weights(self) -> int:
        return len(self._weights)

    def __contains__(self, name: str) -> bool:
        return name in self._weights

    def __iter__(self):
        return iter(self._weights.values())

    def summary(self) -> str:
        lines = [f"WeightStore: {self.num_weights} tensors, "
                 f"{self._total_bytes / 1024**2:.1f}MB on GPU"]
        by_dtype = {}
        for w in self._weights.values():
            by_dtype.setdefault(w.dtype, [0, 0])
            by_dtype[w.dtype][0] += 1
            by_dtype[w.dtype][1] += w.total_gpu_bytes
        for dtype, (count, nbytes) in sorted(by_dtype.items()):
            lines.append(f"  {dtype}: {count} tensors, {nbytes / 1024**2:.1f}MB")
        return '\n'.join(lines)

    def destroy(self, gpu_mem):
        """Free all GPU memory."""
        # If bulk-allocated, free the single bulk pointer
        if hasattr(self, '_bulk_ptr') and self._bulk_ptr:
            self._gpu_mem.free_raw(self._bulk_ptr)
            self._bulk_ptr = 0
        else:
            # Legacy per-weight free
            for w in self._weights.values():
                if w.data_ptr:
                    t = Tensor(shape=(1,))
                    t._data_ptr = w.data_ptr
                    t._nbytes = w.data_nbytes
                    gpu_mem.free(t)
                if w.scales_ptr:
                    t = Tensor(shape=(1,))
                    t._data_ptr = w.scales_ptr
                    t._nbytes = w.scales_nbytes
                    gpu_mem.free(t)
                if w.zeros_ptr:
                    t = Tensor(shape=(1,))
                    t._data_ptr = w.zeros_ptr
                    t._nbytes = w.zeros_nbytes
                    gpu_mem.free(t)
        self._weights.clear()
        self._total_bytes = 0


class WeightLoader:
    """Loads .zse weights from mmap to GPU.

    Uses BULK ALLOCATION: single hipMalloc for all weights, then one large
    memcpy. This reduces 771 hipMalloc calls (~8ms each = 6s) to just ONE
    allocation + ONE copy (~0.6s for 18.7GB on MI300X PCIe 5.0).

    Usage:
        loader = ZSELoader("model.zse")
        wl = WeightLoader(loader, gpu_mem)
        store = wl.load_all()
    """

    def __init__(self, loader: ZSELoader, gpu_mem):
        self._loader = loader
        self._gpu_mem = gpu_mem
        self._bulk_ptr = 0  # Base pointer for bulk allocation

    def load_all(
        self,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> WeightStore:
        """Upload all weights using bulk GPU allocation + per-tensor HtoD copies.

        Strategy:
        1. Scan all entries to compute total bytes needed
        2. Single GPU malloc for the entire weight pool
        3. Per-tensor cuMemcpyHtoD from mmap directly
        4. Assign sub-pointers to each GPUWeight
        """
        store = WeightStore()
        entries = list(self._loader.weight_index)
        total = len(entries)

        # Phase 1: Compute layout (offset for each weight piece)
        ALIGN = 256  # GPU memory alignment
        layout = []  # list of (entry, data_offset, scales_offset, zeros_offset)
        current_offset = 0

        for entry in entries:
            data_offset = current_offset
            current_offset += self._align_up(entry.data_nbytes, ALIGN)

            scales_offset = 0
            if entry.scale_nbytes > 0:
                scales_offset = current_offset
                current_offset += self._align_up(entry.scale_nbytes, ALIGN)

            zeros_offset = 0
            if entry.zeros_nbytes > 0:
                zeros_offset = current_offset
                current_offset += self._align_up(entry.zeros_nbytes, ALIGN)

            layout.append((entry, data_offset, scales_offset, zeros_offset))

        total_bytes = current_offset
        if progress_fn:
            progress_fn("bulk_alloc", 0, total)

        # Phase 2: Single bulk GPU allocation
        self._bulk_ptr = self._gpu_mem.malloc_raw(total_bytes)

        # Phase 3: Copy each weight piece directly from mmap to GPU offset
        for idx, (entry, data_off, scales_off, zeros_off) in enumerate(layout):
            if progress_fn and idx % 100 == 0:
                progress_fn(entry.name, idx, total)

            # Copy weight data
            data = self._loader.get_weight_data(entry)
            if data:
                self._gpu_mem.copy_host_to_device_raw(
                    data, self._bulk_ptr + data_off, len(data)
                )

            # Copy scales
            if entry.scale_nbytes > 0:
                scales = self._loader.get_weight_scales(entry)
                self._gpu_mem.copy_host_to_device_raw(
                    scales, self._bulk_ptr + scales_off, len(scales)
                )

            # Copy zeros
            if entry.zeros_nbytes > 0:
                zeros = self._loader.get_weight_zeros(entry)
                self._gpu_mem.copy_host_to_device_raw(
                    zeros, self._bulk_ptr + zeros_off, len(zeros)
                )

            # Create GPUWeight with sub-pointers
            gpu_weight = GPUWeight(
                name=entry.name,
                shape=entry.shape,
                dtype=entry.dtype,
                data_ptr=self._bulk_ptr + data_off,
                data_nbytes=entry.data_nbytes,
                scales_ptr=(self._bulk_ptr + scales_off) if scales_off else 0,
                scales_nbytes=entry.scale_nbytes,
                zeros_ptr=(self._bulk_ptr + zeros_off) if zeros_off else 0,
                zeros_nbytes=entry.zeros_nbytes,
                group_size=entry.group_size,
                num_elements=entry.num_elements,
            )
            store.add(gpu_weight)

        # Store bulk pointer for cleanup
        store._bulk_ptr = self._bulk_ptr
        store._bulk_nbytes = total_bytes
        store._gpu_mem = self._gpu_mem

        return store

    @staticmethod
    def _align_up(n: int, alignment: int) -> int:
        """Round up to next multiple of alignment."""
        return (n + alignment - 1) & ~(alignment - 1)
