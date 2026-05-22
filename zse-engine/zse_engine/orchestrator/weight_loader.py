"""ZSE Weight Loader — Upload .zse model weights to GPU memory.

Uses ZSELoader's mmap for zero-copy transfer: mmap → cuMemcpyHtoD.
No intermediate Python copy, no deserialization overhead.

For a 7B INT4 model (~3.5GB), this takes <2s on PCIe 4.0 (vs vLLM's 5-10s
which includes PyTorch tensor creation + safetensors deserialization).
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Callable

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16, int4, int8, uint4

from zse_engine.format.loader import ZSELoader
from zse_engine.format.weight_index import WeightEntry


def _read_exact_into(fd: int, mv: memoryview, offset: int, nbytes: int) -> None:
    """Read exactly `nbytes` bytes from fd at `offset` into `mv`.

    Uses os.preadv on Linux for one-copy kernel→pinned reads (no Python
    bytes object materialised). Falls back to os.pread (two copies) on
    platforms / Python builds without preadv.
    """
    read = 0
    _preadv = getattr(os, "preadv", None)
    if _preadv is not None:
        # Reads positional, doesn't move fd cursor. Single kernel→buffer copy.
        while read < nbytes:
            n = _preadv(fd, [mv[read:nbytes]], offset + read)
            if n <= 0:
                raise IOError(f"preadv: unexpected EOF at offset {offset + read}")
            read += n
        return
    # Fallback: os.pread returns a bytes object, copy into mv.
    while read < nbytes:
        chunk = os.pread(fd, nbytes - read, offset + read)
        if not chunk:
            raise IOError(f"pread: unexpected EOF at offset {offset + read}")
        n = len(chunk)
        mv[read:read + n] = chunk
        read += n


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

    Uses STREAMED BULK ALLOCATION:
      1. One cuMemAlloc for all weights (eliminates 771 per-tensor mallocs).
      2. One contiguous read of the WEIGHT_DATA section from the .zse file
         into a small pinned host ring buffer (4 x 64MB by default).
      3. Async cuMemcpyHtoDAsync per chunk on a dedicated stream — pipelines
         storage I/O (NFS / NVMe / S3 FUSE) with PCIe transfer.
      4. Per-tensor GPU pointers are computed as sub-offsets into the bulk
         pool. No GPU-side relayout — the .zse file layout (PAGE_SIZE +
         TENSOR_ALIGN aligned) is preserved 1:1 on the device.

    This works identically on Modal, RunPod, Lambda, bare metal, AWS, GCP —
    any compute provider with a CUDA/HIP driver. Zero dependencies.

    Falls back to per-tensor copy if pinned alloc or streams fail
    (e.g. Metal backend, or constrained host memory).

    Usage:
        loader = ZSELoader("model.zse")
        wl = WeightLoader(loader, gpu_mem)
        store = wl.load_all()
    """

    # Pinned ring buffer config (tunable via env vars for benchmarking).
    _DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024   # 64 MB per pinned slot
    _DEFAULT_RING_SLOTS = 4                   # 4 slots → 256 MB pinned footprint

    def __init__(self, loader: ZSELoader, gpu_mem):
        self._loader = loader
        self._gpu_mem = gpu_mem
        self._bulk_ptr = 0  # Base pointer for bulk allocation

    def load_all(
        self,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> WeightStore:
        """Upload all weights using streamed bulk allocation.

        Returns a WeightStore with sub-pointers into a single GPU pool.
        """
        # Try the fast streamed path; fall back to per-tensor on any failure.
        try:
            return self._load_all_streamed(progress_fn)
        except Exception as e:
            import os as _os
            if _os.environ.get("ZSE_DEBUG_LOAD"):
                import traceback
                traceback.print_exc()
                print(f"[WeightLoader] streamed path failed ({e}); falling back to per-tensor copy")
            return self._load_all_per_tensor(progress_fn)

    # ------------------------------------------------------------------ #
    # Fast path: streamed pinned ring buffer
    # ------------------------------------------------------------------ #
    def _load_all_streamed(
        self,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> WeightStore:
        import os as _os

        store = WeightStore()
        entries = list(self._loader.weight_index)
        total = len(entries)

        # The .zse file layout is already contiguous + aligned. We mirror it
        # verbatim on the GPU: one big allocation = section_size, and each
        # entry's GPU pointer = bulk_ptr + entry.data_offset.
        file_offset, section_size = self._loader.weight_data_section()
        if section_size == 0:
            return store

        # Phase 1: single GPU allocation
        if progress_fn:
            progress_fn("bulk_alloc", 0, total)
        self._bulk_ptr = self._gpu_mem.malloc_raw(section_size)

        # Phase 2: pinned ring buffer + stream
        chunk_bytes = int(_os.environ.get("ZSE_LOAD_CHUNK_MB", "64")) * 1024 * 1024
        ring_slots = int(_os.environ.get("ZSE_LOAD_RING", str(self._DEFAULT_RING_SLOTS)))
        chunk_bytes = min(chunk_bytes, section_size)
        if chunk_bytes <= 0:
            chunk_bytes = self._DEFAULT_CHUNK_BYTES

        pinned_ptrs = [self._gpu_mem.pinned_alloc(chunk_bytes) for _ in range(ring_slots)]
        # ctypes view over each pinned slot for fast os.readinto()
        pinned_views = [
            (ctypes.c_char * chunk_bytes).from_address(p) for p in pinned_ptrs
        ]
        stream = self._gpu_mem.create_stream()

        # Phase 3: pipelined read + async HtoD
        fd = self._loader.file_descriptor
        if fd < 0:
            raise RuntimeError("ZSE loader has no open file descriptor")

        bytes_done = 0
        chunk_idx = 0
        try:
            while bytes_done < section_size:
                this_chunk = min(chunk_bytes, section_size - bytes_done)
                slot = chunk_idx % ring_slots

                # If this slot is in-flight from a previous round, wait for it.
                # We achieve this by syncing the stream once per ring lap.
                if chunk_idx >= ring_slots:
                    self._gpu_mem.synchronize_stream(stream)

                # Read directly from file into pinned buffer (no Python bytes copy).
                # os.pread is thread-safe + doesn't move the fd cursor.
                view = pinned_views[slot]
                # Read into a memoryview slice of the ctypes buffer.
                mv = memoryview(view)[:this_chunk]
                read_pos = file_offset + bytes_done
                _read_exact_into(fd, mv, read_pos, this_chunk)

                # Queue async HtoD copy.
                self._gpu_mem.copy_host_to_device_async_raw_ptr(
                    src_ptr=pinned_ptrs[slot],
                    dst_ptr=self._bulk_ptr + bytes_done,
                    nbytes=this_chunk,
                    stream=stream,
                )

                bytes_done += this_chunk
                chunk_idx += 1

                if progress_fn and chunk_idx % 4 == 0:
                    progress_fn("upload", bytes_done, section_size)

            # Final sync — all queued copies complete.
            self._gpu_mem.synchronize_stream(stream)
        finally:
            self._gpu_mem.destroy_stream(stream)
            for p in pinned_ptrs:
                self._gpu_mem.pinned_free(p)

        # Phase 4: assemble GPUWeight metadata (no GPU work)
        for idx, entry in enumerate(entries):
            data_ptr = self._bulk_ptr + entry.data_offset
            scales_ptr = (self._bulk_ptr + entry.scale_offset) if entry.scale_nbytes > 0 else 0
            zeros_ptr = (self._bulk_ptr + entry.zeros_offset) if entry.zeros_nbytes > 0 else 0
            store.add(GPUWeight(
                name=entry.name,
                shape=entry.shape,
                dtype=entry.dtype,
                data_ptr=data_ptr,
                data_nbytes=entry.data_nbytes,
                scales_ptr=scales_ptr,
                scales_nbytes=entry.scale_nbytes,
                zeros_ptr=zeros_ptr,
                zeros_nbytes=entry.zeros_nbytes,
                group_size=entry.group_size,
                num_elements=entry.num_elements,
            ))

        store._bulk_ptr = self._bulk_ptr
        store._bulk_nbytes = section_size
        store._gpu_mem = self._gpu_mem
        return store

    # ------------------------------------------------------------------ #
    # Fallback path: original per-tensor copy (Metal / unsupported)
    # ------------------------------------------------------------------ #
    def _load_all_per_tensor(
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
