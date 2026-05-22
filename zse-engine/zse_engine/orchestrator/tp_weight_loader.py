"""ZSE TP Weight Loader — Load sharded weights for tensor parallelism.

Each GPU rank loads only its shard of each weight matrix:
- Column parallel (QKV, Gate, Up, LM Head): slice rows [rank*N/tp : (rank+1)*N/tp]
- Row parallel (O, Down): slice columns [rank*K/tp : (rank+1)*K/tp]
- Replicated (norms, embedding): full copy

For INT4 quantized weights, column-parallel slicing is straightforward (rows are
independent). Row-parallel slicing requires splitting the packed K dimension and
the associated scales/zeros along their group dimension.
"""

from typing import Optional, Callable, Tuple

from zse_compiler.types.tensor import Tensor

from zse_engine.format.loader import ZSELoader
from zse_engine.format.weight_index import WeightEntry
from zse_engine.orchestrator.weight_loader import WeightStore, GPUWeight, WeightLoader
from zse_engine.orchestrator.tensor_parallel import (
    TensorParallelGroup, COLUMN_PARALLEL, ROW_PARALLEL, REPLICATED,
)


class TPWeightLoader:
    """Load weight shards for tensor parallelism.

    Each rank reads the full weight from mmap, slices its shard, and uploads
    only the shard to GPU. This means each GPU holds 1/tp_size of the column/row
    parallel weights, saving proportional VRAM.

    Args:
        loader: ZSE model file loader (mmap-based)
        gpu_mem: GPU memory allocator (targeting this rank's device)
        tp_group: Tensor parallel group for this rank
    """

    def __init__(self, loader: ZSELoader, gpu_mem, tp_group: TensorParallelGroup):
        self._loader = loader
        self._gpu_mem = gpu_mem
        self._tp = tp_group

    def load_all(
        self,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> WeightStore:
        """Upload all weight shards using bulk GPU allocation.

        Same bulk-alloc strategy as WeightLoader, but each weight is sliced
        to this rank's shard before upload.
        """
        store = WeightStore()
        entries = list(self._loader.weight_index)
        total = len(entries)

        ALIGN = 256

        # Phase 1: Compute layout with shard sizes
        layout = []
        current_offset = 0

        for entry in entries:
            strategy = self._tp.get_split_strategy(entry.name)
            shard_info = self._compute_shard_info(entry, strategy)

            data_offset = current_offset
            current_offset += self._align_up(shard_info["data_nbytes"], ALIGN)

            scales_offset = 0
            if shard_info["scales_nbytes"] > 0:
                scales_offset = current_offset
                current_offset += self._align_up(shard_info["scales_nbytes"], ALIGN)

            zeros_offset = 0
            if shard_info["zeros_nbytes"] > 0:
                zeros_offset = current_offset
                current_offset += self._align_up(shard_info["zeros_nbytes"], ALIGN)

            layout.append((entry, strategy, shard_info, data_offset, scales_offset, zeros_offset))

        total_bytes = current_offset
        if progress_fn:
            progress_fn("tp_bulk_alloc", 0, total)

        # Phase 2: Single bulk GPU allocation
        bulk_ptr = self._gpu_mem.malloc_raw(total_bytes)

        # Phase 3: Copy each weight shard
        for idx, (entry, strategy, shard_info, data_off, scales_off, zeros_off) in enumerate(layout):
            if progress_fn and idx % 100 == 0:
                progress_fn(f"[rank {self._tp.rank}] {entry.name}", idx, total)

            # Get shard data
            data_shard = self._slice_weight_data(entry, strategy, shard_info)
            if data_shard:
                self._gpu_mem.copy_host_to_device_raw(
                    data_shard, bulk_ptr + data_off, len(data_shard)
                )

            # Scales shard
            if shard_info["scales_nbytes"] > 0:
                scales_shard = self._slice_scales(entry, strategy, shard_info)
                self._gpu_mem.copy_host_to_device_raw(
                    scales_shard, bulk_ptr + scales_off, len(scales_shard)
                )

            # Zeros shard
            if shard_info["zeros_nbytes"] > 0:
                zeros_shard = self._slice_zeros(entry, strategy, shard_info)
                self._gpu_mem.copy_host_to_device_raw(
                    zeros_shard, bulk_ptr + zeros_off, len(zeros_shard)
                )

            # Create GPUWeight with shard dimensions
            gpu_weight = GPUWeight(
                name=entry.name,
                shape=shard_info["shape"],
                dtype=entry.dtype,
                data_ptr=bulk_ptr + data_off,
                data_nbytes=shard_info["data_nbytes"],
                scales_ptr=(bulk_ptr + scales_off) if scales_off else 0,
                scales_nbytes=shard_info["scales_nbytes"],
                zeros_ptr=(bulk_ptr + zeros_off) if zeros_off else 0,
                zeros_nbytes=shard_info["zeros_nbytes"],
                group_size=entry.group_size,
                num_elements=shard_info["num_elements"],
            )
            store.add(gpu_weight)

        store._bulk_ptr = bulk_ptr
        store._bulk_nbytes = total_bytes
        store._gpu_mem = self._gpu_mem

        return store

    def _compute_shard_info(self, entry: WeightEntry, strategy: str) -> dict:
        """Compute shard dimensions and byte sizes for this rank."""
        if not entry.shape or len(entry.shape) < 1:
            # Scalar or 1D — replicate
            return {
                "shape": entry.shape,
                "data_nbytes": entry.data_nbytes,
                "scales_nbytes": entry.scale_nbytes,
                "zeros_nbytes": entry.zeros_nbytes,
                "num_elements": entry.num_elements,
                "row_start": 0,
                "row_end": entry.shape[0] if entry.shape else 0,
                "col_start": 0,
                "col_end": entry.shape[1] if len(entry.shape) > 1 else 0,
            }

        N = entry.shape[0]  # output dim (rows)
        K = entry.shape[1] if len(entry.shape) > 1 else 0  # input dim (cols)

        if strategy == COLUMN_PARALLEL and self._tp.tp_size > 1:
            # Split along rows (output dim N)
            shard_N = N // self._tp.tp_size
            row_start = self._tp.rank * shard_N
            row_end = row_start + shard_N

            if entry.dtype == "int4":
                # INT4: packed data is [N, K/2], scales/zeros are [N, K/group_size]
                data_nbytes = shard_N * (K // 2) if K > 0 else shard_N * entry.data_nbytes // N
                num_groups = K // entry.group_size if K > 0 and entry.group_size > 0 else 0
                scales_nbytes = shard_N * num_groups * 2 if num_groups > 0 else 0  # fp16 scales
                zeros_nbytes = shard_N * num_groups * 2 if num_groups > 0 else 0
            elif entry.dtype == "int8":
                data_nbytes = shard_N * K if K > 0 else shard_N * entry.data_nbytes // N
                num_groups = K // entry.group_size if K > 0 and entry.group_size > 0 else 0
                scales_nbytes = shard_N * num_groups * 2 if num_groups > 0 else 0
                zeros_nbytes = 0
            else:  # float16
                data_nbytes = shard_N * K * 2 if K > 0 else shard_N * 2
                scales_nbytes = 0
                zeros_nbytes = 0

            return {
                "shape": (shard_N, K) if K > 0 else (shard_N,),
                "data_nbytes": data_nbytes,
                "scales_nbytes": scales_nbytes,
                "zeros_nbytes": zeros_nbytes,
                "num_elements": shard_N * K if K > 0 else shard_N,
                "row_start": row_start,
                "row_end": row_end,
                "col_start": 0,
                "col_end": K,
            }

        elif strategy == ROW_PARALLEL and self._tp.tp_size > 1:
            # Split along columns (input dim K)
            shard_K = K // self._tp.tp_size
            col_start = self._tp.rank * shard_K
            col_end = col_start + shard_K

            if entry.dtype == "int4":
                # INT4: packed [N, K/2] → shard is [N, shard_K/2]
                data_nbytes = N * (shard_K // 2)
                # Scales/zeros: [N, K/group] → shard is [N, shard_K/group]
                shard_groups = shard_K // entry.group_size if entry.group_size > 0 else 0
                scales_nbytes = N * shard_groups * 2 if shard_groups > 0 else 0
                zeros_nbytes = N * shard_groups * 2 if shard_groups > 0 else 0
            elif entry.dtype == "int8":
                data_nbytes = N * shard_K
                shard_groups = shard_K // entry.group_size if entry.group_size > 0 else 0
                scales_nbytes = N * shard_groups * 2 if shard_groups > 0 else 0
                zeros_nbytes = 0
            else:  # float16
                data_nbytes = N * shard_K * 2
                scales_nbytes = 0
                zeros_nbytes = 0

            return {
                "shape": (N, shard_K),
                "data_nbytes": data_nbytes,
                "scales_nbytes": scales_nbytes,
                "zeros_nbytes": zeros_nbytes,
                "num_elements": N * shard_K,
                "row_start": 0,
                "row_end": N,
                "col_start": col_start,
                "col_end": col_end,
            }

        else:
            # Replicated — full copy
            return {
                "shape": entry.shape,
                "data_nbytes": entry.data_nbytes,
                "scales_nbytes": entry.scale_nbytes,
                "zeros_nbytes": entry.zeros_nbytes,
                "num_elements": entry.num_elements,
                "row_start": 0,
                "row_end": N,
                "col_start": 0,
                "col_end": K,
            }

    def _slice_weight_data(self, entry: WeightEntry, strategy: str, shard: dict) -> bytes:
        """Slice weight data bytes for this rank's shard."""
        full_data = self._loader.get_weight_data(entry)
        if not full_data:
            return b''

        if strategy == REPLICATED or self._tp.tp_size <= 1:
            return full_data

        N = entry.shape[0]
        K = entry.shape[1] if len(entry.shape) > 1 else 0

        if strategy == COLUMN_PARALLEL:
            # Rows are contiguous — slice rows
            row_start, row_end = shard["row_start"], shard["row_end"]
            row_bytes = len(full_data) // N
            return full_data[row_start * row_bytes : row_end * row_bytes]

        elif strategy == ROW_PARALLEL:
            # Columns — need to extract shard_K cols from each row
            col_start, col_end = shard["col_start"], shard["col_end"]

            if entry.dtype == "int4":
                # Packed: each row is K/2 bytes. Cols are byte-aligned when shard_K is even.
                half_K = K // 2
                shard_half_K = (col_end - col_start) // 2
                col_byte_start = col_start // 2
                result = bytearray()
                for row in range(N):
                    row_offset = row * half_K
                    result.extend(full_data[row_offset + col_byte_start : row_offset + col_byte_start + shard_half_K])
                return bytes(result)
            elif entry.dtype == "int8":
                shard_K = col_end - col_start
                result = bytearray()
                for row in range(N):
                    row_offset = row * K
                    result.extend(full_data[row_offset + col_start : row_offset + col_start + shard_K])
                return bytes(result)
            else:  # float16
                shard_K = col_end - col_start
                result = bytearray()
                for row in range(N):
                    row_offset = row * K * 2
                    result.extend(full_data[row_offset + col_start * 2 : row_offset + (col_start + shard_K) * 2])
                return bytes(result)

        return full_data

    def _slice_scales(self, entry: WeightEntry, strategy: str, shard: dict) -> bytes:
        """Slice quantization scales for this rank's shard."""
        full_scales = self._loader.get_weight_scales(entry)
        if not full_scales:
            return b''

        if strategy == REPLICATED or self._tp.tp_size <= 1:
            return full_scales

        N = entry.shape[0]
        K = entry.shape[1] if len(entry.shape) > 1 else 0
        gs = entry.group_size
        num_groups_full = K // gs if gs > 0 else 0

        if strategy == COLUMN_PARALLEL:
            # Scales are [N, num_groups] fp16 — slice rows
            row_start, row_end = shard["row_start"], shard["row_end"]
            row_bytes = num_groups_full * 2  # fp16 per group
            return full_scales[row_start * row_bytes : row_end * row_bytes]

        elif strategy == ROW_PARALLEL:
            # Scales are [N, num_groups] fp16 — slice group columns
            col_start, col_end = shard["col_start"], shard["col_end"]
            group_start = col_start // gs
            group_end = col_end // gs
            shard_groups = group_end - group_start
            result = bytearray()
            for row in range(N):
                row_offset = row * num_groups_full * 2
                result.extend(full_scales[row_offset + group_start * 2 : row_offset + group_end * 2])
            return bytes(result)

        return full_scales

    def _slice_zeros(self, entry: WeightEntry, strategy: str, shard: dict) -> bytes:
        """Slice quantization zeros for this rank's shard."""
        full_zeros = self._loader.get_weight_zeros(entry)
        if not full_zeros:
            return b''

        # Same slicing logic as scales
        if strategy == REPLICATED or self._tp.tp_size <= 1:
            return full_zeros

        N = entry.shape[0]
        K = entry.shape[1] if len(entry.shape) > 1 else 0
        gs = entry.group_size
        num_groups_full = K // gs if gs > 0 else 0

        if strategy == COLUMN_PARALLEL:
            row_start, row_end = shard["row_start"], shard["row_end"]
            row_bytes = num_groups_full * 2
            return full_zeros[row_start * row_bytes : row_end * row_bytes]

        elif strategy == ROW_PARALLEL:
            col_start, col_end = shard["col_start"], shard["col_end"]
            group_start = col_start // gs
            group_end = col_end // gs
            result = bytearray()
            for row in range(N):
                row_offset = row * num_groups_full * 2
                result.extend(full_zeros[row_offset + group_start * 2 : row_offset + group_end * 2])
            return bytes(result)

        return full_zeros

    @staticmethod
    def _align_up(n: int, alignment: int) -> int:
        return (n + alignment - 1) & ~(alignment - 1)
