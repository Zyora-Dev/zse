"""ZSE Loader — mmap-based .zse file loader for fast model loading.

Designed for the <3s cold start goal:
- mmap the file (no copy into Python heap)
- Parse header + section table (microseconds)
- Deserialize config + weight index (milliseconds)
- Weight data stays mmap'd — GPU kernels read directly via offsets

For GPU loading, the orchestrator will:
1. loader.open("model.zse")
2. For each weight entry, cuMemcpy from mmap pointer + offset to GPU
"""

import mmap
import os
from typing import Optional, List, Tuple

from zse_engine.format.spec import SectionType, HEADER_SIZE, SECTION_ENTRY_SIZE
from zse_engine.format.header import FileHeader, SectionEntry, read_header_and_sections, compute_crc32
from zse_engine.format.config import ModelConfig
from zse_engine.format.weight_index import WeightIndex, WeightEntry
from zse_engine.format.tokenizer import BPETokenizer
from zse_engine.format.quantize import dequantize_tensor


class ZSELoader:
    """Memory-mapped .zse file loader.

    Usage:
        loader = ZSELoader("model.zse")
        config = loader.config
        for entry in loader.weight_index:
            raw = loader.get_weight_data(entry)
            # copy to GPU...
        loader.close()
    """

    def __init__(self, path: str):
        self._path = path
        self._fd = None
        self._mm: Optional[mmap.mmap] = None
        self._header: Optional[FileHeader] = None
        self._sections: Optional[List[SectionEntry]] = None
        self._config: Optional[ModelConfig] = None
        self._weight_index: Optional[WeightIndex] = None
        self._tokenizer: Optional[BPETokenizer] = None
        self._tokenizer_data: Optional[bytes] = None  # Lazy: deserialize on first access
        self._weight_data_offset: int = 0

        self._open()

    def _open(self):
        """Open and mmap the file, parse header and sections.

        Performance: on network storage (Modal NFS, S3 mounts) the WEIGHT_INDEX
        section sits at the END of the file (after ~10 GB of weight data),
        which makes a cold first-access slow. We parallelize the small-section
        reads via os.pread + threads so the long NFS fetch overlaps with the
        cheap header/tokenizer reads near the start of the file.
        """
        file_size = os.path.getsize(self._path)
        self._fd = open(self._path, 'rb')

        if file_size == 0:
            raise ValueError(f"Empty .zse file: {self._path}")

        # mmap for downstream weight-data access via pointer arithmetic.
        # Do NOT fadvise the whole 10 GB file here — that can block on NFS
        # and we don't need the weight pages yet (WeightLoader handles them).
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._mm.madvise(mmap.MADV_RANDOM)  # we only touch a few small regions
        except (AttributeError, OSError):
            pass

        # Parse header + section table (tiny, at start of file)
        self._header, self._sections = read_header_and_sections(self._mm)

        if self._header.total_size != file_size:
            raise ValueError(
                f"File size mismatch: header says {self._header.total_size}, "
                f"actual is {file_size}"
            )

        # Read small sections in PARALLEL via os.pread.
        # On NFS this fans out the round-trips so the worst section (weight
        # index, at end of file) doesn't serialize behind the others.
        import threading

        small_sections = [s for s in self._sections if s.type != SectionType.WEIGHT_DATA]
        results = [None] * len(small_sections)
        fd_no = self._fd.fileno()

        # Pre-fadvise just the small section ranges (cheap, non-blocking).
        if hasattr(os, "posix_fadvise"):
            willneed = getattr(os, "POSIX_FADV_WILLNEED", 3)
            for s in small_sections:
                try:
                    os.posix_fadvise(fd_no, s.offset, s.size, willneed)
                except OSError:
                    pass

        def _read_section(idx, section):
            # os.pread is thread-safe on POSIX and bypasses Python's
            # file-position lock — true parallel NFS reads.
            data = os.pread(fd_no, section.size, section.offset)
            results[idx] = data

        threads = [
            threading.Thread(target=_read_section, args=(i, s), daemon=True)
            for i, s in enumerate(small_sections)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Locate weight-data offset (no read needed, just the offset)
        for section in self._sections:
            if section.type == SectionType.WEIGHT_DATA:
                self._weight_data_offset = section.offset
                break

        # Parse the small sections. Skip CRC on the large weight-index section
        # (same trust model as WEIGHT_DATA — local file, integrity already
        # bounded by the file-size check above).
        SKIP_CRC_TYPES = {SectionType.WEIGHT_INDEX, SectionType.WEIGHT_DATA}

        for idx, section in enumerate(small_sections):
            data = results[idx]
            if data is None or len(data) != section.size:
                raise ValueError(
                    f"Short read on section {section.type.name}: "
                    f"got {0 if data is None else len(data)}, expected {section.size}"
                )

            if section.type not in SKIP_CRC_TYPES:
                expected_crc = compute_crc32(data)
                if section.crc32 != expected_crc:
                    raise ValueError(
                        f"CRC mismatch in section {section.type.name}: "
                        f"expected {expected_crc:#x}, got {section.crc32:#x}"
                    )

            if section.type == SectionType.CONFIG:
                self._config = ModelConfig.deserialize(data)
            elif section.type == SectionType.TOKENIZER:
                self._tokenizer_data = data  # already a bytes object
            elif section.type == SectionType.WEIGHT_INDEX:
                self._weight_index = WeightIndex.deserialize(data)

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise ValueError("No config section found in .zse file")
        return self._config

    @property
    def weight_index(self) -> WeightIndex:
        if self._weight_index is None:
            raise ValueError("No weight index section found in .zse file")
        return self._weight_index

    @property
    def tokenizer(self) -> Optional[BPETokenizer]:
        if self._tokenizer is None and self._tokenizer_data is not None:
            self._tokenizer = BPETokenizer.deserialize(self._tokenizer_data)
            self._tokenizer_data = None  # Free raw data
        return self._tokenizer

    @property
    def header(self) -> FileHeader:
        return self._header

    def get_weight_data(self, entry: WeightEntry) -> bytes:
        """Get raw packed weight data for a tensor.

        Returns the INT4 packed bytes (or fp16 bytes for unquantized tensors).
        The offset is relative to the weight data section start.
        """
        abs_offset = self._weight_data_offset + entry.data_offset
        return self._mm[abs_offset:abs_offset + entry.data_nbytes]

    def get_weight_scales(self, entry: WeightEntry) -> bytes:
        """Get quantization scales for a tensor."""
        if entry.scale_nbytes == 0:
            return b''
        abs_offset = self._weight_data_offset + entry.scale_offset
        return self._mm[abs_offset:abs_offset + entry.scale_nbytes]

    def get_weight_zeros(self, entry: WeightEntry) -> bytes:
        """Get quantization zero-points for a tensor."""
        if entry.zeros_nbytes == 0:
            return b''
        abs_offset = self._weight_data_offset + entry.zeros_offset
        return self._mm[abs_offset:abs_offset + entry.zeros_nbytes]

    def get_weight_as_float(self, entry: WeightEntry) -> List[float]:
        """Dequantize a weight tensor to float32 list.

        For verification/debugging only — at inference time the GPU
        kernel dequantizes on-device.
        """
        packed = self.get_weight_data(entry)

        if entry.dtype == "float16":
            import struct as _struct
            n = entry.num_elements
            return [_struct.unpack_from('<e', packed, i * 2)[0] for i in range(n)]
        elif entry.dtype == "int4":
            scales = self.get_weight_scales(entry)
            zeros = self.get_weight_zeros(entry)
            return dequantize_tensor(
                packed, scales, zeros, entry.shape, entry.group_size
            )
        elif entry.dtype == "int8":
            from zse_engine.format.quantize import dequantize_tensor_int8
            scales = self.get_weight_scales(entry)
            return dequantize_tensor_int8(
                packed, scales, entry.shape, entry.group_size
            )
        else:
            raise ValueError(f"Unknown dtype: {entry.dtype}")

    def get_mmap_pointer_and_offset(self, entry: WeightEntry) -> Tuple[mmap.mmap, int]:
        """Get the mmap object and absolute offset for direct GPU transfer.

        The orchestrator can use this to pass the pointer directly to
        cuMemcpyHtoD without any Python-side copy.
        """
        abs_offset = self._weight_data_offset + entry.data_offset
        return self._mm, abs_offset

    def weight_data_section(self) -> Tuple[int, int]:
        """Return (absolute_file_offset, section_size_bytes) of WEIGHT_DATA.

        All `entry.data_offset` / `scale_offset` / `zeros_offset` values are
        relative to the start of this section. Used by bulk weight loaders
        that issue one contiguous read of the section into pinned host memory.
        """
        # Find the WEIGHT_DATA section size from the section table
        for s in self._sections or []:
            if s.type == SectionType.WEIGHT_DATA:
                return self._weight_data_offset, s.size
        raise ValueError("No WEIGHT_DATA section found in .zse file")

    @property
    def file_descriptor(self) -> int:
        """Return raw OS file descriptor for pread()-based streaming reads."""
        return self._fd.fileno() if self._fd else -1

    def summary(self) -> str:
        """Human-readable summary of the loaded model."""
        lines = [f"ZSE Model: {self._path}"]
        lines.append(f"  Version: {self._header.version}")
        lines.append(f"  Size: {self._header.total_size:,} bytes")
        if self._config:
            c = self._config
            lines.append(f"  Arch: {c.arch}")
            lines.append(f"  Layers: {c.num_layers}, Heads: {c.num_heads}, "
                         f"KV Heads: {c.num_kv_heads}")
            lines.append(f"  Hidden: {c.hidden_size}, Intermediate: {c.intermediate_size}")
            lines.append(f"  Vocab: {c.vocab_size}, Max Seq: {c.max_seq_len}")
            lines.append(f"  Quant: {c.quant.bits}-bit, group_size={c.quant.group_size}")
        if self._weight_index:
            lines.append(f"  Tensors: {len(self._weight_index)}")
            q_count = sum(1 for e in self._weight_index if e.dtype == "int4")
            fp_count = sum(1 for e in self._weight_index if e.dtype == "float16")
            lines.append(f"    INT4: {q_count}, FP16: {fp_count}")
        if self._tokenizer:
            lines.append(f"  Tokenizer: {self._tokenizer.vocab_size} tokens")
        return '\n'.join(lines)

    def close(self):
        """Close the memory-mapped file."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            self._fd.close()
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
