"""ZSE Writer — Assembles all sections into a .zse file with proper alignment.

Layout:
    [64B header] [section table] [config] [tokenizer?] [page pad] [weight data] [weight index]

Weight index is written AFTER weight data so it can be any size.
Section table and header are rewritten at finalize.
"""

import os
import struct
from typing import Optional

from zse_engine.format.spec import (
    MAGIC, VERSION, HEADER_SIZE, SECTION_ENTRY_SIZE, PAGE_SIZE, TENSOR_ALIGN,
    SectionType, Flags, QuantMethod, align_offset, pad_to_alignment,
)
from zse_engine.format.header import FileHeader, SectionEntry, compute_crc32
from zse_engine.format.config import ModelConfig
from zse_engine.format.weight_index import WeightIndex, WeightEntry
from zse_engine.format.quantize import quantize_tensor, quantize_fp16, compute_packed_size


class ZSEWriter:
    """Writes a .zse model file.

    Usage:
        writer = ZSEWriter("model.zse")
        writer.set_config(config)
        writer.set_tokenizer(tokenizer_data)
        writer.begin_weights()
        for name, weights, shape in tensors:
            writer.add_weight(name, weights, shape)
        writer.finalize()
    """

    def __init__(self, path: str):
        self._path = path
        self._fp = open(path, 'wb')
        self._config: Optional[ModelConfig] = None
        self._tokenizer_bytes: Optional[bytes] = None
        self._weight_index = WeightIndex()
        self._sections = []
        self._weight_data_start = 0
        self._current_weight_offset = 0
        self._finalized = False
        self._has_tokenizer = False
        self._num_sections = 0

    def set_config(self, config: ModelConfig):
        self._config = config

    def set_tokenizer(self, tokenizer_bytes: bytes):
        """Set serialized tokenizer data."""
        self._tokenizer_bytes = tokenizer_bytes

    def begin_weights(self):
        """Write header placeholder + config + tokenizer, then start weight data
        at page-aligned offset. Weight index is written at finalize (after data).
        """
        if self._config is None:
            raise ValueError("Must set_config() before begin_weights()")

        # Determine number of sections
        self._has_tokenizer = self._tokenizer_bytes is not None
        self._num_sections = 3  # CONFIG + WEIGHT_INDEX + WEIGHT_DATA
        if self._has_tokenizer:
            self._num_sections += 1

        # Write placeholder header (rewritten at finalize)
        self._fp.write(b'\x00' * HEADER_SIZE)

        # Write placeholder section table (rewritten at finalize)
        self._fp.write(b'\x00' * (self._num_sections * SECTION_ENTRY_SIZE))

        # --- CONFIG section ---
        config_offset = self._fp.tell()
        config_data = self._config.serialize()
        self._fp.write(config_data)
        self._sections.append(SectionEntry(
            type=SectionType.CONFIG,
            offset=config_offset,
            size=len(config_data),
            crc32=compute_crc32(config_data),
        ))

        # --- TOKENIZER section ---
        if self._has_tokenizer:
            tok_offset = self._fp.tell()
            self._fp.write(self._tokenizer_bytes)
            self._sections.append(SectionEntry(
                type=SectionType.TOKENIZER,
                offset=tok_offset,
                size=len(self._tokenizer_bytes),
                crc32=compute_crc32(self._tokenizer_bytes),
            ))

        # --- Pad to page boundary for WEIGHT_DATA ---
        current = self._fp.tell()
        padding = pad_to_alignment(current, PAGE_SIZE)
        self._fp.write(padding)

        self._weight_data_start = self._fp.tell()
        self._current_weight_offset = 0

    def add_weight(self, name: str, weights: list, shape: tuple, quantize: bool = True):
        """Add a weight tensor. Quantizes to INT4 if quantize=True, else stores as fp16."""
        if quantize:
            packed, scales, zeros = quantize_tensor(
                weights, group_size=self._config.quant.group_size
            )
            dtype = "int4"
            quant_method = self._config.quant.method

            # Write packed data (aligned)
            pad = pad_to_alignment(self._current_weight_offset, TENSOR_ALIGN)
            self._fp.write(pad)
            self._current_weight_offset += len(pad)

            data_offset = self._current_weight_offset
            self._fp.write(packed)
            self._current_weight_offset += len(packed)

            # Write scales
            scale_offset = self._current_weight_offset
            self._fp.write(scales)
            self._current_weight_offset += len(scales)

            # Write zeros
            zeros_offset = self._current_weight_offset
            self._fp.write(zeros)
            self._current_weight_offset += len(zeros)

            entry = WeightEntry(
                name=name, shape=shape, dtype=dtype,
                quant_method=quant_method,
                group_size=self._config.quant.group_size,
                data_offset=data_offset, data_nbytes=len(packed),
                scale_offset=scale_offset, scale_nbytes=len(scales),
                zeros_offset=zeros_offset, zeros_nbytes=len(zeros),
            )
        else:
            fp16_data = quantize_fp16(weights)
            dtype = "float16"

            pad = pad_to_alignment(self._current_weight_offset, TENSOR_ALIGN)
            self._fp.write(pad)
            self._current_weight_offset += len(pad)

            data_offset = self._current_weight_offset
            self._fp.write(fp16_data)
            self._current_weight_offset += len(fp16_data)

            entry = WeightEntry(
                name=name, shape=shape, dtype=dtype,
                quant_method=QuantMethod.NONE,
                data_offset=data_offset, data_nbytes=len(fp16_data),
            )

        self._weight_index.add(entry)

    def add_weight_raw(
        self, name: str, shape: tuple,
        packed_data: bytes, scales_data: bytes, zeros_data: bytes,
        group_size: int = 128,
    ):
        """Add a pre-quantized INT4 weight tensor (fast path).

        Takes already-quantized bytes directly — no Python-side quantization.
        Used by the C-accelerated convert pipeline.
        """
        # Write packed data (aligned)
        pad = pad_to_alignment(self._current_weight_offset, TENSOR_ALIGN)
        self._fp.write(pad)
        self._current_weight_offset += len(pad)

        data_offset = self._current_weight_offset
        self._fp.write(packed_data)
        self._current_weight_offset += len(packed_data)

        scale_offset = self._current_weight_offset
        self._fp.write(scales_data)
        self._current_weight_offset += len(scales_data)

        zeros_offset = self._current_weight_offset
        self._fp.write(zeros_data)
        self._current_weight_offset += len(zeros_data)

        entry = WeightEntry(
            name=name, shape=shape, dtype="int4",
            quant_method=self._config.quant.method,
            group_size=group_size,
            data_offset=data_offset, data_nbytes=len(packed_data),
            scale_offset=scale_offset, scale_nbytes=len(scales_data),
            zeros_offset=zeros_offset, zeros_nbytes=len(zeros_data),
        )
        self._weight_index.add(entry)

    def add_weight_raw_fp16(self, name: str, shape: tuple, fp16_data: bytes):
        """Add a pre-converted fp16 weight tensor (fast path).

        Takes float16 bytes directly — no Python-side conversion.
        """
        pad = pad_to_alignment(self._current_weight_offset, TENSOR_ALIGN)
        self._fp.write(pad)
        self._current_weight_offset += len(pad)

        data_offset = self._current_weight_offset
        self._fp.write(fp16_data)
        self._current_weight_offset += len(fp16_data)

        entry = WeightEntry(
            name=name, shape=shape, dtype="float16",
            quant_method=QuantMethod.NONE,
            data_offset=data_offset, data_nbytes=len(fp16_data),
        )
        self._weight_index.add(entry)

    def add_weight_raw_int8(
        self, name: str, shape: tuple,
        packed_data: bytes, scales_data: bytes,
        group_size: int = 128,
    ):
        """Add a pre-quantized INT8 symmetric weight tensor.

        Takes already-quantized bytes directly.
        """
        pad = pad_to_alignment(self._current_weight_offset, TENSOR_ALIGN)
        self._fp.write(pad)
        self._current_weight_offset += len(pad)

        data_offset = self._current_weight_offset
        self._fp.write(packed_data)
        self._current_weight_offset += len(packed_data)

        scale_offset = self._current_weight_offset
        self._fp.write(scales_data)
        self._current_weight_offset += len(scales_data)

        entry = WeightEntry(
            name=name, shape=shape, dtype="int8",
            quant_method=QuantMethod.INT8_SYM,
            group_size=group_size,
            data_offset=data_offset, data_nbytes=len(packed_data),
            scale_offset=scale_offset, scale_nbytes=len(scales_data),
        )
        self._weight_index.add(entry)

    def finalize(self):
        """Write weight index after weight data, then rewrite header and section table."""
        if self._finalized:
            return

        # --- WEIGHT_DATA section entry ---
        self._sections.append(SectionEntry(
            type=SectionType.WEIGHT_DATA,
            offset=self._weight_data_start,
            size=self._current_weight_offset,
            crc32=0,  # not CRC'd (too large)
        ))

        # --- WEIGHT_INDEX section (written after weight data) ---
        index_data = self._weight_index.serialize()
        index_offset = self._fp.tell()
        self._fp.write(index_data)
        self._sections.append(SectionEntry(
            type=SectionType.WEIGHT_INDEX,
            offset=index_offset,
            size=len(index_data),
            crc32=compute_crc32(index_data),
        ))

        # Total file size
        total_size = self._fp.tell()

        # --- Rewrite header ---
        flags = Flags.MMAP_READY
        if self._has_tokenizer:
            flags |= Flags.HAS_TOKENIZER

        header = FileHeader(
            version=VERSION,
            total_size=total_size,
            flags=flags,
            num_sections=self._num_sections,
        )
        self._fp.seek(0)
        self._fp.write(header.pack())

        # --- Rewrite section table ---
        # Order: CONFIG, [TOKENIZER], WEIGHT_DATA, WEIGHT_INDEX
        for s in self._sections:
            self._fp.write(s.pack())

        self._fp.close()
        self._finalized = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._finalized:
            self.finalize()
