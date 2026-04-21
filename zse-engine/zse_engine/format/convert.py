"""ZSE Convert — safetensors to .zse conversion pipeline.

Pure Python safetensors reader (no dependencies). Reads the safetensors
header to get tensor metadata, then reads weight data directly.

When the C-accelerated quantizer is available (fast_quant.py), conversion
is ~600x faster — raw bytes go straight from safetensors to INT4 without
ever creating Python float objects.

Usage:
    from zse_engine.format.convert import convert_hf_to_zse
    convert_hf_to_zse("/path/to/hf_model", "output.zse")
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Optional, Callable

from zse_engine.format.config import ModelConfig
from zse_engine.format.writer import ZSEWriter
from zse_engine.format.tokenizer import BPETokenizer
from zse_engine.format.arch.base import get_adapter, detect_architecture
from zse_engine.format.spec import QuantMethod


# --------------------------------------------------------------------------- #
# Pure Python safetensors reader
# --------------------------------------------------------------------------- #

# safetensors dtype string -> (byte_size, struct_format)
_SAFETENSOR_DTYPES = {
    "F32": (4, '<f'),
    "F16": (2, '<e'),
    "BF16": (2, '<H'),   # Read as uint16, convert manually
    "I32": (4, '<i'),
    "I64": (8, '<q'),
    "U8": (1, 'B'),
    "I8": (1, 'b'),
    "BOOL": (1, '?'),
    "F64": (8, '<d'),
}


def _bf16_to_f32(raw: int) -> float:
    """Convert a BF16 value (as uint16) to float32."""
    f32_bytes = struct.pack('<I', raw << 16)
    return struct.unpack('<f', f32_bytes)[0]


def read_safetensors_metadata(path: str) -> Tuple[dict, int]:
    """Read safetensors header metadata.

    Returns: (metadata_dict, header_size)
    The metadata_dict maps tensor_name -> {dtype, shape, data_offsets}
    """
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')

    metadata = json.loads(header_json)
    return metadata, 8 + header_size


def read_safetensors_tensor_raw(
    path: str,
    tensor_info: dict,
    data_offset_base: int,
) -> Tuple[bytes, str, Tuple[int, ...]]:
    """Read raw tensor bytes from safetensors (no conversion).

    Returns: (raw_bytes, dtype_str, shape)
    """
    dtype_str = tensor_info["dtype"]
    shape = tuple(tensor_info["shape"])
    start, end = tensor_info["data_offsets"]

    with open(path, 'rb') as f:
        f.seek(data_offset_base + start)
        raw = f.read(end - start)

    return raw, dtype_str, shape


def read_safetensors_tensor(
    path: str,
    tensor_info: dict,
    data_offset_base: int,
) -> Tuple[List[float], Tuple[int, ...]]:
    """Read a single tensor from safetensors as List[float] (slow path).

    Used when C accelerator is not available.
    """
    raw, dtype_str, shape = read_safetensors_tensor_raw(path, tensor_info, data_offset_base)

    dtype_info = _SAFETENSOR_DTYPES.get(dtype_str)
    if dtype_info is None:
        raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")

    elem_size, fmt = dtype_info
    num_elements = len(raw) // elem_size

    # Batch unpack where possible
    if dtype_str == "BF16":
        u16_vals = struct.unpack(f'<{num_elements}H', raw)
        packed_u32 = struct.pack(f'<{num_elements}I', *(v << 16 for v in u16_vals))
        weights = list(struct.unpack(f'<{num_elements}f', packed_u32))
    elif dtype_str == "F16":
        weights = list(struct.unpack(f'<{num_elements}e', raw))
    elif dtype_str == "F32":
        weights = list(struct.unpack(f'<{num_elements}f', raw))
    else:
        weights = [float(struct.unpack_from(fmt, raw, i * elem_size)[0])
                    for i in range(num_elements)]

    return weights, shape


# --------------------------------------------------------------------------- #
# Conversion pipeline
# --------------------------------------------------------------------------- #

def find_safetensor_files(model_dir: str) -> List[str]:
    """Find all .safetensors files in a model directory."""
    files = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            files.append(os.path.join(model_dir, fname))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return files


def build_tensor_map(
    safetensor_files: List[str],
) -> Dict[str, Tuple[str, dict, int]]:
    """Build a map of tensor_name -> (file_path, tensor_info, data_offset)."""
    tensor_map = {}
    for path in safetensor_files:
        metadata, data_offset = read_safetensors_metadata(path)
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
            tensor_map[name] = (path, info, data_offset)
    return tensor_map


def convert_hf_to_zse(
    model_dir: str,
    output_path: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    quant_method: int = None,
    quant_bits: int = None,
    group_size: int = None,
    arch_override: Optional[str] = None,
    skip_tokenizer: bool = False,
) -> None:
    """Convert a HuggingFace model directory to .zse format.

    Automatically uses C-accelerated quantizer if available (~600x faster).
    Falls back to pure Python if no C compiler is found.

    Args:
        model_dir: Path to HF model directory (with config.json, *.safetensors)
        output_path: Where to write the .zse file
        progress_callback: Optional fn(tensor_name, current, total) for progress
        quant_method: QuantMethod enum value (default: INT4_ASYM)
        quant_bits: Quantization bits (default: 4)
        group_size: Group size (default: 128)
        arch_override: Override auto-detected architecture
        skip_tokenizer: Skip tokenizer embedding
    """
    if quant_method is None:
        quant_method = QuantMethod.INT4_ASYM
    if quant_bits is None:
        quant_bits = 4
    if group_size is None:
        group_size = 128
    # Check for C accelerator
    try:
        from zse_engine.format import fast_quant
        use_fast = fast_quant.is_available()
    except Exception:
        use_fast = False

    if use_fast:
        print("Using C-accelerated quantizer (fast path)")
    else:
        print("WARNING: C compiler not found, using pure Python quantizer (slow)")

    # 1. Read HF config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    with open(config_path, 'r') as f:
        hf_config = json.load(f)

    # 2. Detect architecture and get adapter
    arch_name = arch_override or detect_architecture(hf_config)
    adapter = get_adapter(arch_name)
    model_config = adapter.config_from_hf(hf_config)

    # Override quant config from CLI args
    model_config.quant.method = quant_method
    model_config.quant.bits = quant_bits
    model_config.quant.group_size = group_size

    warnings = adapter.validate_config(model_config)
    for w in warnings:
        print(f"WARNING: {w}")

    # 3. Find safetensors files and build tensor index
    st_files = find_safetensor_files(model_dir)
    tensor_map = build_tensor_map(st_files)

    # 4. Load tokenizer (optional)
    tokenizer_bytes = None
    if not skip_tokenizer:
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = BPETokenizer.from_hf_dir(model_dir)
            tokenizer_bytes = tokenizer.serialize()

    # 5. Check for resumable conversion
    progress_path = output_path + ".progress"
    resume_from = 0

    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                progress_data = json.load(f)
            # Validate it matches this conversion
            if (progress_data.get("model_dir") == os.path.abspath(model_dir)
                    and progress_data.get("quant_method") == quant_method
                    and progress_data.get("group_size") == group_size):
                resume_from = progress_data.get("completed_tensors", 0)
                truncate_pos = progress_data.get("file_position", 0)
                print(f"Resuming from tensor {resume_from} (truncating to {truncate_pos:,} bytes)")
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted progress file, start fresh

    # 6. Write .zse file
    if resume_from > 0:
        # Resume: reopen the partial file and seek to saved position
        writer = ZSEWriter.__new__(ZSEWriter)
        writer._path = output_path
        writer._fp = open(output_path, 'r+b')
        writer._fp.truncate(truncate_pos)
        writer._fp.seek(truncate_pos)
        writer._config = model_config
        writer._tokenizer_bytes = tokenizer_bytes
        writer._weight_index = __import__('zse_engine.format.weight_index',
                                           fromlist=['WeightIndex']).WeightIndex()
        writer._sections = []
        writer._finalized = False
        writer._has_tokenizer = tokenizer_bytes is not None
        writer._num_sections = 3 + (1 if tokenizer_bytes else 0)
        writer._weight_data_start = progress_data["weight_data_start"]
        writer._current_weight_offset = progress_data["weight_offset"]
        # Reload previously written weight entries
        for entry_d in progress_data.get("weight_entries", []):
            from zse_engine.format.weight_index import WeightEntry
            writer._weight_index.add(WeightEntry(**entry_d))
        # Reload section entries
        from zse_engine.format.header import SectionEntry
        for sec_d in progress_data.get("sections", []):
            writer._sections.append(SectionEntry(**sec_d))
    else:
        writer = ZSEWriter(output_path)
        writer.set_config(model_config)
        if tokenizer_bytes:
            writer.set_tokenizer(tokenizer_bytes)
        writer.begin_weights()

    hf_names = adapter.tensor_load_order(list(tensor_map.keys()))
    total = len(hf_names)

    try:
        for idx, hf_name in enumerate(hf_names):
            if idx < resume_from:
                continue  # Skip already-converted tensors

            zse_name = adapter.map_tensor_name(hf_name)
            quantize = adapter.should_quantize(zse_name)

            if progress_callback:
                progress_callback(zse_name, idx + 1, total)

            file_path, tensor_info, data_offset = tensor_map[hf_name]

            if use_fast:
                _convert_tensor_fast(
                    writer, zse_name, file_path, tensor_info,
                    data_offset, quantize, model_config.quant.group_size,
                    quant_method,
                )
            else:
                weights, shape = read_safetensors_tensor(
                    file_path, tensor_info, data_offset,
                )
                if quantize and quant_method == QuantMethod.INT8_SYM:
                    from zse_engine.format.quantize import quantize_tensor_int8
                    packed, scales = quantize_tensor_int8(
                        weights, group_size=model_config.quant.group_size,
                    )
                    writer.add_weight_raw_int8(zse_name, shape, packed, scales,
                                               group_size=model_config.quant.group_size)
                else:
                    writer.add_weight(zse_name, weights, shape, quantize=quantize)

            # Save progress checkpoint
            _save_progress(progress_path, model_dir, quant_method, group_size,
                           idx + 1, writer)

        writer.finalize()

        # Remove progress file on success
        if os.path.exists(progress_path):
            os.unlink(progress_path)

    except BaseException:
        # Don't finalize on crash — the progress file lets us resume
        writer._fp.close()
        writer._finalized = True  # Prevent __exit__ from finalizing
        raise

    print(f"Wrote {output_path} ({os.path.getsize(output_path):,} bytes)")
    print(f"  Architecture: {model_config.arch}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Hidden: {model_config.hidden_size}")
    print(f"  Tensors: {total}")


def _save_progress(
    progress_path: str, model_dir: str, quant_method: int,
    group_size: int, completed: int, writer,
):
    """Save conversion progress for crash recovery."""
    data = {
        "model_dir": os.path.abspath(model_dir),
        "quant_method": quant_method,
        "group_size": group_size,
        "completed_tensors": completed,
        "file_position": writer._fp.tell(),
        "weight_data_start": writer._weight_data_start,
        "weight_offset": writer._current_weight_offset,
        "weight_entries": [
            {"name": e.name, "shape": e.shape, "dtype": e.dtype,
             "quant_method": e.quant_method, "group_size": e.group_size,
             "data_offset": e.data_offset, "data_nbytes": e.data_nbytes,
             "scale_offset": e.scale_offset, "scale_nbytes": e.scale_nbytes,
             "zeros_offset": e.zeros_offset, "zeros_nbytes": e.zeros_nbytes}
            for e in writer._weight_index
        ],
        "sections": [
            {"type": s.type, "offset": s.offset, "size": s.size, "crc32": s.crc32}
            for s in writer._sections
        ],
    }
    # Write atomically
    tmp = progress_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, progress_path)


def _convert_tensor_fast(
    writer: ZSEWriter,
    zse_name: str,
    file_path: str,
    tensor_info: dict,
    data_offset: int,
    quantize: bool,
    group_size: int,
    quant_method: int = None,
) -> None:
    """Fast path: read raw bytes → C quantize → write directly."""
    from zse_engine.format import fast_quant
    if quant_method is None:
        quant_method = QuantMethod.INT4_ASYM

    raw, dtype_str, shape = read_safetensors_tensor_raw(
        file_path, tensor_info, data_offset,
    )

    num_elements = 1
    for d in shape:
        num_elements *= d

    if quantize and dtype_str in ("BF16", "F16", "F32"):
        if quant_method == QuantMethod.INT8_SYM:
            packed, scales = fast_quant.quantize_tensor_int8_fast(
                raw, dtype_str, num_elements, group_size,
            )
            writer.add_weight_raw_int8(
                zse_name, shape, packed, scales,
                group_size=group_size,
            )
        else:
            packed, scales, zeros = fast_quant.quantize_tensor_fast(
                raw, dtype_str, num_elements, group_size,
            )
            # Repack INT4 to tiled format for tensor core coalesced access
            if len(shape) == 2:
                N, K = shape
                packed = fast_quant.repack_int4_tiled(packed, N, K)
            writer.add_weight_raw(
                zse_name, shape, packed, scales, zeros,
                group_size=group_size,
            )
    elif dtype_str in ("BF16", "F16", "F32"):
        fp16_data = fast_quant.convert_to_fp16_fast(raw, dtype_str, num_elements)
        writer.add_weight_raw_fp16(zse_name, shape, fp16_data)
    else:
        weights, shape = read_safetensors_tensor(
            file_path, tensor_info, data_offset,
        )
        writer.add_weight(zse_name, weights, shape, quantize=quantize)
