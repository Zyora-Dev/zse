"""ZSE LoRA Format — Load/save LoRA adapter weights.

.zse-lora file format:
    Header (32 bytes):
        - magic: 4 bytes "ZLRA"
        - version: 4 bytes (uint32)
        - rank: 4 bytes (uint32)
        - alpha: 4 bytes (float32)
        - num_layers: 4 bytes (uint32)
        - num_targets: 4 bytes (uint32)
        - total_weight_bytes: 8 bytes (uint64)

    Target module names (null-terminated, padded to 64 bytes each):
        - e.g., "q_proj\0...", "v_proj\0..."

    Per-layer weight data (sequential):
        For each layer, for each target:
            - A matrix: rank * in_features * 2 bytes (fp16)
            - B matrix: out_features * rank * 2 bytes (fp16)

Usage:
    # Save
    save_lora("adapter.zse-lora", adapter)

    # Load
    adapter = load_lora("adapter.zse-lora", gpu_mem)
"""

import struct
from typing import Dict, List, Tuple, Optional

from zse_engine.orchestrator.lora_weights import LoRAWeight, LoRAAdapter


LORA_MAGIC = b'ZLRA'
LORA_VERSION = 1
TARGET_NAME_SIZE = 64  # Padded size for each target module name


def save_lora(path: str, adapter: LoRAAdapter, weight_data: Dict[Tuple[int, str], Tuple[bytes, bytes]]):
    """Save a LoRA adapter to .zse-lora format.

    Args:
        path: Output file path
        adapter: LoRA adapter metadata
        weight_data: (layer_idx, module_name) → (a_bytes, b_bytes)
    """
    targets = adapter.target_modules
    num_targets = len(targets)

    # Calculate total weight bytes
    total_bytes = 0
    for (a_bytes, b_bytes) in weight_data.values():
        total_bytes += len(a_bytes) + len(b_bytes)

    with open(path, 'wb') as f:
        # Header (32 bytes)
        f.write(LORA_MAGIC)
        f.write(struct.pack('<I', LORA_VERSION))
        f.write(struct.pack('<I', adapter.rank))
        f.write(struct.pack('<f', adapter.alpha))
        f.write(struct.pack('<I', adapter.num_layers))
        f.write(struct.pack('<I', num_targets))
        f.write(struct.pack('<Q', total_bytes))

        # Target module names
        for target in targets:
            name_bytes = target.encode('utf-8')[:TARGET_NAME_SIZE - 1]
            padded = name_bytes + b'\x00' * (TARGET_NAME_SIZE - len(name_bytes))
            f.write(padded)

        # Weight data: sequential per layer, per target
        for layer_idx in range(adapter.num_layers):
            for target in targets:
                key = (layer_idx, target)
                if key in weight_data:
                    a_bytes, b_bytes = weight_data[key]
                    f.write(a_bytes)
                    f.write(b_bytes)


def load_lora(
    path: str,
    weight_shapes: Dict[str, Tuple[int, int]],
) -> Tuple[LoRAAdapter, Dict[Tuple[int, str], Tuple[bytes, bytes]]]:
    """Load a LoRA adapter from .zse-lora format.

    Args:
        path: Input file path
        weight_shapes: module_name → (in_features, out_features)

    Returns:
        (adapter, weight_data) where weight_data maps (layer, module) → (a_bytes, b_bytes)
    """
    with open(path, 'rb') as f:
        # Header
        magic = f.read(4)
        if magic != LORA_MAGIC:
            raise ValueError(f"Invalid .zse-lora file: bad magic {magic!r}")

        version = struct.unpack('<I', f.read(4))[0]
        if version != LORA_VERSION:
            raise ValueError(f"Unsupported .zse-lora version: {version}")

        rank = struct.unpack('<I', f.read(4))[0]
        alpha = struct.unpack('<f', f.read(4))[0]
        num_layers = struct.unpack('<I', f.read(4))[0]
        num_targets = struct.unpack('<I', f.read(4))[0]
        total_bytes = struct.unpack('<Q', f.read(8))[0]

        # Target module names
        targets = []
        for _ in range(num_targets):
            raw = f.read(TARGET_NAME_SIZE)
            name = raw.split(b'\x00', 1)[0].decode('utf-8')
            targets.append(name)

        # Create adapter
        adapter = LoRAAdapter(
            adapter_id="",  # Caller sets this
            rank=rank,
            alpha=alpha,
            target_modules=targets,
            num_layers=num_layers,
        )

        # Read weight data
        weight_data = {}
        for layer_idx in range(num_layers):
            for target in targets:
                if target not in weight_shapes:
                    continue
                in_features, out_features = weight_shapes[target]

                a_size = rank * in_features * 2  # fp16
                b_size = out_features * rank * 2
                a_bytes = f.read(a_size)
                b_bytes = f.read(b_size)

                if len(a_bytes) != a_size or len(b_bytes) != b_size:
                    raise ValueError(
                        f"Truncated .zse-lora at layer {layer_idx}, {target}"
                    )

                weight_data[(layer_idx, target)] = (a_bytes, b_bytes)

                # Also create LoRAWeight entry (no GPU pointers yet)
                weight = LoRAWeight(
                    layer_name=f"model.layers.{layer_idx}.{target}",
                    rank=rank,
                    in_features=in_features,
                    out_features=out_features,
                )
                adapter.add_weight(layer_idx, target, weight)

    return adapter, weight_data


def estimate_lora_size(
    rank: int,
    num_layers: int,
    target_modules: List[str],
    hidden_size: int,
    intermediate_size: int = 0,
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
) -> int:
    """Estimate GPU memory needed for a LoRA adapter (in bytes).

    Returns total fp16 bytes for all A and B matrices.
    """
    module_shapes = {
        "q_proj": (hidden_size, num_heads * head_dim),
        "k_proj": (hidden_size, num_kv_heads * head_dim),
        "v_proj": (hidden_size, num_kv_heads * head_dim),
        "o_proj": (num_heads * head_dim, hidden_size),
    }
    if intermediate_size > 0:
        module_shapes.update({
            "gate_proj": (hidden_size, intermediate_size),
            "up_proj": (hidden_size, intermediate_size),
            "down_proj": (intermediate_size, hidden_size),
        })

    total = 0
    for target in target_modules:
        if target in module_shapes:
            in_features, out_features = module_shapes[target]
            # A: [rank, in_features], B: [out_features, rank]
            total += rank * in_features * 2 + out_features * rank * 2

    return total * num_layers
