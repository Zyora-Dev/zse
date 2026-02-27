"""
ZSE Format Reader v2 - Proper INT4 Loading

Loads .zse format files with proper INT4 dequantization.

Key features:
- Memory-mapped file access
- Direct GPU loading
- On-the-fly INT4 dequantization
- Fast cold start
"""

import json
import mmap
import struct
import tempfile
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from .spec import (
    ZSEHeader,
    TensorInfo,
    TensorDType,
    QuantizationType,
    decode_header,
    zse_dtype_to_torch,
)


def dequantize_int4_zse(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT4 packed tensor back to FP16.
    
    Args:
        packed: [out_features, in_features//2] as UINT8
        scales: [out_features, num_groups] as FP16
        group_size: quantization group size
        dtype: output dtype
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2
    
    # Unpack INT4 values
    low = (packed & 0x0F).to(torch.int8) - 8  # Back to [-7, 7]
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    
    # Interleave back to original order
    unpacked = torch.zeros(out_features, in_features, dtype=torch.int8, device=packed.device)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    
    # Reshape for group-wise dequantization
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.view(out_features, num_groups, group_size)
    scales_expanded = scales.unsqueeze(-1)  # [out_features, num_groups, 1]
    
    # Dequantize
    dequantized = (unpacked_grouped.float() * scales_expanded).view(out_features, in_features)
    
    return dequantized.to(dtype)


def dequantize_int8_zse(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 tensor back to FP16."""
    # Per-channel scale
    if scales.ndim == 1:
        scales = scales.unsqueeze(1)
    return (quantized.float() * scales).to(dtype)


class ZSEReaderV2:
    """
    Reader for .zse format with proper INT4 support.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        device: str = "cuda",
    ):
        self.path = Path(path)
        self.device = device
        
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        # Open and mmap file
        self._file = open(self.path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data = self._mmap
        
        # Parse header
        self.header, self._header_end = decode_header(self._data)
        
        # Build tensor lookup
        self._tensor_map = {t.name: t for t in self.header.tensors}
    
    def close(self):
        if self._mmap:
            self._mmap.close()
        self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def quantization(self) -> str:
        return self.header.quantization
    
    @property
    def num_layers(self) -> int:
        return self.header.num_hidden_layers
    
    def get_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "path": str(self.path),
            "size_gb": self.path.stat().st_size / 1e9,
            "architecture": self.header.architecture,
            "num_layers": self.header.num_hidden_layers,
            "hidden_size": self.header.hidden_size,
            "quantization": self.header.quantization,
            "num_tensors": len(self.header.tensors),
        }
    
    def load_tokenizer(self):
        """Load embedded tokenizer."""
        from transformers import AutoTokenizer
        
        offset = self.header.tokenizer_offset
        size_bytes = self._data[offset:offset + 4]
        tokenizer_size = struct.unpack('<I', size_bytes)[0]
        
        tokenizer_json = self._data[offset + 4:offset + 4 + tokenizer_size].decode('utf-8')
        tokenizer_data = json.loads(tokenizer_json)
        
        # Extract to temp dir
        temp_dir = tempfile.mkdtemp(prefix="zse_tok_")
        for filename, content_b64 in tokenizer_data.items():
            content = base64.b64decode(content_b64)
            with open(Path(temp_dir) / filename, 'wb') as f:
                f.write(content)
        
        return AutoTokenizer.from_pretrained(temp_dir)
    
    def _read_raw_tensor(self, info: TensorInfo) -> torch.Tensor:
        """Read raw tensor bytes."""
        raw = self._data[info.offset:info.offset + info.size]
        
        # Map dtype
        np_dtype_map = {
            TensorDType.FLOAT32: np.float32,
            TensorDType.FLOAT16: np.float16,
            TensorDType.INT8: np.int8,
            TensorDType.INT4: np.uint8,  # INT4 is packed as uint8
            TensorDType.UINT8: np.uint8,
        }
        np_dtype = np_dtype_map.get(info.dtype, np.float16)
        
        array = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
        return torch.from_numpy(array.copy())
    
    def load_tensor_dequantized(
        self,
        name: str,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Load a tensor, dequantizing if needed."""
        device = device or self.device
        info = self._tensor_map.get(name)
        
        if info is None:
            raise KeyError(f"Tensor not found: {name}")
        
        tensor = self._read_raw_tensor(info)
        
        # Handle quantized tensors
        if info.quant_type != QuantizationType.NONE:
            scales_name = f"{name}.scales"
            if scales_name in self._tensor_map:
                scales_info = self._tensor_map[scales_name]
                scales = self._read_raw_tensor(scales_info).to(device)
                tensor = tensor.to(device)
                
                if info.dtype == TensorDType.INT4:
                    tensor = dequantize_int4_zse(tensor, scales, info.group_size)
                    # Truncate to original shape if padded during quantization
                    if info.original_shape:
                        target_shape = info.original_shape
                        if tensor.shape[1] > target_shape[1]:
                            tensor = tensor[:, :target_shape[1]]
                elif info.dtype == TensorDType.INT8:
                    tensor = dequantize_int8_zse(tensor, scales)
                    # INT8 doesn't change shape, but just in case
                    if info.original_shape:
                        target_shape = info.original_shape
                        if tensor.shape != target_shape:
                            tensor = tensor.view(target_shape)
            else:
                tensor = tensor.to(device).half()
        else:
            tensor = tensor.to(device)
            if tensor.dtype != torch.float16 and tensor.is_floating_point():
                tensor = tensor.half()
        
        return tensor
    
    def load_state_dict_dequantized(
        self,
        device: Optional[str] = None,
        progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Load full state dict with dequantization."""
        device = device or self.device
        state_dict = {}
        
        # Get weight tensor names (skip scales)
        weight_names = [t.name for t in self.header.tensors if not t.name.endswith(".scales")]
        
        if progress:
            weight_names = tqdm(weight_names, desc="Loading tensors")
        
        for name in weight_names:
            tensor = self.load_tensor_dequantized(name, device)
            state_dict[name] = tensor
        
        return state_dict


class QuantizedLinearZSE(nn.Module):
    """
    Quantized linear layer that stores INT4 weights.
    
    Uses bitsandbytes CUDA kernels for fast inference.
    On first forward, converts ZSE INT4 format to bnb format for fast matmul.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Packed INT4 weights: [out_features, in_features//2]
        self.register_buffer('weight_packed', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        
        # Scales: [out_features, num_groups]
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer('weight_scales', torch.zeros(out_features, num_groups, dtype=torch.float16))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer('bias', None)
        
        # bnb format (set on first forward or by convert_to_bnb)
        self._bnb_weight = None
        self._bnb_quant_state = None
        
        # Cached FP16 weight (optional, for cache_weights mode)
        self._cached_weight = None
    
    def convert_to_bnb(self):
        """Convert ZSE INT4 format to bnb format for fast inference."""
        if self._bnb_weight is not None:
            return  # Already converted
        
        import bitsandbytes as bnb
        from bitsandbytes.functional import quantize_4bit
        
        # Dequantize our format to FP16
        weight_fp16 = dequantize_int4_zse(
            self.weight_packed,
            self.weight_scales,
            self.group_size,
            dtype=torch.float16,
        )
        if weight_fp16.shape[1] > self.in_features:
            weight_fp16 = weight_fp16[:, :self.in_features]
        
        # Re-quantize with bnb (uses nf4 by default - better quality)
        self._bnb_weight, self._bnb_quant_state = quantize_4bit(
            weight_fp16,
            compress_statistics=False,
            quant_type='nf4',
        )
        
        # Free original packed weights to save VRAM
        self.weight_packed = None
        self.weight_scales = None
    
    def cache_weights(self):
        """Pre-dequantize weights to FP16 for maximum speed. Uses ~4x VRAM."""
        if self._cached_weight is None:
            self._cached_weight = dequantize_int4_zse(
                self.weight_packed,
                self.weight_scales,
                self.group_size,
                dtype=torch.float16,
            )
            if self._cached_weight.shape[1] > self.in_features:
                self._cached_weight = self._cached_weight[:, :self.in_features]
    
    def clear_cache(self):
        """Clear cached weights to free VRAM."""
        self._cached_weight = None
        self._bnb_weight = None
        self._bnb_quant_state = None
        torch.cuda.empty_cache()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import bitsandbytes as bnb
        
        # Ensure float16
        if x.dtype != torch.float16:
            x = x.half()
        
        # Fast path 1: Cached FP16 weights (fastest, highest VRAM)
        if self._cached_weight is not None:
            return nn.functional.linear(x, self._cached_weight, self.bias)
        
        # Fast path 2: bnb format (fast CUDA kernel, low VRAM)
        if self._bnb_weight is not None:
            out = bnb.matmul_4bit(x, self._bnb_weight.t(), quant_state=self._bnb_quant_state)
            if self.bias is not None:
                out = out + self.bias
            return out
        
        # First call: convert to bnb format
        self.convert_to_bnb()
        out = bnb.matmul_4bit(x, self._bnb_weight.t(), quant_state=self._bnb_quant_state)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_model_to_bnb(model: nn.Module) -> int:
    """
    Pre-convert all INT4 layers to bnb format.
    
    This enables fast CUDA inference while keeping VRAM low.
    Call this after loading if you want fast inference without cache_weights.
    
    Returns number of layers converted.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinearZSE):
            module.convert_to_bnb()
            count += 1
    return count


def cache_model_weights(model: nn.Module) -> int:
    """
    Pre-dequantize all INT4 weights to FP16 for maximum inference speed.
    
    Trade-off: Uses ~4x VRAM but gives maximum throughput.
    
    Returns number of layers cached.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinearZSE):
            module.cache_weights()
            count += 1
    return count


def clear_model_cache(model: nn.Module) -> int:
    """Clear cached weights to free VRAM."""
    count = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinearZSE):
            module.clear_cache()
            count += 1
    torch.cuda.empty_cache()
    return count


def _auto_decide_cache(file_size_gb: float) -> bool:
    """
    Auto-decide whether to cache weights based on available VRAM.
    
    Logic: Cache if free VRAM > 2.5x file size (after model skeleton loaded)
    This ensures enough headroom for KV cache and inference.
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Get free VRAM
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_vram_gb = free_vram / (1024**3)
        
        # Estimate memory needed for caching
        # INT4 packed -> FP16 dequantized = ~4x expansion
        cache_memory_gb = file_size_gb * 3  # Conservative estimate
        
        # Need cache memory + 2GB headroom for KV cache
        required_gb = cache_memory_gb + 2.0
        
        should_cache = free_vram_gb >= required_gb
        return should_cache
    except Exception:
        return True  # Default to caching if detection fails


def load_zse_model(
    zse_path: Union[str, Path],
    device: str = "cuda",
    cache_weights: Union[bool, str] = "auto",
) -> Tuple[nn.Module, Any, Dict]:
    """
    Load a .zse file into a usable model.
    
    Args:
        zse_path: Path to .zse file
        device: Device to load to ("cuda" or "cpu")
        cache_weights: Weight caching mode:
            - True: Always cache (fast inference, ~15GB for 7B)
            - False: Never cache (memory efficient, ~6GB for 7B)
            - "auto": Detect available VRAM and decide automatically
    
    Returns:
        (model, tokenizer, info)
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    
    print(f"Loading {zse_path}...")
    
    with ZSEReaderV2(zse_path, device) as reader:
        info = reader.get_info()
        print(f"  Architecture: {info['architecture']}")
        print(f"  Quantization: {info['quantization']}")
        print(f"  File size: {info['size_gb']:.2f} GB")
        
        # Load tokenizer from embedded data
        print("  Loading tokenizer...")
        tokenizer = reader.load_tokenizer()
        
        # Load config from embedded JSON (no network call)
        print("  Loading config...")
        if reader.header.hf_config_json:
            # Use embedded config - no network call
            config_dict = json.loads(reader.header.hf_config_json)
            # Remove model_type from dict if present (it's passed separately)
            model_type = config_dict.pop('model_type', reader.header.model_type)
            config = AutoConfig.for_model(model_type, **config_dict)
        else:
            # Fallback to network (old .zse files)
            print("    Warning: No embedded config, fetching from HuggingFace...")
            config = AutoConfig.from_pretrained(
                reader.header.source_model,
                trust_remote_code=True,
            )
        
        # Check if INT4 quantized
        is_int4 = info['quantization'] == 'int4'
        
        if is_int4:
            # Load with INT4 weights kept in packed format
            print("  Loading INT4 weights (direct GPU)...")
            model = _load_int4_model_direct_gpu(reader, config, device)
            
            # Decide whether to cache weights
            if cache_weights == "auto":
                should_cache = _auto_decide_cache(info['size_gb'])
                if should_cache:
                    print("  Auto-detected sufficient VRAM - caching for speed...")
                else:
                    print("  Auto-detected limited VRAM - memory-efficient mode...")
            else:
                should_cache = cache_weights
            
            # Cache weights for faster inference
            if should_cache:
                print("  Caching dequantized weights (for fast inference)...")
                num_cached = cache_model_weights(model)
                vram_gb = torch.cuda.memory_allocated() / 1e9
                print(f"    Cached {num_cached} layers, VRAM: {vram_gb:.2f} GB")
            else:
                # Convert to bnb format for fast CUDA kernels at low VRAM
                print("  Converting to bnb format (fast CUDA kernels)...")
                num_converted = convert_model_to_bnb(model)
                vram_gb = torch.cuda.memory_allocated() / 1e9
                print(f"    Converted {num_converted} layers, VRAM: {vram_gb:.2f} GB")
        else:
            # Load with dequantization (FP16)
            print("  Loading weights...")
            state_dict = reader.load_state_dict_dequantized(device)
            
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
            model.load_state_dict(state_dict, assign=True, strict=False)
        
        return model, tokenizer, info


def _load_int4_model_direct_gpu(
    reader: 'ZSEReaderV2',
    config: Any,
    device: str,
) -> nn.Module:
    """
    Load INT4 model directly to GPU.
    
    Strategy:
    1. Create model on meta device (no memory)
    2. Replace Linear with QuantizedLinearZSE 
    3. Load INT4 packed weights directly to GPU
    4. Load other weights (embeddings, norms) directly to GPU
    5. Initialize computed buffers on GPU
    """
    from transformers import AutoModelForCausalLM
    
    # Build mapping of which tensors are INT4 quantized
    int4_weights = {}
    for t in reader.header.tensors:
        if t.dtype == TensorDType.INT4 and not t.name.endswith('.scales'):
            scales_name = f"{t.name}.scales"
            scales_info = reader._tensor_map.get(scales_name)
            if scales_info:
                int4_weights[t.name] = (t, scales_info)
    
    # Track what we load
    loaded_names = set()
    for wname in int4_weights:
        loaded_names.add(wname)
        loaded_names.add(f"{wname}.scales")
        bias_name = wname.replace('.weight', '.bias')
        loaded_names.add(bias_name)
    
    # Create model skeleton on meta device (zero memory)
    print("    Creating model skeleton...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    # Replace Linear layers with QuantizedLinearZSE and load INT4 weights to GPU
    print("    Loading INT4 layers to GPU...")
    replaced = _replace_linear_with_quantized_v2(model, int4_weights, reader, device)
    print(f"    Replaced {replaced} layers with INT4")
    
    # Load non-quantized weights directly to GPU
    print("    Loading embeddings and norms...")
    non_quant_state_dict = {}
    for t in reader.header.tensors:
        if t.name not in loaded_names:
            tensor = reader.load_tensor_dequantized(t.name, device)
            non_quant_state_dict[t.name] = tensor
    
    if non_quant_state_dict:
        model.load_state_dict(non_quant_state_dict, assign=True, strict=False)
    
    # Initialize computed buffers (like rotary embeddings) on GPU
    _initialize_computed_buffers(model, config, device)
    
    return model


def _initialize_computed_buffers(model: nn.Module, config: Any, device: str):
    """
    Initialize computed buffers that aren't stored in weights.
    E.g., rotary embedding inv_freq
    """
    # Find rotary embedding modules and initialize their buffers
    for name, module in model.named_modules():
        # Check if this is a rotary embedding module
        if hasattr(module, 'inv_freq') and module.inv_freq.device == torch.device('meta'):
            # Compute inv_freq on device
            dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
            base = getattr(config, 'rope_theta', 10000.0)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            # Replace the buffer
            module.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Some models have cos/sin caches
        if hasattr(module, '_cos_cached') and module._cos_cached is not None:
            if module._cos_cached.device == torch.device('meta'):
                module._cos_cached = None
                module._sin_cached = None


def _replace_linear_with_quantized_v2(
    model: nn.Module,
    int4_weights: dict,
    reader: 'ZSEReaderV2',
    device: str,
) -> int:
    """
    Replace Linear layers with QuantizedLinearZSE using module path traversal.
    Returns count of replaced layers.
    """
    replaced = 0
    
    for weight_name, (weight_info, scales_info) in int4_weights.items():
        # weight_name is like "model.layers.0.self_attn.q_proj.weight"
        # We need to get to "model.layers.0.self_attn" and replace "q_proj"
        parts = weight_name.replace('.weight', '').split('.')
        
        # Navigate to parent module
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part, None)
                if parent is None:
                    break
        
        if parent is None:
            continue
        
        layer_name = parts[-1]
        old_layer = getattr(parent, layer_name, None)
        
        if old_layer is None or not isinstance(old_layer, (nn.Linear, type(old_layer))):
            continue
        
        # Get dimensions from original shape or infer from packed weight
        group_size = weight_info.group_size or 128
        if weight_info.original_shape:
            out_features, in_features = weight_info.original_shape
        else:
            # Infer from packed shape: [out_features, in_features//2]
            out_features = weight_info.shape[0]
            in_features = weight_info.shape[1] * 2
        
        # Create quantized layer
        has_bias = hasattr(old_layer, 'bias') and old_layer.bias is not None
        quant_layer = QuantizedLinearZSE(
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bias=has_bias,
        )
        
        # Load packed weights directly (no dequantization)
        packed = reader._read_raw_tensor(weight_info).to(device)
        scales = reader._read_raw_tensor(scales_info).to(device).half()
        
        quant_layer.weight_packed = packed
        quant_layer.weight_scales = scales
        
        # Load bias if present
        bias_name = weight_name.replace('.weight', '.bias')
        if bias_name in reader._tensor_map:
            bias = reader.load_tensor_dequantized(bias_name, device)
            quant_layer.bias = bias
        
        # Replace the layer
        setattr(parent, layer_name, quant_layer)
        replaced += 1
    
    return replaced
