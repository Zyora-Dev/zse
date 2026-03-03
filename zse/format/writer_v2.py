"""
ZSE Format Writer v2 - Proper INT4 Quantization

This module properly converts models to .zse format with real INT4 quantization
using ZSE's native quantization functions (NOT bitsandbytes).

Key features:
- Uses ZSE's own INT4/INT8 quantization
- Saves packed weights + scales
- Direct GPU loading support
- Memory-efficient conversion
"""

import json
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import io

import torch
import torch.nn as nn
from tqdm import tqdm

from .spec import (
    ZSEHeader,
    TensorInfo,
    LayerGroup,
    TensorDType,
    QuantizationType,
    ZSE_MAGIC,
    ZSE_VERSION,
    encode_header,
    torch_dtype_to_zse,
)


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    quantization: str = "none"  # "none", "int8", "int4"
    group_size: int = 128  # For grouped quantization
    compute_dtype: torch.dtype = torch.float16
    include_tokenizer: bool = True
    target_memory_gb: Optional[float] = None


def quantize_tensor_int4_zse(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16/FP32 tensor to INT4 (packed into UINT8).
    
    Returns:
        (packed_weights, scales)
        - packed_weights: [out_features, in_features//2] as UINT8
        - scales: [out_features, num_groups] as FP16
    """
    out_features, in_features = tensor.shape
    
    # Pad if needed
    pad_size = (group_size - in_features % group_size) % group_size
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        in_features = tensor.shape[1]
    
    # Reshape for grouped quantization
    num_groups = in_features // group_size
    tensor_grouped = tensor.view(out_features, num_groups, group_size)
    
    # Compute per-group scales
    group_max = tensor_grouped.abs().amax(dim=2, keepdim=True)
    group_max = torch.clamp(group_max, min=1e-8)
    scales = group_max / 7.0  # INT4 symmetric: [-7, 7]
    
    # Quantize to [-7, 7]
    quantized = torch.round(tensor_grouped / scales).clamp(-7, 7).to(torch.int8)
    quantized = quantized.view(out_features, in_features)
    
    # Pack: shift to [0, 15] and pack two values into one byte
    quantized_shifted = quantized + 8  # Now [1, 15]
    packed = (quantized_shifted[:, 0::2] & 0x0F) | ((quantized_shifted[:, 1::2] & 0x0F) << 4)
    packed = packed.to(torch.uint8)
    
    scales = scales.squeeze(-1).to(torch.float16)  # [out_features, num_groups]
    
    return packed, scales


def quantize_tensor_int8_zse(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16/FP32 tensor to INT8.
    
    Returns:
        (quantized_weights, scales)
    """
    # Per-channel quantization
    tensor_max = tensor.abs().amax(dim=1, keepdim=True)
    tensor_max = torch.clamp(tensor_max, min=1e-8)
    scales = tensor_max / 127.0
    
    quantized = torch.round(tensor / scales).clamp(-127, 127).to(torch.int8)
    scales = scales.squeeze(-1).to(torch.float16)
    
    return quantized, scales


class ZSEWriterV2:
    """
    Writer for creating .zse format files with proper quantization.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[ConversionConfig] = None,
    ):
        self.output_path = Path(output_path)
        self.config = config or ConversionConfig()
        
        if self.output_path.suffix != ".zse":
            self.output_path = self.output_path.with_suffix(".zse")
        
        self.header = ZSEHeader()
        self._scales_data: Dict[str, Tuple[torch.Tensor, int]] = {}  # name -> (scales, offset)
    
    def convert_from_hf(
        self,
        model_id: str,
        trust_remote_code: bool = True,
    ) -> Path:
        """Convert a HuggingFace model to .zse format with proper quantization."""
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        print(f"Converting {model_id} to .zse format...")
        print(f"Quantization: {self.config.quantization}")
        print(f"Group size: {self.config.group_size}")
        
        # Load config
        print("  Loading config...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        
        # Build header
        self._build_header_from_config(config, model_id)
        
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer_bytes = self._serialize_tokenizer(tokenizer)
        
        # Load model in FP16 (we'll quantize ourselves)
        print("  Loading model in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
        
        # Get state dict and quantize
        print(f"  Quantizing to {self.config.quantization}...")
        state_dict = model.state_dict()
        quantized_state_dict, scales_dict = self._quantize_state_dict(state_dict)
        
        # Write the file
        print(f"  Writing {self.output_path}...")
        self._write_file(quantized_state_dict, scales_dict, tokenizer_bytes)
        
        file_size = self.output_path.stat().st_size
        print(f"  Done! Output: {self.output_path} ({file_size / 1e9:.2f} GB)")
        
        return self.output_path
    
    def _quantize_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Quantize all weight tensors in state dict."""
        quantized = {}
        scales = {}
        
        for name, tensor in tqdm(state_dict.items(), desc="Quantizing"):
            # Only quantize 2D weight tensors (linear layers)
            if tensor.ndim == 2 and "weight" in name and tensor.numel() > 1024:
                if self.config.quantization == "int4":
                    q_tensor, q_scales = quantize_tensor_int4_zse(
                        tensor.float(), 
                        self.config.group_size
                    )
                    quantized[name] = q_tensor
                    scales[f"{name}.scales"] = q_scales
                elif self.config.quantization == "int8":
                    q_tensor, q_scales = quantize_tensor_int8_zse(tensor.float())
                    quantized[name] = q_tensor
                    scales[f"{name}.scales"] = q_scales
                else:
                    quantized[name] = tensor.half()
            else:
                # Keep as FP16
                quantized[name] = tensor.half() if tensor.is_floating_point() else tensor
        
        return quantized, scales
    
    def _build_header_from_config(self, config: Any, source_model: str) -> None:
        """Build header from HuggingFace config."""
        self.header = ZSEHeader(
            version=ZSE_VERSION,
            architecture=config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "unknown",
            model_type=getattr(config, 'model_type', "unknown"),
            hidden_size=getattr(config, 'hidden_size', 0),
            intermediate_size=getattr(config, 'intermediate_size', 0),
            num_hidden_layers=getattr(config, 'num_hidden_layers', 0),
            num_attention_heads=getattr(config, 'num_attention_heads', 0),
            num_key_value_heads=getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 0)),
            vocab_size=getattr(config, 'vocab_size', 0),
            max_position_embeddings=getattr(config, 'max_position_embeddings', 0),
            rope_theta=getattr(config, 'rope_theta', 10000.0),
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
            quantization=self.config.quantization,
            source_model=source_model,
        )
    
    def _serialize_tokenizer(self, tokenizer: Any) -> bytes:
        """Serialize tokenizer to bytes."""
        import base64
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            
            tokenizer_data = {}
            for file_path in Path(temp_dir).iterdir():
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        tokenizer_data[file_path.name] = f.read()
            
            packed = {
                k: base64.b64encode(v).decode('ascii')
                for k, v in tokenizer_data.items()
            }
            return json.dumps(packed).encode('utf-8')
    
    def _write_file(
        self,
        state_dict: Dict[str, torch.Tensor],
        scales_dict: Dict[str, torch.Tensor],
        tokenizer_bytes: bytes,
    ) -> None:
        """Write the complete .zse file with quantized tensors."""
        # Merge scales into state dict for writing
        all_tensors = {**state_dict, **scales_dict}
        
        # Group tensors by layer
        layer_tensors, other_tensors = self._group_tensors(all_tensors)
        
        # Estimate header size
        tensor_count = len(all_tensors)
        estimated_header_size = 4096 + tensor_count * 256
        header_size = ((estimated_header_size + 4095) // 4096) * 4096
        
        with open(self.output_path, 'wb') as f:
            # 1. Write placeholder header
            f.write(b'\x00' * header_size)
            
            # 2. Write tokenizer
            self.header.tokenizer_offset = f.tell()
            f.write(struct.pack('<I', len(tokenizer_bytes)))
            f.write(tokenizer_bytes)
            self.header.tokenizer_size = len(tokenizer_bytes) + 4
            
            # 3. Write tensor data
            self.header.tensor_data_offset = f.tell()
            
            # Write non-layer tensors first
            for name, tensor in tqdm(other_tensors.items(), desc="Writing shared"):
                self._write_tensor(f, name, tensor)
            
            # Write layer tensors
            for layer_idx in sorted(layer_tensors.keys()):
                group_start = f.tell()
                group_names = []
                
                for name, tensor in layer_tensors[layer_idx].items():
                    self._write_tensor(f, name, tensor)
                    group_names.append(name)
                
                self.header.layer_groups.append(LayerGroup(
                    layer_idx=layer_idx,
                    tensor_names=group_names,
                    offset=group_start,
                    size=f.tell() - group_start,
                ))
            
            # 4. Write real header
            header_bytes = encode_header(self.header)
            if len(header_bytes) > header_size:
                raise ValueError(f"Header too large: {len(header_bytes)} > {header_size}")
            
            f.seek(0)
            f.write(header_bytes)
            f.write(b'\x00' * (header_size - len(header_bytes)))
    
    def _write_tensor(self, f: io.BufferedWriter, name: str, tensor: torch.Tensor) -> None:
        """Write a tensor with proper quantization metadata."""
        offset = f.tell()
        tensor = tensor.contiguous().cpu()
        
        # Determine dtype and quantization info
        if tensor.dtype == torch.uint8 and ".scales" not in name:
            # This is INT4 packed weights
            dtype = TensorDType.INT4
            quant_type = QuantizationType.ABSMAX
            quant_bits = 4
        elif tensor.dtype == torch.int8 and ".scales" not in name:
            dtype = TensorDType.INT8
            quant_type = QuantizationType.ABSMAX
            quant_bits = 8
        else:
            dtype = torch_dtype_to_zse(tensor.dtype)
            quant_type = QuantizationType.NONE
            quant_bits = 0
        
        # Convert to bytes
        tensor_bytes = tensor.numpy().tobytes()
        f.write(tensor_bytes)
        
        # Record tensor info
        self.header.tensors.append(TensorInfo(
            name=name,
            shape=tuple(tensor.shape),
            dtype=dtype,
            offset=offset,
            size=len(tensor_bytes),
            quant_type=quant_type,
            quant_bits=quant_bits,
            group_size=self.config.group_size if quant_bits > 0 else 0,
        ))
    
    def _group_tensors(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Group tensors by layer index."""
        import re
        
        layer_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        other_tensors: Dict[str, torch.Tensor] = {}
        
        patterns = [
            r'model\.layers\.(\d+)\.',
            r'transformer\.h\.(\d+)\.',
            r'layers\.(\d+)\.',
        ]
        
        for name, tensor in state_dict.items():
            layer_idx = None
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    break
            
            if layer_idx is not None:
                if layer_idx not in layer_tensors:
                    layer_tensors[layer_idx] = {}
                layer_tensors[layer_idx][name] = tensor
            else:
                other_tensors[name] = tensor
        
        return layer_tensors, other_tensors


def convert_model_v2(
    model_id: str,
    output_path: str,
    quantization: str = "int4",
    group_size: int = 128,
) -> Path:
    """Convenience function to convert a model."""
    config = ConversionConfig(
        quantization=quantization,
        group_size=group_size,
    )
    writer = ZSEWriterV2(output_path, config)
    return writer.convert_from_hf(model_id)
