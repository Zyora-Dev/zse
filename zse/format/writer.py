"""
ZSE Format Writer

Converts HuggingFace/SafeTensors models to .zse format.

Features:
- Single-file output with embedded tokenizer
- Optional quantization during conversion
- Layer grouping for efficient streaming
- Memory-efficient conversion (stream large models)
"""

import json
import mmap
import struct
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass
import io

import torch
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
    calculate_tensor_size,
    torch_dtype_to_zse,
)


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    quantization: str = "none"  # "none", "int8", "int4"
    quant_method: str = "absmax"  # "absmax", "gptq", "awq", "hqq"
    group_size: int = 128  # For grouped quantization
    compute_dtype: torch.dtype = torch.float16
    include_tokenizer: bool = True
    target_memory_gb: Optional[float] = None  # Auto-select quant if set


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16 weight to INT4 with absmax scaling.
    
    Args:
        weight: [out_features, in_features] FP16 tensor
        group_size: quantization group size
        
    Returns:
        (packed_int4, scales)
        - packed_int4: [out_features, in_features//2] uint8
        - scales: [out_features, num_groups] float16
    """
    out_features, in_features = weight.shape
    
    # Pad in_features to be divisible by group_size
    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad_size), value=0)
        in_features = weight.shape[1]
    
    # Reshape for group-wise quantization
    num_groups = in_features // group_size
    weight_grouped = weight.view(out_features, num_groups, group_size)
    
    # Compute absmax scales per group
    absmax = weight_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scales = absmax.squeeze(-1) / 7.0  # Scale to [-7, 7] range for INT4
    
    # Quantize to INT4 range [-7, 7]
    weight_scaled = weight_grouped / absmax * 7.0
    weight_int4 = weight_scaled.round().clamp(-7, 7).to(torch.int8)
    
    # Reshape back to [out_features, in_features]
    weight_int4 = weight_int4.view(out_features, in_features)
    
    # Pack two INT4 values into one uint8
    # Shift from [-7, 7] to [0, 15] range for packing
    weight_uint4 = (weight_int4 + 8).to(torch.uint8)
    
    # Pack: even indices in low 4 bits, odd indices in high 4 bits
    packed = torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=weight.device)
    packed = (weight_uint4[:, 0::2] & 0x0F) | ((weight_uint4[:, 1::2] & 0x0F) << 4)
    
    return packed, scales.to(torch.float16)


def quantize_to_int8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16 weight to INT8 with per-channel absmax scaling.
    
    Returns:
        (quantized_int8, scales)
    """
    # Per-channel absmax
    absmax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scales = absmax / 127.0
    
    # Quantize
    weight_int8 = (weight / scales).round().clamp(-127, 127).to(torch.int8)
    
    return weight_int8, scales.squeeze(-1).to(torch.float16)


class ZSEWriter:
    """
    Writer for creating .zse format files.
    
    Usage:
        writer = ZSEWriter(output_path)
        writer.convert_from_hf("meta-llama/Llama-3-8B", quantization="int4")
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[ConversionConfig] = None,
    ):
        """
        Initialize writer.
        
        Args:
            output_path: Path for output .zse file
            config: Conversion configuration
        """
        self.output_path = Path(output_path)
        self.config = config or ConversionConfig()
        
        # Ensure .zse extension
        if self.output_path.suffix != ".zse":
            self.output_path = self.output_path.with_suffix(".zse")
        
        # Header to be built during conversion
        self.header = ZSEHeader()
        
        # Temporary file for building
        self._temp_file: Optional[io.BufferedWriter] = None
        self._current_offset = 0
    
    def convert_from_hf(
        self,
        model_id: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Convert a HuggingFace model to .zse format.
        
        Args:
            model_id: HuggingFace model ID or local path
            revision: Model revision/branch
            trust_remote_code: Trust remote code for custom models
            progress_callback: Optional callback(step, total, message)
            
        Returns:
            Path to created .zse file
        """
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        print(f"Converting {model_id} to .zse format...")
        
        # Load config first to check architecture
        print("  Loading model config...")
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        
        # Build header from config
        self._build_header_from_config(config, model_id, revision)
        
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_bytes = self._serialize_tokenizer(tokenizer)
        
        # Determine quantization if target memory specified
        if self.config.target_memory_gb:
            self._auto_select_quantization(config)
        
        # Load model weights in FP16 - we apply our own quantization
        print(f"  Loading model weights in FP16 (will quantize to: {self.config.quantization})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=self.config.compute_dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Write the .zse file
        print(f"  Writing {self.output_path}...")
        self._write_file(state_dict, tokenizer_bytes)
        
        # Calculate final size
        file_size = self.output_path.stat().st_size
        print(f"  Done! Output: {self.output_path} ({file_size / 1e9:.2f} GB)")
        
        return self.output_path
    
    def convert_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        config: Any,
        tokenizer: Any,
    ) -> Path:
        """
        Convert from a state dict directly.
        
        Args:
            state_dict: Model state dictionary
            config: Model configuration
            tokenizer: Tokenizer instance
            
        Returns:
            Path to created .zse file
        """
        self._build_header_from_config(config, "local", None)
        tokenizer_bytes = self._serialize_tokenizer(tokenizer)
        self._write_file(state_dict, tokenizer_bytes)
        return self.output_path
    
    def _build_header_from_config(
        self,
        config: Any,
        source_model: str,
        revision: Optional[str],
    ) -> None:
        """Build header from HuggingFace config."""
        # Serialize full config to JSON for offline loading
        hf_config_json = ""
        try:
            hf_config_json = config.to_json_string()
        except Exception:
            # Fallback: try to convert config to dict
            try:
                hf_config_json = json.dumps(config.to_dict())
            except Exception:
                pass
        
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
            quant_method=self.config.quant_method if self.config.quantization != "none" else "",
            source_model=source_model,
            source_revision=revision or "",
            hf_config_json=hf_config_json,
        )
    
    def _serialize_tokenizer(self, tokenizer: Any) -> bytes:
        """Serialize tokenizer to bytes."""
        # Save tokenizer to temp dir, then read files
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            
            # Pack all tokenizer files
            tokenizer_data = {}
            for file_path in Path(temp_dir).iterdir():
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        tokenizer_data[file_path.name] = f.read()
            
            # Simple serialization: JSON with base64 for binary
            import base64
            packed = {
                k: base64.b64encode(v).decode('ascii')
                for k, v in tokenizer_data.items()
            }
            return json.dumps(packed).encode('utf-8')
    
    def _auto_select_quantization(self, config: Any) -> None:
        """Auto-select quantization based on target memory."""
        # Estimate base model size
        hidden_size = getattr(config, 'hidden_size', 4096)
        num_layers = getattr(config, 'num_hidden_layers', 32)
        vocab_size = getattr(config, 'vocab_size', 32000)
        
        # Rough parameter count
        params_per_layer = 4 * hidden_size * hidden_size  # Simplified
        total_params = num_layers * params_per_layer + vocab_size * hidden_size * 2
        
        fp16_size_gb = total_params * 2 / 1e9
        int8_size_gb = total_params * 1 / 1e9
        int4_size_gb = total_params * 0.5 / 1e9
        
        target = self.config.target_memory_gb
        
        if int4_size_gb <= target:
            self.config.quantization = "int4"
        elif int8_size_gb <= target:
            self.config.quantization = "int8"
        else:
            self.config.quantization = "none"
        
        print(f"  Auto-selected quantization: {self.config.quantization}")
    
    def _write_file(
        self,
        state_dict: Dict[str, torch.Tensor],
        tokenizer_bytes: bytes,
    ) -> None:
        """Write the complete .zse file."""
        # Group tensors by layer
        layer_tensors, other_tensors = self._group_tensors(state_dict)
        
        # Estimate header size: ~400 bytes per tensor + base
        # INT4 quantization doubles tensor count (weight + scales)
        tensor_count = len(state_dict)
        if self.config.quantization in ("int4", "int8"):
            tensor_count *= 2  # Account for scales tensors
        estimated_header_size = 8192 + tensor_count * 512  # Conservative estimate
        # Round up to next 4KB boundary
        header_size = ((estimated_header_size + 4095) // 4096) * 4096
        
        with open(self.output_path, 'wb') as f:
            # 1. Write placeholder header (will update at end)
            header_placeholder = b'\x00' * header_size
            f.write(header_placeholder)
            
            # 2. Write tokenizer
            self.header.tokenizer_offset = f.tell()
            f.write(struct.pack('<I', len(tokenizer_bytes)))
            f.write(tokenizer_bytes)
            self.header.tokenizer_size = len(tokenizer_bytes) + 4
            
            # 3. Write tensor data and build index
            self.header.tensor_data_offset = f.tell()
            
            # Write non-layer tensors first (embeddings, etc.)
            for name, tensor in tqdm(other_tensors.items(), desc="Writing shared tensors"):
                self._write_tensor(f, name, tensor)
            
            # Write layer tensors with grouping
            for layer_idx in sorted(layer_tensors.keys()):
                group_start = f.tell()
                group_names = []
                
                for name, tensor in layer_tensors[layer_idx].items():
                    self._write_tensor(f, name, tensor)
                    group_names.append(name)
                
                group_end = f.tell()
                
                self.header.layer_groups.append(LayerGroup(
                    layer_idx=layer_idx,
                    tensor_names=group_names,
                    offset=group_start,
                    size=group_end - group_start,
                ))
            
            # 4. Go back and write real header
            header_bytes = encode_header(self.header)
            
            if len(header_bytes) > header_size:
                # Header too big for reserved space
                raise ValueError(
                    f"Header too large ({len(header_bytes)} bytes > {header_size} reserved). "
                    "This shouldn't happen - please report as a bug."
                )
            
            f.seek(0)
            f.write(header_bytes)
            # Pad to original placeholder size
            f.write(b'\x00' * (header_size - len(header_bytes)))
    
    def _group_tensors(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Group tensors by layer index.
        
        Returns:
            (layer_tensors, other_tensors)
            - layer_tensors: {layer_idx: {name: tensor}}
            - other_tensors: {name: tensor} (embeddings, lm_head, etc.)
        """
        layer_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        other_tensors: Dict[str, torch.Tensor] = {}
        
        for name, tensor in state_dict.items():
            # Try to extract layer index
            layer_idx = self._get_layer_index(name)
            
            if layer_idx is not None:
                if layer_idx not in layer_tensors:
                    layer_tensors[layer_idx] = {}
                layer_tensors[layer_idx][name] = tensor
            else:
                other_tensors[name] = tensor
        
        return layer_tensors, other_tensors
    
    def _get_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from tensor name."""
        import re
        
        # Common patterns
        patterns = [
            r'model\.layers\.(\d+)\.',
            r'transformer\.h\.(\d+)\.',
            r'layers\.(\d+)\.',
            r'decoder\.layers\.(\d+)\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        
        return None
    
    def _write_tensor(
        self,
        f: io.BufferedWriter,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Write a single tensor to file, applying quantization if configured."""
        # Get tensor info
        offset = f.tell()
        original_shape = tuple(tensor.shape)
        
        # Convert to contiguous CPU tensor
        tensor = tensor.contiguous().cpu()
        
        # Determine if this tensor should be quantized (only linear layer weights)
        should_quantize = (
            self.config.quantization in ("int4", "int8") and
            len(tensor.shape) == 2 and
            tensor.shape[0] > 64 and tensor.shape[1] > 64 and  # Skip small tensors
            "weight" in name and
            not any(skip in name for skip in ["embed", "norm", "lm_head"])  # Skip embeddings and norms
        )
        
        if should_quantize and self.config.quantization == "int4":
            # Apply INT4 quantization
            packed, scales = quantize_to_int4(tensor, self.config.group_size)
            
            # Write packed weights
            packed_bytes = packed.numpy().tobytes()
            f.write(packed_bytes)
            
            # Record weight tensor info
            self.header.tensors.append(TensorInfo(
                name=name,
                shape=tuple(packed.shape),  # packed shape
                dtype=TensorDType.INT4,
                offset=offset,
                size=len(packed_bytes),
                quant_type=QuantizationType.ABSMAX,
                quant_bits=4,
                group_size=self.config.group_size,
                scale_offset=0,  # Updated below
                zeros_offset=0,
                original_shape=original_shape,  # Store original shape for dequant
            ))
            
            # Write scales
            scales_offset = f.tell()
            scales_bytes = scales.numpy().tobytes()
            f.write(scales_bytes)
            
            # Record scales tensor info
            self.header.tensors.append(TensorInfo(
                name=f"{name}.scales",
                shape=tuple(scales.shape),
                dtype=TensorDType.FLOAT16,
                offset=scales_offset,
                size=len(scales_bytes),
                quant_type=QuantizationType.NONE,
                quant_bits=0,
                group_size=0,
                scale_offset=0,
                zeros_offset=0,
            ))
            
        elif should_quantize and self.config.quantization == "int8":
            # Apply INT8 quantization
            quantized, scales = quantize_to_int8(tensor)
            
            # Write quantized weights
            quant_bytes = quantized.numpy().tobytes()
            f.write(quant_bytes)
            
            self.header.tensors.append(TensorInfo(
                name=name,
                shape=tuple(quantized.shape),
                dtype=TensorDType.INT8,
                offset=offset,
                size=len(quant_bytes),
                quant_type=QuantizationType.ABSMAX,
                quant_bits=8,
                group_size=0,
                scale_offset=0,
                zeros_offset=0,
                original_shape=original_shape,
            ))
            
            # Write scales
            scales_offset = f.tell()
            scales_bytes = scales.numpy().tobytes()
            f.write(scales_bytes)
            
            self.header.tensors.append(TensorInfo(
                name=f"{name}.scales",
                shape=tuple(scales.shape),
                dtype=TensorDType.FLOAT16,
                offset=scales_offset,
                size=len(scales_bytes),
                quant_type=QuantizationType.NONE,
                quant_bits=0,
                group_size=0,
                scale_offset=0,
                zeros_offset=0,
            ))
            
        else:
            # Write as-is (FP16/FP32)
            dtype = torch_dtype_to_zse(tensor.dtype)
            tensor_bytes = tensor.numpy().tobytes()
            f.write(tensor_bytes)
            
            self.header.tensors.append(TensorInfo(
                name=name,
                shape=original_shape,
                dtype=dtype,
                offset=offset,
                size=len(tensor_bytes),
                quant_type=QuantizationType.NONE,
                quant_bits=0,
                group_size=0,
                scale_offset=0,
                zeros_offset=0,
            ))


def convert_model(
    model_id: str,
    output_path: str,
    quantization: str = "none",
    target_memory_gb: Optional[float] = None,
) -> Path:
    """
    Convenience function to convert a model.
    
    Args:
        model_id: HuggingFace model ID or local path
        output_path: Output .zse file path
        quantization: "none", "int8", or "int4"
        target_memory_gb: Auto-select quantization if set
        
    Returns:
        Path to created .zse file
    """
    config = ConversionConfig(
        quantization=quantization,
        target_memory_gb=target_memory_gb,
    )
    
    writer = ZSEWriter(output_path, config)
    return writer.convert_from_hf(model_id)
