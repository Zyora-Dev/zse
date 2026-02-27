"""
ZSE Native Format (.zse)

Custom optimized model format for ZSE.

Features:
- Memory-mappable (instant load, zero-copy)
- Single file (model + tokenizer + config)
- Pre-quantized with per-layer precision
- Streaming-ready (layer offsets for zStream)
- Includes architecture metadata

File Structure:
┌──────────────────────────────┐
│ Magic (8 bytes)              │ "ZSE\\x00" + version
├──────────────────────────────┤
│ Header (JSON)                │ Config, tensor index, layer groups
├──────────────────────────────┤
│ Tokenizer (base64 JSON)      │ Vocab, merges, special tokens
├──────────────────────────────┤
│ Tensor Data                  │ Memory-mapped weights by layer
└──────────────────────────────┘

Usage:
    # Convert HuggingFace model to .zse
    from zse.format import ZSEWriter, convert_model
    convert_model("meta-llama/Llama-3-8B", "llama-8b.zse", quantization="int4")
    
    # Load .zse model
    from zse.format import ZSEReader, load_zse
    
    # Full load
    state_dict, tokenizer, info = load_zse("model.zse")
    
    # Streaming load (for large models)
    with ZSEReader("model.zse") as reader:
        for layer_idx, layer_tensors in reader.iter_layers():
            # Process layer...
            pass
"""

from .spec import (
    ZSEHeader,
    TensorInfo,
    LayerGroup,
    TensorDType,
    QuantizationType,
    ZSE_MAGIC,
    ZSE_VERSION,
    encode_header,
    decode_header,
)

from .reader import (
    ZSEReader,
    ZSEStreamLoader,
    load_zse,
)

from .writer import (
    ZSEWriter,
    ConversionConfig,
    convert_model,
)

# V2 implementations with proper INT4 support
from .writer_v2 import (
    ZSEWriterV2,
    convert_model_v2,
    quantize_tensor_int4_zse,
    quantize_tensor_int8_zse,
)

from .reader_v2 import (
    ZSEReaderV2,
    load_zse_model,
    dequantize_int4_zse,
    dequantize_int8_zse,
    QuantizedLinearZSE,
)

__all__ = [
    # Spec
    "ZSEHeader",
    "TensorInfo",
    "LayerGroup",
    "TensorDType",
    "QuantizationType",
    "ZSE_MAGIC",
    "ZSE_VERSION",
    "encode_header",
    "decode_header",
    # Reader
    "ZSEReader",
    "ZSEStreamLoader",
    "load_zse",
    # Writer
    "ZSEWriter",
    "ConversionConfig",
    "convert_model",
    # V2 - Proper INT4
    "ZSEWriterV2",
    "ZSEReaderV2",
    "convert_model_v2",
    "load_zse_model",
    "quantize_tensor_int4_zse",
    "quantize_tensor_int8_zse",
    "dequantize_int4_zse",
    "dequantize_int8_zse",
    "QuantizedLinearZSE",
]
