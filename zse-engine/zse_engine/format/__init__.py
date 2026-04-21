"""ZSE Format — .zse model file format for zero-dependency LLM inference."""

from zse_engine.format.spec import MAGIC, VERSION, SectionType, Flags, QuantMethod
from zse_engine.format.header import FileHeader, SectionEntry, read_header_and_sections
from zse_engine.format.config import ModelConfig, QuantConfig
from zse_engine.format.weight_index import WeightIndex, WeightEntry
from zse_engine.format.quantize import quantize_tensor, dequantize_tensor
from zse_engine.format.tokenizer import BPETokenizer
from zse_engine.format.writer import ZSEWriter

__all__ = [
    "MAGIC", "VERSION", "SectionType", "Flags", "QuantMethod",
    "FileHeader", "SectionEntry", "read_header_and_sections",
    "ModelConfig", "QuantConfig",
    "WeightIndex", "WeightEntry",
    "quantize_tensor", "dequantize_tensor",
    "BPETokenizer",
    "ZSEWriter",
]
