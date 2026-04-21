"""ZSE Architecture Adapters — Map HuggingFace model configs to ZSE format."""

from zse_engine.format.arch.base import ArchAdapter, get_adapter, detect_architecture
from zse_engine.format.arch.llama import LlamaAdapter, MistralAdapter
from zse_engine.format.arch.qwen2 import Qwen2Adapter
from zse_engine.format.arch.phi3 import Phi3Adapter
from zse_engine.format.arch.gemma2 import Gemma2Adapter

__all__ = [
    "ArchAdapter", "get_adapter", "detect_architecture",
    "LlamaAdapter", "MistralAdapter", "Qwen2Adapter",
    "Phi3Adapter", "Gemma2Adapter",
]
