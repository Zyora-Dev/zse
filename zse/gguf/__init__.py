"""
GGUF Compatibility Layer

Provides full support for GGUF models via llama-cpp-python.

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
and is widely used for quantized models (Q4_K_M, Q5_K_M, etc.).

This module provides:
- GGUFReader: Parse GGUF file metadata and tensors
- LlamaCppBackend: Direct llama.cpp inference backend
- GGUFWrapper: High-level wrapper matching ZSE orchestrator API

Usage:
    from zse.gguf import GGUFWrapper, is_gguf_file

    # Check if file is GGUF
    if is_gguf_file("model.gguf"):
        wrapper = GGUFWrapper("model.gguf")
        wrapper.load()
        
        # Generate text
        for text in wrapper.generate("Hello"):
            print(text, end="")

Requirements:
    pip install llama-cpp-python
    
    For GPU support:
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
"""

from zse.gguf.reader import (
    GGUFReader,
    GGUFFile,
    GGUFTensorInfo,
    GGMLQuantType,
    is_gguf_file,
)
from zse.gguf.backend import (
    LlamaCppBackend,
    GGUFModelConfig,
    GGUFGenerationConfig,
    check_llama_cpp_available,
)
from zse.gguf.wrapper import (
    GGUFWrapper,
    load_gguf_model,
    detect_model_format,
)

__all__ = [
    # Reader
    "GGUFReader",
    "GGUFFile",
    "GGUFTensorInfo",
    "GGMLQuantType",
    "is_gguf_file",
    # Backend
    "LlamaCppBackend",
    "GGUFModelConfig", 
    "GGUFGenerationConfig",
    "check_llama_cpp_available",
    # Wrapper
    "GGUFWrapper",
    "load_gguf_model",
    "detect_model_format",
]
