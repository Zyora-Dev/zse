"""
GGUF Wrapper for ZSE Integration

Provides a unified interface for GGUF models that matches
the ZSE orchestrator API pattern.
"""

from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Union
from dataclasses import dataclass
import time

from zse.gguf.reader import GGUFReader, is_gguf_file
from zse.gguf.backend import (
    LlamaCppBackend,
    GGUFModelConfig,
    GGUFGenerationConfig,
    check_llama_cpp_available,
)


@dataclass
class GGUFStats:
    """Statistics from GGUF inference."""
    tokens_generated: int
    total_time_sec: float
    tokens_per_sec: float
    prompt_tokens: int
    completion_tokens: int


class GGUFWrapper:
    """
    High-level wrapper for GGUF models.
    
    Provides the same interface as IntelligenceOrchestrator
    for seamless integration with ZSE.
    
    Usage:
        # Load a GGUF model
        wrapper = GGUFWrapper("model-Q4_K_M.gguf")
        wrapper.load()
        
        # Generate text
        for text in wrapper.generate("Hello"):
            print(text, end="")
        
        # Chat
        response = wrapper.chat("What is 2+2?")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all on GPU
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self._config = GGUFModelConfig(
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )
        self._backend: Optional[LlamaCppBackend] = None
        self._gguf_info: Optional[Dict[str, Any]] = None
        self._is_loaded = False
        
    @classmethod
    def from_path(
        cls,
        model_path: Union[str, Path],
        **kwargs,
    ) -> "GGUFWrapper":
        """Create wrapper from a GGUF file path."""
        return cls(model_path, **kwargs)
    
    @classmethod
    def auto(
        cls,
        model_path: Union[str, Path],
        max_memory_gb: Optional[float] = None,
    ) -> "GGUFWrapper":
        """
        Create wrapper with auto-detected optimal settings.
        
        Args:
            model_path: Path to GGUF file
            max_memory_gb: Maximum GPU memory to use (None = auto)
        """
        # Read GGUF metadata to determine settings
        reader = GGUFReader(model_path)
        info = reader.get_model_info()
        
        # Determine GPU layers based on available memory
        n_gpu_layers = -1  # Default: all on GPU
        
        if max_memory_gb is not None:
            model_size = info.get("total_size_gb", 10)
            if model_size > max_memory_gb * 0.9:
                # Need to offload some layers to CPU
                # Rough estimate: distribute layers proportionally
                total_layers = info.get("layers", 32)
                gpu_ratio = max_memory_gb / model_size
                n_gpu_layers = int(total_layers * gpu_ratio * 0.8)  # 80% headroom
        
        # Determine context length
        n_ctx = min(info.get("context_length", 4096), 4096)
        
        return cls(
            model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load(self, verbose: bool = True) -> "GGUFWrapper":
        """Load the GGUF model."""
        if self._is_loaded:
            return self
        
        # Check llama.cpp availability
        llama_status = check_llama_cpp_available()
        if not llama_status["available"]:
            raise ImportError(
                "llama-cpp-python is required for GGUF support.\n"
                f"Error: {llama_status.get('error', 'Not installed')}\n"
                "Install with: pip install llama-cpp-python"
            )
        
        # Read GGUF metadata
        if verbose:
            print(f"📂 Reading GGUF file: {self.model_path.name}")
        
        reader = GGUFReader(self.model_path)
        self._gguf_info = reader.get_model_info()
        
        if verbose:
            print(f"   Architecture: {self._gguf_info['architecture']}")
            print(f"   Quantization: {self._gguf_info['quantization']}")
            print(f"   Layers: {self._gguf_info['layers']}")
            print(f"   Size: {self._gguf_info['total_size_gb']:.2f} GB")
        
        # Create and load backend
        if verbose:
            gpu_layers = self._config.n_gpu_layers
            if gpu_layers == -1:
                print(f"📥 Loading model (all layers on GPU)...")
            elif gpu_layers == 0:
                print(f"📥 Loading model (CPU only)...")
            else:
                print(f"📥 Loading model ({gpu_layers} layers on GPU)...")
        
        start = time.perf_counter()
        self._backend = LlamaCppBackend(self.model_path, self._config)
        self._backend.load()
        load_time = time.perf_counter() - start
        
        if verbose:
            print(f"\n✅ GGUF model loaded in {load_time:.1f}s")
            print(f"   Context length: {self._config.n_ctx}")
        
        self._is_loaded = True
        return self
    
    def unload(self) -> None:
        """Unload the model."""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        self._is_loaded = False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stream: If True, return iterator of chunks
            
        Returns:
            Generated text or iterator
        """
        if not self._is_loaded:
            self.load()
        
        config = GGUFGenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        
        if stream:
            return self._backend.generate(prompt, config)
        else:
            # Collect all output
            result = []
            for chunk in self._backend.generate(prompt, config):
                result.append(chunk)
            return "".join(result)
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Union[str, Iterator[str]]:
        """
        Chat with the model.
        
        Args:
            message: User message
            system_prompt: Optional system prompt  
            history: Previous messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, return iterator of chunks
            
        Returns:
            Assistant response or iterator
        """
        if not self._is_loaded:
            self.load()
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": message})
        
        config = GGUFGenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        
        if stream:
            return self._backend.chat(messages, config)
        else:
            result = []
            for chunk in self._backend.chat(messages, config):
                result.append(chunk)
            return "".join(result)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "path": str(self.model_path),
            "format": "gguf",
            "is_loaded": self._is_loaded,
        }
        
        if self._gguf_info:
            info.update(self._gguf_info)
        
        if self._backend and self._is_loaded:
            info["backend_info"] = self._backend.get_model_info()
        
        return info
    
    # Compatibility methods for orchestrator API
    
    @property 
    def model_name(self) -> str:
        """Model name for compatibility."""
        return self.model_path.stem
    
    @property
    def quantization(self) -> str:
        """Quantization type."""
        if self._gguf_info:
            return self._gguf_info.get("quantization", "unknown")
        return "gguf"
    
    @property
    def device(self) -> str:
        """Device being used."""
        if self._config.n_gpu_layers == 0:
            return "cpu"
        return "cuda"  # llama.cpp uses CUDA or Metal


def load_gguf_model(
    path: Union[str, Path],
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    verbose: bool = True,
) -> GGUFWrapper:
    """
    Convenience function to load a GGUF model.
    
    Args:
        path: Path to GGUF file
        n_gpu_layers: GPU layers (-1 = all, 0 = CPU only)
        n_ctx: Context length
        verbose: Print loading progress
        
    Returns:
        Loaded GGUFWrapper
    """
    wrapper = GGUFWrapper(path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    wrapper.load(verbose=verbose)
    return wrapper


def detect_model_format(path: Union[str, Path]) -> str:
    """
    Detect the format of a model file/directory.
    
    Returns:
        "gguf", "safetensors", "pytorch", "zse", or "huggingface"
    """
    path = Path(path)
    
    if not path.exists():
        # Might be a HuggingFace model ID
        if "/" in str(path) and not str(path).startswith("/"):
            return "huggingface"
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Check file extension
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in (".gguf", ".ggml"):
            return "gguf"
        elif suffix == ".zse":
            return "zse"
        elif suffix == ".safetensors":
            return "safetensors"
        elif suffix in (".pt", ".pth", ".bin"):
            return "pytorch"
    
    # Check directory contents
    if path.is_dir():
        files = list(path.iterdir())
        file_names = [f.name for f in files]
        
        if "config.json" in file_names:
            return "huggingface"
        if any(f.endswith(".gguf") for f in file_names):
            return "gguf"
        if any(f.endswith(".zse") for f in file_names):
            return "zse"
        if any(f.endswith(".safetensors") for f in file_names):
            return "safetensors"
    
    return "unknown"
