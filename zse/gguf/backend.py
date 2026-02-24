"""
GGUF Backend using llama-cpp-python

Provides inference for GGUF models using the llama.cpp library.
This ensures maximum compatibility with GGUF quantization formats.
"""

import os
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class GGUFGenerationConfig:
    """Configuration for GGUF text generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = True


@dataclass
class GGUFModelConfig:
    """Configuration for loading GGUF models."""
    n_ctx: int = 4096  # Context length
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: Optional[int] = None  # CPU threads (None = auto)
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    use_mmap: bool = True  # Memory-map the model
    use_mlock: bool = False  # Lock model in RAM
    verbose: bool = False
    
    # GPU specific
    main_gpu: int = 0  # Main GPU for computations
    tensor_split: Optional[List[float]] = None  # Split across GPUs


class LlamaCppBackend:
    """
    Backend for running GGUF models using llama-cpp-python.
    
    Usage:
        backend = LlamaCppBackend("model.gguf")
        backend.load()
        
        # Generate text
        for text in backend.generate("Hello, world!"):
            print(text, end="", flush=True)
        
        # Chat completion
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        for text in backend.chat(messages):
            print(text, end="", flush=True)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[GGUFModelConfig] = None,
    ):
        self.model_path = Path(model_path)
        self.config = config or GGUFModelConfig()
        self._llama = None
        self._model_info: Optional[Dict[str, Any]] = None
    
    @property
    def is_loaded(self) -> bool:
        return self._llama is not None
    
    def load(self) -> "LlamaCppBackend":
        """Load the GGUF model."""
        if self._llama is not None:
            return self
        
        # Lazy import to avoid dependency issues
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF support.\n"
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load the model
        self._llama = Llama(
            model_path=str(self.model_path),
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            verbose=self.config.verbose,
            main_gpu=self.config.main_gpu,
            tensor_split=self.config.tensor_split,
        )
        
        # Extract model info
        self._model_info = {
            "path": str(self.model_path),
            "n_ctx": self._llama.n_ctx(),
            "n_vocab": self._llama.n_vocab(),
            "n_embd": self._llama.n_embd(),
        }
        
        return self
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._llama is not None:
            del self._llama
            self._llama = None
            self._model_info = None
    
    def generate(
        self,
        prompt: str,
        config: Optional[GGUFGenerationConfig] = None,
    ) -> Iterator[str]:
        """
        Generate text from a prompt.
        
        Yields text chunks for streaming.
        """
        if self._llama is None:
            self.load()
        
        config = config or GGUFGenerationConfig()
        
        if config.stream:
            # Streaming generation
            stream = self._llama(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop,
                stream=True,
            )
            
            for output in stream:
                choice = output["choices"][0]
                text = choice.get("text", "")
                if text:
                    yield text
                    
                # Check for stop
                if choice.get("finish_reason") is not None:
                    break
        else:
            # Non-streaming
            output = self._llama(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop,
                stream=False,
            )
            yield output["choices"][0]["text"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GGUFGenerationConfig] = None,
    ) -> Iterator[str]:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            config: Generation configuration
            
        Yields:
            Text chunks for streaming.
        """
        if self._llama is None:
            self.load()
        
        config = config or GGUFGenerationConfig()
        
        if config.stream:
            # Streaming chat
            stream = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop,
                stream=True,
            )
            
            for output in stream:
                choice = output["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
                    
                if choice.get("finish_reason") is not None:
                    break
        else:
            # Non-streaming
            output = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop,
                stream=False,
            )
            yield output["choices"][0]["message"]["content"]
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        if self._llama is None:
            self.load()
        return self._llama.tokenize(text.encode("utf-8"))
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        if self._llama is None:
            self.load()
        return self._llama.detokenize(tokens).decode("utf-8")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._llama is None:
            self.load()
        return self._model_info.copy()
    
    def embed(self, text: str) -> List[float]:
        """Get text embeddings (if model supports it)."""
        if self._llama is None:
            self.load()
        
        # Check if embedding mode is available
        output = self._llama.embed(text)
        return output


def check_llama_cpp_available() -> Dict[str, Any]:
    """
    Check if llama-cpp-python is installed and get its capabilities.
    
    Returns:
        Dict with 'available', 'version', 'cuda_available' keys.
    """
    result = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "metal_available": False,
        "error": None,
    }
    
    try:
        import llama_cpp
        result["available"] = True
        result["version"] = getattr(llama_cpp, "__version__", "unknown")
        
        # Check for CUDA support
        try:
            # Try to detect GPU support
            if hasattr(llama_cpp, "LLAMA_SUPPORTS_GPU_OFFLOAD"):
                result["cuda_available"] = llama_cpp.LLAMA_SUPPORTS_GPU_OFFLOAD
            else:
                # Check by attempting to load with GPU layers
                result["cuda_available"] = True  # Assume yes if installed
        except Exception:
            pass
        
        # Check for Metal support (macOS)
        try:
            import platform
            if platform.system() == "Darwin":
                result["metal_available"] = True  # llama.cpp supports Metal on macOS
        except Exception:
            pass
            
    except ImportError as e:
        result["error"] = str(e)
    
    return result
