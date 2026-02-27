"""
ZSE Intelligence Orchestrator

Automatically optimizes model loading based on available VRAM.
Provides flexible memory/speed trade-offs for inference.

Memory Targets for 7B models:
- INT4: ~3.5 GB (minimum memory, slower inference)
- INT8: ~7.5 GB (balanced)
- FP16: ~14 GB (maximum speed, full precision)

Usage:
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    # Auto-detect best config
    orchestrator = IntelligenceOrchestrator.auto("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    # Or specify preference
    orchestrator = IntelligenceOrchestrator.for_vram(4.0, "Qwen/Qwen2.5-Coder-7B-Instruct")
    orchestrator = IntelligenceOrchestrator.min_memory("Qwen/Qwen2.5-Coder-7B-Instruct")
    orchestrator = IntelligenceOrchestrator.max_speed("Qwen/Qwen2.5-Coder-7B-Instruct")
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Iterator, Union
from pathlib import Path
import gc
import time


class OptimizationMode(Enum):
    """Memory/speed optimization modes."""
    MIN_MEMORY = "min_memory"      # INT4 - Minimum VRAM, slower
    BALANCED = "balanced"          # INT8 - Balanced
    MAX_SPEED = "max_speed"        # FP16 - Maximum speed
    AUTO = "auto"                  # Auto-detect based on VRAM


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str
    quantization: str  # "fp16", "int8", "int4"
    device: str
    estimated_vram_gb: float
    expected_tokens_per_sec: float
    
    def __repr__(self):
        return (
            f"ModelConfig(\n"
            f"  model={self.model_name},\n"
            f"  quant={self.quantization},\n"
            f"  vram={self.estimated_vram_gb:.1f}GB,\n"
            f"  speed=~{self.expected_tokens_per_sec:.0f} tok/s\n"
            f")"
        )


@dataclass 
class InferenceStats:
    """Statistics from inference."""
    tokens_generated: int
    total_time_sec: float
    tokens_per_sec: float
    avg_latency_ms: float
    peak_memory_gb: float
    quantization: str


class IntelligenceOrchestrator:
    """
    Orchestrates model loading with optimal memory/speed configuration.
    
    Automatically selects quantization based on:
    - Available VRAM
    - User preference (min memory vs max speed)
    - Model size
    """
    
    # Approximate memory requirements per billion parameters
    BYTES_PER_PARAM = {
        "fp32": 4.0,      # 4 bytes per param (for CPU)
        "fp16": 2.0,      # 2 bytes per param
        "int8": 1.0,      # 1 byte per param  
        "int4": 0.5,      # 0.5 bytes per param (packed)
    }
    
    # Approximate tokens/sec for A10G GPU (baseline)
    SPEED_MULTIPLIER = {
        "fp32": 0.1,      # CPU FP32 is much slower
        "fp16": 1.0,      # Baseline speed
        "int8": 0.3,      # ~30% of FP16 speed (dequantization overhead)
        "int4": 0.2,      # ~20% of FP16 speed (more overhead)
    }
    
    # Base tokens/sec for 7B model on A10G
    BASE_TOKENS_PER_SEC_7B = 15.0
    
    def __init__(
        self,
        model_name: str,
        quantization: str = "auto",
        target_vram_gb: Optional[float] = None,
        device: str = "auto",
        multi_gpu: bool = False,
        gpu_ids: Optional[list] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            model_name: HuggingFace model name
            quantization: "fp16", "int8", "int4", or "auto"
            target_vram_gb: Target VRAM usage (for auto mode)
            device: Target device ("auto", "cuda", "cpu", or "cuda:N")
            multi_gpu: Enable multi-GPU tensor parallelism
            gpu_ids: Specific GPU IDs to use (e.g., [0, 1, 2])
        """
        self.model_name = model_name
        self.target_vram_gb = target_vram_gb
        self._multi_gpu = multi_gpu
        self._gpu_ids = gpu_ids
        
        # Resolve "auto" device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Track if using CPU
        self._is_cpu = self.device == "cpu"
        
        # Detect GPU count
        if torch.cuda.is_available():
            self._gpu_count = torch.cuda.device_count()
        else:
            self._gpu_count = 0
        
        # Determine quantization (CPU can only use FP16/FP32, no bitsandbytes)
        if self._is_cpu:
            # CPU mode: use FP32 for compatibility (FP16 can be slow on CPU)
            if quantization in ("int4", "int8"):
                print(f"⚠️  Note: {quantization.upper()} quantization requires GPU. Using FP32 for CPU.")
                self.quantization = "fp32"
            elif quantization == "auto":
                self.quantization = "fp32"  # Default to FP32 for CPU
            else:
                self.quantization = quantization
        else:
            # GPU mode: auto-select or use specified
            if quantization == "auto":
                self.quantization = self._auto_select_quantization(target_vram_gb)
            else:
                self.quantization = quantization
        
        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._is_loaded = False
        self._config = None
        
    @classmethod
    def auto(cls, model_name: str, device: str = "auto") -> "IntelligenceOrchestrator":
        """
        Create orchestrator with auto-detected optimal configuration.
        
        Detects available VRAM and selects best quantization.
        Auto-detects GPU/CPU if device='auto'.
        """
        return cls(model_name, quantization="auto", device=device)
    
    @classmethod
    def for_vram(cls, vram_gb: float, model_name: str, device: str = "auto") -> "IntelligenceOrchestrator":
        """
        Create orchestrator optimized for specific VRAM budget.
        
        Args:
            vram_gb: Available VRAM in GB
            model_name: Model to load
        """
        return cls(model_name, quantization="auto", target_vram_gb=vram_gb, device=device)
    
    @classmethod
    def min_memory(cls, model_name: str, device: str = "auto") -> "IntelligenceOrchestrator":
        """
        Create orchestrator with minimum memory (INT4 on GPU, FP32 on CPU).
        
        Best for: Limited VRAM, willing to sacrifice speed.
        """
        return cls(model_name, quantization="int4", device=device)
    
    @classmethod
    def max_speed(cls, model_name: str, device: str = "auto") -> "IntelligenceOrchestrator":
        """
        Create orchestrator with maximum speed (FP16).
        
        Best for: Abundant VRAM, want fastest inference.
        """
        return cls(model_name, quantization="fp16", device=device)
    
    @classmethod
    def balanced(cls, model_name: str, device: str = "auto") -> "IntelligenceOrchestrator":
        """
        Create orchestrator with balanced config (INT8 on GPU, FP32 on CPU).
        
        Best for: Good balance of memory and speed.
        """
        return cls(model_name, quantization="int8", device=device)
    
    @classmethod
    def multi_gpu(
        cls, 
        model_name: str, 
        quantization: str = "auto",
        gpu_ids: Optional[list] = None
    ) -> "IntelligenceOrchestrator":
        """
        Create orchestrator that splits model across multiple GPUs.
        
        Uses tensor parallelism to distribute model layers across GPUs.
        
        Args:
            model_name: HuggingFace model name
            quantization: "fp16", "int8", "int4", or "auto"
            gpu_ids: List of GPU IDs to use (e.g., [0, 1]). None = use all GPUs.
            
        Example:
            # Use all available GPUs
            orch = IntelligenceOrchestrator.multi_gpu("meta-llama/Llama-2-70b-hf")
            
            # Use specific GPUs
            orch = IntelligenceOrchestrator.multi_gpu(
                "meta-llama/Llama-2-70b-hf", 
                gpu_ids=[0, 2, 3]
            )
        """
        return cls(
            model_name, 
            quantization=quantization, 
            device="auto",  # Will be handled by device_map
            multi_gpu=True,
            gpu_ids=gpu_ids
        )
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Get information about available GPUs.
        
        Returns:
            Dict with GPU count, names, and memory info.
        """
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "gpus": []}
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        total_vram = 0
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_info = torch.cuda.mem_get_info(i)
            free_gb = mem_info[0] / (1024**3)
            total_gb = mem_info[1] / (1024**3)
            
            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": round(total_gb, 2),
                "free_memory_gb": round(free_gb, 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })
            total_vram += total_gb
        
        return {
            "available": True,
            "count": gpu_count,
            "total_vram_gb": round(total_vram, 2),
            "gpus": gpus
        }
    
    def _auto_select_quantization(self, target_vram_gb: Optional[float] = None) -> str:
        """Auto-select quantization based on available/target VRAM."""
        if target_vram_gb is None:
            # Detect available VRAM
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                available_gb = props.total_memory / (1024**3)
                # Reserve 2GB for KV cache and overhead
                target_vram_gb = available_gb - 2.0
            else:
                target_vram_gb = 4.0  # Default to minimum
        
        # Estimate model size (assume 7B if not known)
        estimated_params = 7.0  # billion
        
        # Calculate memory for each quantization
        fp16_gb = estimated_params * self.BYTES_PER_PARAM["fp16"]
        int8_gb = estimated_params * self.BYTES_PER_PARAM["int8"]
        int4_gb = estimated_params * self.BYTES_PER_PARAM["int4"]
        
        # Select best fitting quantization
        if target_vram_gb >= fp16_gb * 1.1:  # 10% headroom
            return "fp16"
        elif target_vram_gb >= int8_gb * 1.1:
            return "int8"
        else:
            return "int4"
    
    def get_config(self) -> ModelConfig:
        """Get the selected configuration."""
        if self._config is None:
            # Estimate parameters (will be updated after loading)
            estimated_params_b = 7.0
            
            vram = estimated_params_b * self.BYTES_PER_PARAM[self.quantization]
            speed = self.BASE_TOKENS_PER_SEC_7B * self.SPEED_MULTIPLIER[self.quantization]
            
            self._config = ModelConfig(
                model_name=self.model_name,
                quantization=self.quantization,
                device=self.device,
                estimated_vram_gb=vram,
                expected_tokens_per_sec=speed,
            )
        return self._config
    
    def load(self, verbose: bool = True) -> "IntelligenceOrchestrator":
        """
        Load the model with selected configuration.
        
        Returns self for chaining.
        """
        if self._is_loaded:
            return self
        
        config = self.get_config()
        
        if verbose:
            print(f"🚀 ZSE Intelligence Orchestrator")
            print(f"   Model: {self.model_name}")
            print(f"   Quantization: {self.quantization.upper()}")
            if self._multi_gpu:
                gpu_str = f"GPUs {self._gpu_ids}" if self._gpu_ids else f"All {self._gpu_count} GPUs"
                print(f"   Multi-GPU: {gpu_str}")
            print(f"   Expected VRAM: ~{config.estimated_vram_gb:.1f} GB")
            print(f"   Expected Speed: ~{config.expected_tokens_per_sec:.0f} tok/s")
            print()
        
        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from zse.engine.generation import TextGenerator
        from zse.efficiency.quantization import quantize_model, QuantType
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        start = time.perf_counter()
        
        # Load tokenizer
        if verbose:
            print("📥 Loading tokenizer...")
        
        # Check if model_name is a local .zse file/directory
        model_path = Path(self.model_name)
        self._is_zse_format = False  # Track if loading from .zse
        self._zse_path = None
        
        if model_path.exists():
            # Local path - check for tokenizer files
            if model_path.is_dir():
                tokenizer_config = model_path / "tokenizer_config.json"
                if tokenizer_config.exists():
                    # Load tokenizer from local directory
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path),
                        trust_remote_code=True,
                    )
                    # Check if this is a converted .zse directory (has model.safetensors)
                    if (model_path / "model.safetensors").exists() or (model_path / "model.zse").exists():
                        self._is_zse_format = True
                        self._zse_path = model_path
                else:
                    # Check if it's a .zse binary file with embedded tokenizer
                    zse_file = model_path / "model.zse"
                    if zse_file.exists():
                        from zse.format.reader import ZSEReader
                        reader = ZSEReader(zse_file)
                        self.tokenizer = reader.load_tokenizer()
                        self._is_zse_format = True
                        self._zse_path = zse_file
                    else:
                        raise ValueError(f"No tokenizer found in {model_path}. Directory must contain tokenizer_config.json or model.zse")
            elif model_path.suffix == ".zse" or str(model_path).endswith(".zse"):
                # Single .zse binary file
                from zse.format.reader import ZSEReader
                reader = ZSEReader(model_path)
                self.tokenizer = reader.load_tokenizer()
                self._is_zse_format = True
                self._zse_path = model_path
            else:
                # Try loading as HuggingFace local path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                )
        else:
            # HuggingFace model ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
        
        # Determine device_map for model loading
        if self._is_cpu:
            # CPU mode: load directly to CPU
            device_map = "cpu"
            max_memory = None
        elif self._multi_gpu:
            # Multi-GPU: use device_map="auto" or custom mapping
            if self._gpu_ids:
                # Custom GPU selection - create max_memory dict
                max_memory = {i: "80GiB" for i in self._gpu_ids}
                max_memory["cpu"] = "100GiB"
                device_map = "auto"
            else:
                # Use all available GPUs
                max_memory = None
                device_map = "auto"
        else:
            device_map = self.device
            max_memory = None
        
        # Load model
        if verbose:
            if self._is_cpu:
                mode = f"CPU {self.quantization.upper()}"
            elif self._multi_gpu:
                mode = "Multi-GPU"
            elif self._is_zse_format:
                mode = "ZSE Format (Pre-quantized)"
            else:
                mode = self.quantization.upper()
            print(f"📥 Loading model ({mode})...")
        
        # Special fast path for .zse format
        if self._is_zse_format and self._zse_path is not None:
            # Load from pre-converted .zse format (memory-mapped, fast)
            from safetensors.torch import load_file
            
            zse_path = Path(self._zse_path)
            
            if zse_path.is_dir():
                # Directory format - load safetensors directly
                safetensors_file = zse_path / "model.safetensors"
                if safetensors_file.exists():
                    # Get config for model architecture
                    config_file = zse_path / "config.json"
                    if config_file.exists():
                        from transformers import AutoConfig
                        model_config = AutoConfig.from_pretrained(str(zse_path), trust_remote_code=True)
                        
                        # Create model structure
                        with torch.device("meta"):
                            self.model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
                        
                        # Load weights with memory mapping
                        state_dict = load_file(str(safetensors_file), device=self.device)
                        self.model.load_state_dict(state_dict, assign=True, strict=False)
                    else:
                        # Fall back to standard loading
                        self.model = AutoModelForCausalLM.from_pretrained(
                            str(zse_path),
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                else:
                    # No safetensors, use standard loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(zse_path),
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
            else:
                # Binary .zse file - use reader
                from zse.format.reader_v2 import load_zse_model
                self.model, _, _ = load_zse_model(str(zse_path), device=self.device)
        
        elif self.quantization in ("fp32", "fp16") or self._is_cpu:
            # Direct loading (CPU uses FP32, GPU can use FP16)
            dtype = torch.float32 if (self.quantization == "fp32" or self._is_cpu) else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Efficient loading for CPU
            )
        elif self.quantization == "int8":
            # Use bitsandbytes INT8 quantization (battle-tested)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
            )
        else:
            # INT4: Use bitsandbytes NF4 quantization (battle-tested)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Extra compression
                bnb_4bit_quant_type="nf4",  # NormalFloat4 for better quality
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
            )
        
        # Create generator
        gen_device = self.device
        if self._multi_gpu:
            gen_device = "cuda"  # Multi-GPU: model handles placement
        self.generator = TextGenerator(self.model, self.tokenizer, device=gen_device)
        
        load_time = time.perf_counter() - start
        
        # Update config with actual values
        if self._is_cpu:
            # CPU mode: report RAM usage (approximate based on model size)
            import psutil
            process = psutil.Process()
            actual_memory = process.memory_info().rss / (1024**3)  # RSS in GB
        elif self._multi_gpu:
            # Sum VRAM across all GPUs
            total_vram = 0
            for i in range(self._gpu_count):
                total_vram += torch.cuda.memory_allocated(i) / (1024**3)
            actual_memory = total_vram
        else:
            actual_memory = torch.cuda.memory_allocated() / (1024**3)
        actual_params = sum(p.numel() for p in self.model.parameters())
        
        self._config = ModelConfig(
            model_name=self.model_name,
            quantization=self.quantization,
            device=self.device,
            estimated_vram_gb=actual_memory,
            expected_tokens_per_sec=config.expected_tokens_per_sec,
        )
        
        if verbose:
            print(f"\n✅ Model loaded in {load_time:.1f}s")
            print(f"   Parameters: {actual_params:,} ({actual_params/1e9:.2f}B)")
            if self._is_cpu:
                print(f"   RAM Used: {actual_memory:.2f} GB")
                print(f"   Mode: CPU inference (slower but GPU-free)")
            elif self._multi_gpu:
                print(f"   Total VRAM Used: {actual_memory:.2f} GB (across {self._gpu_count} GPUs)")
                # Show per-GPU breakdown
                for i in range(self._gpu_count):
                    gpu_vram = torch.cuda.memory_allocated(i) / (1024**3)
                    if gpu_vram > 0:
                        print(f"   GPU {i}: {gpu_vram:.2f} GB")
            else:
                print(f"   VRAM Used: {actual_memory:.2f} GB")
            print()
        
        self._is_loaded = True
        return self
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stream: If True, return iterator of chunks
            
        Returns:
            Generated text or iterator of chunks
        """
        if not self._is_loaded:
            self.load()
        
        from zse.engine.generation import SamplingParams
        
        params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        if stream:
            return self._stream_generate(prompt, params)
        else:
            return self._batch_generate(prompt, params)
    
    def _stream_generate(self, prompt: str, params) -> Iterator[str]:
        """Stream generation."""
        for chunk in self.generator.generate_stream(prompt, params):
            yield chunk.text
    
    def _batch_generate(self, prompt: str, params) -> str:
        """Batch generation."""
        result = []
        for chunk in self.generator.generate_stream(prompt, params):
            result.append(chunk.text)
        return "".join(result)
    
    def benchmark(self, prompt: str = "Write a hello world in Python", tokens: int = 100) -> InferenceStats:
        """
        Run a benchmark to measure actual performance.
        
        Returns stats about generation speed and memory.
        """
        if not self._is_loaded:
            self.load()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        token_count = 0
        
        for chunk in self.generate(prompt, max_tokens=tokens, stream=True):
            token_count += 1  # Approximate
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        return InferenceStats(
            tokens_generated=tokens,
            total_time_sec=elapsed,
            tokens_per_sec=tokens / elapsed,
            avg_latency_ms=(elapsed / tokens) * 1000,
            peak_memory_gb=peak_memory,
            quantization=self.quantization,
        )
    
    def unload(self):
        """Unload model and free memory."""
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._is_loaded = False
        torch.cuda.empty_cache()
        gc.collect()
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, *args):
        self.unload()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_model(
    model_name: str,
    mode: str = "auto",
    vram_gb: Optional[float] = None,
) -> IntelligenceOrchestrator:
    """
    Load a model with automatic optimization.
    
    Args:
        model_name: HuggingFace model name
        mode: "auto", "min_memory", "balanced", or "max_speed"
        vram_gb: Target VRAM (for auto mode)
        
    Returns:
        Loaded orchestrator ready for inference
        
    Example:
        # Auto-detect best config
        orch = load_model("Qwen/Qwen2.5-Coder-7B-Instruct")
        
        # Fit in 4GB
        orch = load_model("Qwen/Qwen2.5-Coder-7B-Instruct", vram_gb=4.0)
        
        # Minimum memory
        orch = load_model("Qwen/Qwen2.5-Coder-7B-Instruct", mode="min_memory")
        
        # Generate
        for text in orch.generate("Hello"):
            print(text, end="", flush=True)
    """
    if mode == "min_memory":
        orch = IntelligenceOrchestrator.min_memory(model_name)
    elif mode == "max_speed":
        orch = IntelligenceOrchestrator.max_speed(model_name)
    elif mode == "balanced":
        orch = IntelligenceOrchestrator.balanced(model_name)
    elif vram_gb is not None:
        orch = IntelligenceOrchestrator.for_vram(vram_gb, model_name)
    else:
        orch = IntelligenceOrchestrator.auto(model_name)
    
    return orch.load()


def estimate_requirements(model_name: str) -> Dict[str, Any]:
    """
    Estimate VRAM requirements for a model.
    
    Returns dict with requirements for each quantization level.
    """
    import re
    
    # Estimate params from model name (heuristic)
    params_b = 7.0  # Default 7B
    
    name_lower = model_name.lower()
    
    # Try to extract number from patterns like "7b", "70b", "1.5b", "2.7b"
    patterns = [
        r'(\d+\.?\d*)b(?:\W|$)',  # matches "7b", "70b", "1.5b"
        r'-(\d+)b-',              # matches "-7b-", "-70b-"
        r'_(\d+)b_',              # matches "_7b_"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_lower)
        if match:
            try:
                params_b = float(match.group(1))
                break
            except ValueError:
                continue
    
    # Fallback to explicit checks if regex failed
    if params_b == 7.0:  # Still default
        if "1b" in name_lower or "1.1b" in name_lower:
            params_b = 1.0
        elif "3b" in name_lower:
            params_b = 3.0
        elif "8b" in name_lower:
            params_b = 8.0
        elif "13b" in name_lower or "14b" in name_lower:
            params_b = 14.0
        elif "32b" in name_lower or "33b" in name_lower or "34b" in name_lower:
            params_b = 32.0
        elif "65b" in name_lower or "67b" in name_lower or "70b" in name_lower:
            params_b = 70.0
        elif "72b" in name_lower:
            params_b = 72.0
        elif "180b" in name_lower:
            params_b = 180.0
        elif "405b" in name_lower:
            params_b = 405.0
    
    return {
        "model": model_name,
        "estimated_params_b": params_b,
        "requirements": {
            "fp16": {
                "vram_gb": params_b * 2.0,
                "speed": "fastest",
                "quality": "best",
            },
            "int8": {
                "vram_gb": params_b * 1.0,
                "speed": "medium",
                "quality": "good",
            },
            "int4": {
                "vram_gb": params_b * 0.5,
                "speed": "slower",
                "quality": "acceptable",
            },
        },
        "recommendations": {
            "4gb_gpu": "int4",
            "8gb_gpu": "int8",
            "16gb_gpu": "fp16",
            "24gb_gpu": "fp16",
        }
    }
