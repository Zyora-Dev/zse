"""
zStream Streaming Model Wrapper

High-level API for streaming inference with HuggingFace models.

This wraps any transformer model and enables layer streaming,
allowing models much larger than GPU memory to run efficiently.

Example Usage:
    from transformers import AutoModelForCausalLM
    from zse.core.zstream import StreamingModel
    
    # Load 70B model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        torch_dtype=torch.float16,
        device_map="cpu",  # Load to CPU first
    )
    
    # Wrap for streaming (only need 24GB GPU!)
    streaming = StreamingModel(
        model,
        gpu_layers=4,      # Keep 4 layers on GPU at once
        prefetch_layers=2, # Prefetch next 2 layers
    )
    
    # Generate as normal
    output = streaming.generate(input_ids, max_new_tokens=100)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time

from .memory_tracker import MemoryTracker, MemoryPressure
from .streamer import LayerStreamer, StreamingForward
from .prefetcher import AsyncPrefetcher, PrefetchStrategy
from .offload import OffloadManager, StorageTier


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    
    # GPU settings
    gpu_layers: int = 4
    gpu_budget_gb: float = 20.0
    device: int = 0
    
    # Prefetching
    prefetch_layers: int = 2
    prefetch_strategy: str = "sequential"
    num_streams: int = 2
    
    # CPU settings  
    cpu_budget_gb: float = 32.0
    use_pinned_memory: bool = True
    
    # Disk settings
    enable_disk_offload: bool = False
    disk_path: Optional[str] = None
    
    # Memory management
    memory_threshold: float = 0.85
    critical_threshold: float = 0.95
    
    # Optimization
    compile_layers: bool = False
    use_flash_attention: bool = True


class StreamingModel(nn.Module):
    """
    Streaming wrapper for large language models.
    
    This is the main API for zStream. It wraps a HuggingFace model
    and enables running models larger than GPU memory through
    intelligent layer streaming.
    
    Key Features:
    - Automatic layer streaming based on forward pass
    - Async prefetching to hide transfer latency
    - Memory pressure-aware eviction
    - Seamless integration with transformers API
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[StreamingConfig] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.config = config or StreamingConfig(**kwargs)
        self.model = model
        
        # Detect model architecture
        self._detect_architecture()
        
        # Initialize components
        self._init_memory_tracker()
        self._init_offload_manager()
        self._init_layer_streamer()
        self._init_prefetcher()
        
        # Prepare layers for streaming
        self._prepare_layers()
        
        # Stats
        self._forward_count = 0
        self._total_time = 0.0
    
    def _detect_architecture(self):
        """Detect model architecture and layer structure."""
        self.model_type = "unknown"
        self.layers_attr = None
        self.num_layers = 0
        
        # Try common architectures
        architectures = [
            ("model.layers", "llama"),           # LLaMA, Mistral
            ("transformer.h", "gpt2"),           # GPT-2
            ("transformer.blocks", "falcon"),    # Falcon
            ("model.decoder.layers", "opt"),     # OPT
            ("gpt_neox.layers", "gpt_neox"),     # GPT-NeoX
        ]
        
        for attr_path, arch_name in architectures:
            try:
                layers = self._get_nested_attr(self.model, attr_path)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    self.model_type = arch_name
                    self.layers_attr = attr_path
                    self.num_layers = len(layers)
                    break
            except AttributeError:
                continue
        
        if self.num_layers == 0:
            raise ValueError("Could not detect model architecture")
        
        print(f"[zStream] Detected {self.model_type} model with {self.num_layers} layers")
    
    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Get nested attribute by path."""
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    
    def _set_nested_attr(self, obj: Any, attr_path: str, value: Any):
        """Set nested attribute by path."""
        parts = attr_path.split(".")
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, parts[-1], value)
    
    def _get_layers(self) -> nn.ModuleList:
        """Get the layers ModuleList."""
        return self._get_nested_attr(self.model, self.layers_attr)
    
    def _init_memory_tracker(self):
        """Initialize memory tracking."""
        self.memory_tracker = MemoryTracker(
            device=self.config.device,
            safety_margin=0.1,  # Keep 10% free for activations
        )
    
    def _init_offload_manager(self):
        """Initialize offload manager."""
        self.offload_manager = OffloadManager(
            gpu_budget_bytes=int(self.config.gpu_budget_gb * 1024**3),
            cpu_budget_bytes=int(self.config.cpu_budget_gb * 1024**3),
            disk_path=self.config.disk_path,
            use_pinned_memory=self.config.use_pinned_memory,
            device=self.config.device,
        )
    
    def _init_layer_streamer(self):
        """Initialize layer streamer."""
        from .streamer import StreamerConfig
        
        # Create config for streamer
        streamer_config = StreamerConfig(
            max_window_size=self.config.gpu_layers,
            prefetch_count=self.config.prefetch_layers,
            use_pinned_memory=self.config.use_pinned_memory,
        )
        
        self.streamer = LayerStreamer(
            model=self.model,
            config=streamer_config,
            device=self.config.device,
        )
    
    def _init_prefetcher(self):
        """Initialize async prefetcher."""
        strategy_map = {
            "sequential": PrefetchStrategy.SEQUENTIAL,
            "attention": PrefetchStrategy.PRIORITY,  # Map to priority-based
            "adaptive": PrefetchStrategy.ADAPTIVE,
        }
        strategy = strategy_map.get(
            self.config.prefetch_strategy,
            PrefetchStrategy.SEQUENTIAL,
        )
        
        # Create load/is_loaded functions from streamer
        def load_fn(layer_idx: int):
            self.streamer.get_layer(layer_idx)
        
        def is_loaded_fn(layer_idx: int) -> bool:
            return self.streamer.is_layer_on_gpu(layer_idx)
        
        self.prefetcher = AsyncPrefetcher(
            load_fn=load_fn,
            is_loaded_fn=is_loaded_fn,
            num_layers=self.num_layers,
            strategy=strategy,
            num_streams=self.config.num_streams,
            device=self.config.device,
        )
    
    def _prepare_layers(self):
        """Prepare layers for streaming - move initial layers to GPU."""
        device = f"cuda:{self.config.device}"
        layers = self._get_layers()
        
        # ALWAYS keep embeddings and final components on GPU
        # These are needed for every forward pass
        self._move_static_components_to_gpu(device)
        
        # Move first gpu_layers to GPU, rest stay on CPU
        # LayerStreamer already discovered and registered the layers
        for i, layer in enumerate(layers):
            if i < self.config.gpu_layers:
                layer.to(device)
                # Update streamer's layer state to GPU
                if i in self.streamer.layers:
                    from .streamer import LayerLocation
                    self.streamer.layers[i].location = LayerLocation.GPU
            else:
                layer.cpu()
                # Already on CPU by default
        
        print(f"[zStream] Prepared {self.num_layers} layers "
              f"({self.config.gpu_layers} on GPU, {self.num_layers - self.config.gpu_layers} on CPU)")
    
    def _move_static_components_to_gpu(self, device: str):
        """Move embedding and final layers to GPU - these must always be on GPU."""
        # Move embed_tokens
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            self.model.model.embed_tokens.to(device)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            self.model.transformer.wte.to(device)
            if hasattr(self.model.transformer, 'wpe'):
                self.model.transformer.wpe.to(device)
        
        # Move rotary_emb (required for Qwen2/Llama models)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb.to(device)
        
        # Move lm_head
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.to(device)
        
        # Move final norm layer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            self.model.model.norm.to(device)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            self.model.transformer.ln_f.to(device)
    
    def _get_layer(self, idx: int) -> nn.Module:
        """Get a layer, streaming it to GPU if needed."""
        layers = self._get_layers()
        
        # Use streamer to ensure layer is on GPU
        with self.streamer.get_layer(idx):
            pass  # Layer is now on GPU
        
        return layers[idx]
    
    def _streaming_forward_pass(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Run forward pass with layer streaming.
        
        This is the core streaming logic that iterates through layers
        while managing GPU memory.
        """
        layers = self._get_layers()
        
        # Compute position_embeddings for Qwen2/Llama models
        # This must be done once at model level and passed to all layers
        position_embeddings = None
        if self.model_type == "llama":
            inner_model = getattr(self.model, 'model', self.model)
            if hasattr(inner_model, 'rotary_emb'):
                rotary_emb = inner_model.rotary_emb
                position_embeddings = rotary_emb(hidden_states, position_ids)
        
        # Start prefetcher
        self.prefetcher.start()
        
        new_past_key_values = [] if past_key_values is not None else None
        
        try:
            for layer_idx in range(self.num_layers):
                # Ensure layer is on GPU
                layer = self.streamer.get_layer(layer_idx)
                
                # Signal prefetcher about layer access
                self.prefetcher.notify_access(layer_idx)
                
                try:
                    # Get past KV for this layer
                    layer_past = None
                    if past_key_values is not None:
                        layer_past = past_key_values[layer_idx] if layer_idx < len(past_key_values) else None
                    
                    # Forward through layer
                    layer_outputs = self._forward_layer(
                        layer=layer,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=layer_past,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    
                    # Unpack outputs - handle both tuple and tensor returns
                    if isinstance(layer_outputs, tuple):
                        hidden_states = layer_outputs[0]
                        if new_past_key_values is not None and len(layer_outputs) > 1:
                            new_past_key_values.append(layer_outputs[1])
                    elif isinstance(layer_outputs, torch.Tensor):
                        hidden_states = layer_outputs
                    else:
                        # Try to get hidden_states from first element
                        hidden_states = layer_outputs[0]
                        if new_past_key_values is not None and len(layer_outputs) > 1:
                            new_past_key_values.append(layer_outputs[1])
                finally:
                    # Release layer (marks as evictable)
                    self.streamer.release_layer(layer_idx)
                    
        finally:
            self.prefetcher.stop()
        
        return hidden_states, new_past_key_values
    
    def _forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple:
        """Forward through a single layer with architecture-specific handling."""
        
        # Architecture-specific forward calls
        if self.model_type == "llama":
            return layer(
                hidden_states,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=kwargs.get("position_ids"),
                past_key_value=kwargs.get("past_key_value"),
                use_cache=kwargs.get("past_key_value") is not None,
                position_embeddings=kwargs.get("position_embeddings"),
            )
        
        elif self.model_type == "gpt2":
            return layer(
                hidden_states,
                attention_mask=kwargs.get("attention_mask"),
                layer_past=kwargs.get("past_key_value"),
                use_cache=kwargs.get("past_key_value") is not None,
            )
        
        else:
            # Generic forward
            return layer(hidden_states, **kwargs)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with layer streaming.
        
        Compatible with HuggingFace generate() method.
        """
        start_time = time.perf_counter()
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self._get_embeddings(input_ids)
        
        # Compute position_ids if not provided
        if position_ids is None:
            device = inputs_embeds.device
            seq_length = inputs_embeds.shape[1]
            # Handle past key values offset
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                if past_key_values[0] is not None:
                    # Past key value shape: (2, batch, num_heads, past_len, head_dim) or similar
                    past_length = past_key_values[0][0].shape[-2]
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Run streaming forward
        hidden_states, new_past_key_values = self._streaming_forward_pass(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        
        # Apply final layer norm
        hidden_states = self._apply_final_layernorm(hidden_states)
        
        # Get logits
        logits = self._get_lm_head(hidden_states)
        
        # Update stats
        self._forward_count += 1
        self._total_time += time.perf_counter() - start_time
        
        # Return in HuggingFace format
        return StreamingModelOutput(
            logits=logits,
            past_key_values=new_past_key_values,
            hidden_states=hidden_states,
        )
    
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings."""
        if self.model_type == "llama":
            return self.model.model.embed_tokens(input_ids)
        elif self.model_type == "gpt2":
            return self.model.transformer.wte(input_ids)
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}")
    
    def _apply_final_layernorm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization."""
        if self.model_type == "llama":
            return self.model.model.norm(hidden_states)
        elif self.model_type == "gpt2":
            return self.model.transformer.ln_f(hidden_states)
        else:
            return hidden_states
    
    def _get_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get output logits."""
        if self.model_type == "llama":
            return self.model.lm_head(hidden_states)
        elif self.model_type == "gpt2":
            return self.model.lm_head(hidden_states)
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text with streaming inference.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy
            
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        device = f"cuda:{self.config.device}"
        input_ids = input_ids.to(device)
        
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Only use last token for incremental generation
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            outputs = self.forward(
                input_ids=curr_input,
                past_key_values=past_key_values,
            )
            
            logits = outputs.logits[:, -1, :]  # Last position
            past_key_values = outputs.past_key_values
            
            # Sample next token
            if do_sample and temperature > 0:
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "forward_count": self._forward_count,
            "total_time": self._total_time,
            "avg_time_per_forward": self._total_time / max(1, self._forward_count),
            "streamer_stats": self.streamer.get_stats(),
            "prefetcher_stats": self.prefetcher.get_stats(),
            "offload_stats": self.offload_manager.get_stats(),
        }
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop prefetcher if running
        if hasattr(self.prefetcher, 'stop'):
            self.prefetcher.stop()
        elif hasattr(self.prefetcher, 'cleanup'):
            self.prefetcher.cleanup()
        
        # Cleanup offload manager if method exists
        if hasattr(self.offload_manager, 'cleanup'):
            self.offload_manager.cleanup()


class StreamingModelOutput:
    """Output from streaming model forward pass."""
    
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: Optional[List] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states


def wrap_model_for_streaming(
    model: nn.Module,
    gpu_layers: int = 4,
    prefetch_layers: int = 2,
    device: int = 0,
    **kwargs,
) -> StreamingModel:
    """
    Convenience function to wrap a model for streaming.
    
    Args:
        model: HuggingFace model
        gpu_layers: Number of layers to keep on GPU
        prefetch_layers: Number of layers to prefetch
        device: GPU device ID
        
    Returns:
        StreamingModel wrapper
        
    Example:
        model = AutoModelForCausalLM.from_pretrained("...", device_map="cpu")
        streaming = wrap_model_for_streaming(model, gpu_layers=6)
        output = streaming.generate(input_ids, max_new_tokens=50)
    """
    config = StreamingConfig(
        gpu_layers=gpu_layers,
        prefetch_layers=prefetch_layers,
        device=device,
        **kwargs,
    )
    return StreamingModel(model, config)
