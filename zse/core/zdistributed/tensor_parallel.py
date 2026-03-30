"""
Tensor Parallelism for ZSE

Splits model layers across multiple GPUs using column/row parallelism.
Supports both standard nn.Linear and QuantizedLinearZSE layers.

Architecture:
- ColumnParallelLinear: Splits output dimension across GPUs (for QKV, gate, up projections)
- RowParallelLinear: Splits input dimension across GPUs (for O, down projections)
- After ColumnParallel, each GPU has a shard of the output
- After RowParallel, outputs are all-reduced to sync

Transformer Layer Pattern:
    QKV projections  →  ColumnParallel (split heads across GPUs)
    O projection      →  RowParallel (all-reduce output)
    Gate/Up projections →  ColumnParallel (split FFN across GPUs)  
    Down projection    →  RowParallel (all-reduce output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass


class _AllReduceFunc(torch.autograd.Function):
    """All-reduce in forward, identity in backward (inference only)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None):
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=group)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def all_reduce(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """All-reduce tensor across TP group."""
    return _AllReduceFunc.apply(tensor, group)


class _AllGatherFunc(torch.autograd.Function):
    """All-gather along a dimension (inference only)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup] = None):
        if group is None or dist.get_world_size(group) <= 1:
            return input_
        
        world_size = dist.get_world_size(group)
        tensors = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensors, input_, group=group)
        return torch.cat(tensors, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def all_gather(tensor: torch.Tensor, dim: int = -1, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """All-gather tensor along dimension across TP group."""
    return _AllGatherFunc.apply(tensor, dim, group)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with output dimension split across TP ranks.
    
    Full: Y = XW^T + b, where W is [out_features, in_features]
    Shard: Each GPU holds W_i with shape [out_features // tp_size, in_features]
    
    Used for: Q, K, V projections, gate_proj, up_proj
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.gather_output = gather_output
        
        # This rank's shard
        self.out_features_per_rank = out_features // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=torch.float16)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = all_gather(out, dim=-1, group=self.tp_group)
        return out
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ) -> "ColumnParallelLinear":
        """Create from an existing nn.Linear by slicing weights."""
        out_per_rank = linear.out_features // tp_size
        start = tp_rank * out_per_rank
        end = start + out_per_rank
        
        layer = ColumnParallelLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            gather_output=gather_output,
        )
        
        with torch.no_grad():
            layer.weight.copy_(linear.weight.data[start:end])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.data[start:end])
        
        return layer


class RowParallelLinear(nn.Module):
    """
    Linear layer with input dimension split across TP ranks.
    
    Full: Y = XW^T + b, where W is [out_features, in_features]
    Shard: Each GPU holds W_i with shape [out_features, in_features // tp_size]
    Result: All-reduce across ranks to sum partial results.
    
    Used for: O projection, down_proj
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
        reduce_output: bool = True,
    ):
        super().__init__()
        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )
        
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.reduce_output = reduce_output
        
        # This rank's shard
        self.in_features_per_rank = in_features // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=torch.float16)
        )
        if bias:
            # Bias is NOT split — only rank 0 adds it (after all-reduce)
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if self.reduce_output:
            out = all_reduce(out, group=self.tp_group)
        # Add bias after all-reduce (only one copy needed)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        reduce_output: bool = True,
    ) -> "RowParallelLinear":
        """Create from an existing nn.Linear by slicing weights."""
        in_per_rank = linear.in_features // tp_size
        start = tp_rank * in_per_rank
        end = start + in_per_rank
        
        layer = RowParallelLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            reduce_output=reduce_output,
        )
        
        with torch.no_grad():
            layer.weight.copy_(linear.weight.data[:, start:end])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.data)
        
        return layer


class ColumnParallelQuantized(nn.Module):
    """
    Column-parallel wrapper for QuantizedLinearZSE layers.
    
    Slices packed INT4 weights along the output dimension.
    weight_packed: [out_features, in_features//2] → [out_features//tp_size, in_features//2]
    weight_scales: [out_features, num_groups] → [out_features//tp_size, num_groups]
    """
    
    def __init__(
        self,
        original_layer,  # QuantizedLinearZSE
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.gather_output = gather_output
        
        out_per_rank = original_layer.out_features // tp_size
        start = tp_rank * out_per_rank
        end = start + out_per_rank
        
        # Import the actual class to create a new instance
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        self.layer = QuantizedLinearZSE(
            in_features=original_layer.in_features,
            out_features=out_per_rank,
            group_size=original_layer.group_size,
            bias=original_layer.bias is not None,
            backend=original_layer.backend,
        )
        
        # Slice packed weights along output dimension
        if original_layer.weight_packed is not None:
            self.layer.weight_packed = original_layer.weight_packed[start:end].contiguous()
        else:
            # Already converted to triton v2 or bnb — free the zero buffer
            self.layer.weight_packed = None
        if original_layer.weight_scales is not None:
            self.layer.weight_scales = original_layer.weight_scales[start:end].contiguous()
        else:
            self.layer.weight_scales = None
        if original_layer.bias is not None:
            self.layer.bias = original_layer.bias[start:end].contiguous()
        
        # Slice cached/converted formats if they exist
        if original_layer._cached_weight is not None:
            self.layer._cached_weight = original_layer._cached_weight[start:end].contiguous()
        if original_layer._triton_v2_weight is not None:
            # Triton v2 layout: [K//2, N] — slice along N (dim=1)
            self.layer._triton_v2_weight = original_layer._triton_v2_weight[:, start:end].contiguous()
            # Triton v2 scales: [num_groups, N] — slice along N (dim=1)
            if original_layer._triton_v2_scales is not None:
                self.layer._triton_v2_scales = original_layer._triton_v2_scales[:, start:end].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        if self.gather_output:
            out = all_gather(out, dim=-1, group=self.tp_group)
        return out


class RowParallelQuantized(nn.Module):
    """
    Row-parallel wrapper for QuantizedLinearZSE layers.
    
    Slices packed INT4 weights along the input dimension.
    weight_packed: [out_features, in_features//2] → [out_features, in_features//(2*tp_size)]
    weight_scales: Per-group scales need regrouping based on which input groups this rank owns.
    """
    
    def __init__(
        self,
        original_layer,  # QuantizedLinearZSE
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        reduce_output: bool = True,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.reduce_output = reduce_output
        
        in_per_rank = original_layer.in_features // tp_size
        packed_in_per_rank = in_per_rank // 2  # INT4: 2 values per byte
        
        start_packed = tp_rank * packed_in_per_rank
        end_packed = start_packed + packed_in_per_rank
        
        # Figure out which scale groups belong to this rank
        group_size = original_layer.group_size
        groups_per_rank = in_per_rank // group_size
        start_group = tp_rank * groups_per_rank
        end_group = start_group + groups_per_rank
        
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        # Create inner layer WITHOUT bias — we handle bias after all-reduce
        self.layer = QuantizedLinearZSE(
            in_features=in_per_rank,
            out_features=original_layer.out_features,
            group_size=group_size,
            bias=False,
            backend=original_layer.backend,
        )
        
        # Slice packed weights along input dimension (dim=1)
        if original_layer.weight_packed is not None:
            self.layer.weight_packed = original_layer.weight_packed[:, start_packed:end_packed].contiguous()
        else:
            self.layer.weight_packed = None
        if original_layer.weight_scales is not None:
            self.layer.weight_scales = original_layer.weight_scales[:, start_group:end_group].contiguous()
        else:
            self.layer.weight_scales = None
        
        # Bias: stored separately, added after all-reduce
        if original_layer.bias is not None:
            self.register_buffer("_bias", original_layer.bias.clone())
        else:
            self._bias = None
        
        # Slice Triton v2 format if already converted
        if original_layer._triton_v2_weight is not None:
            # Triton v2 layout: [K//2, N] — slice along K//2 (dim=0)
            self.layer._triton_v2_weight = original_layer._triton_v2_weight[
                start_packed:end_packed
            ].contiguous()
            if original_layer._triton_v2_scales is not None:
                # Triton v2 scales: [num_groups, N] — slice along groups (dim=0)
                self.layer._triton_v2_scales = original_layer._triton_v2_scales[
                    start_group:end_group
                ].contiguous()
        
        if original_layer._cached_weight is not None:
            start_in = tp_rank * in_per_rank
            end_in = start_in + in_per_rank
            self.layer._cached_weight = original_layer._cached_weight[:, start_in:end_in].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        if self.reduce_output:
            out = all_reduce(out, group=self.tp_group)
        # Add bias AFTER all-reduce
        if self._bias is not None:
            out = out + self._bias
        return out


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocab split across TP ranks.
    
    Each rank holds a shard of the embedding table.
    Tokens outside this rank's range produce zeros, then all-reduce sums them.
    """
    
    def __init__(
        self,
        original_embedding: nn.Embedding,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        
        num_embeddings = original_embedding.num_embeddings
        embedding_dim = original_embedding.embedding_dim
        
        # Pad vocab to be divisible by tp_size
        self.padded_vocab = ((num_embeddings + tp_size - 1) // tp_size) * tp_size
        self.vocab_per_rank = self.padded_vocab // tp_size
        self.vocab_start = tp_rank * self.vocab_per_rank
        self.vocab_end = min(self.vocab_start + self.vocab_per_rank, num_embeddings)
        self.original_vocab_size = num_embeddings
        
        # Extract this rank's shard
        weight_shard = original_embedding.weight.data[self.vocab_start:self.vocab_end]
        # Pad if needed (last rank may have fewer)
        if weight_shard.shape[0] < self.vocab_per_rank:
            pad = torch.zeros(
                self.vocab_per_rank - weight_shard.shape[0],
                embedding_dim,
                dtype=weight_shard.dtype,
                device=weight_shard.device,
            )
            weight_shard = torch.cat([weight_shard, pad], dim=0)
        
        self.embedding = nn.Embedding(
            self.vocab_per_rank, embedding_dim,
            _weight=weight_shard,
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Shift input IDs to local range
        local_ids = input_ids - self.vocab_start
        # Mask out-of-range tokens
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = local_ids.clamp(0, self.vocab_per_rank - 1)
        
        output = self.embedding(local_ids)
        # Zero out tokens not belonging to this rank
        output = output * mask.unsqueeze(-1).to(output.dtype)
        
        # All-reduce to combine shards
        if self.tp_group is not None:
            output = all_reduce(output, group=self.tp_group)
        return output


# --------------------------------------------------------------------------
# Layer name patterns for different architectures
# --------------------------------------------------------------------------

# Maps architecture → (column_parallel_names, row_parallel_names)
# Column: output dimension split (Q, K, V, gate, up)
# Row: input dimension split (O, down)
PARALLEL_PATTERNS: Dict[str, Tuple[List[str], List[str]]] = {
    "llama": (
        # Column parallel
        ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        # Row parallel
        ["o_proj", "down_proj"],
    ),
    "mistral": (
        ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        ["o_proj", "down_proj"],
    ),
    "qwen2": (
        ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        ["o_proj", "down_proj"],
    ),
    "phi3": (
        ["qkv_proj", "gate_up_proj"],
        ["o_proj", "down_proj"],
    ),
    "gemma2": (
        ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        ["o_proj", "down_proj"],
    ),
}


def _detect_architecture(model: nn.Module) -> str:
    """Detect model architecture from class name."""
    class_name = model.__class__.__name__.lower()
    
    for arch in PARALLEL_PATTERNS:
        if arch in class_name:
            return arch
    
    # Default to llama pattern (most common)
    return "llama"


def _get_layer_name(name: str) -> str:
    """Extract the last component of a module name (e.g., 'q_proj' from 'model.layers.0.self_attn.q_proj')."""
    return name.split(".")[-1]


class TensorParallel:
    """
    Applies tensor parallelism to a transformer model.
    
    Replaces linear layers with column/row parallel versions that
    shard weights across GPUs and communicate via NCCL AllReduce.
    
    Supports both nn.Linear and QuantizedLinearZSE layers.
    
    Usage:
        tp = TensorParallel(tp_size=4, tp_rank=0)
        model = tp.apply(model)  # Replaces layers in-place
    """
    
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
    
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply tensor parallelism to model, replacing layers in-place.
        
        Returns the modified model.
        """
        arch = _detect_architecture(model)
        col_names, row_names = PARALLEL_PATTERNS[arch]
        
        replaced = 0
        
        for name, module in list(model.named_modules()):
            layer_name = _get_layer_name(name)
            
            if layer_name in col_names:
                new_layer = self._make_column_parallel(module)
                if new_layer is not None:
                    _replace_module(model, name, new_layer)
                    replaced += 1
            
            elif layer_name in row_names:
                new_layer = self._make_row_parallel(module)
                if new_layer is not None:
                    _replace_module(model, name, new_layer)
                    replaced += 1
        
        # Handle embedding and lm_head
        self._parallelize_embedding(model)
        self._parallelize_lm_head(model)
        
        # Fix attention head counts for this rank's shard
        self._fix_attention_config(model)
        
        return model
    
    def _make_column_parallel(self, module: nn.Module) -> Optional[nn.Module]:
        """Convert a module to column-parallel."""
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        if isinstance(module, QuantizedLinearZSE):
            if module.out_features % self.tp_size != 0:
                return None
            return ColumnParallelQuantized(
                module, self.tp_size, self.tp_rank, self.tp_group,
            )
        elif isinstance(module, nn.Linear):
            if module.out_features % self.tp_size != 0:
                return None
            return ColumnParallelLinear.from_linear(
                module, self.tp_size, self.tp_rank, self.tp_group,
            )
        return None
    
    def _make_row_parallel(self, module: nn.Module) -> Optional[nn.Module]:
        """Convert a module to row-parallel."""
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        if isinstance(module, QuantizedLinearZSE):
            if module.in_features % self.tp_size != 0:
                return None
            return RowParallelQuantized(
                module, self.tp_size, self.tp_rank, self.tp_group,
            )
        elif isinstance(module, nn.Linear):
            if module.in_features % self.tp_size != 0:
                return None
            return RowParallelLinear.from_linear(
                module, self.tp_size, self.tp_rank, self.tp_group,
            )
        return None
    
    def _parallelize_embedding(self, model: nn.Module):
        """Split embedding table across ranks."""
        # Find the input embedding
        embed = None
        embed_name = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and "embed_tokens" in name:
                embed = module
                embed_name = name
                break
        
        if embed is not None and embed.num_embeddings >= self.tp_size:
            parallel_embed = VocabParallelEmbedding(
                embed, self.tp_size, self.tp_rank, self.tp_group,
            )
            _replace_module(model, embed_name, parallel_embed)
    
    def _parallelize_lm_head(self, model: nn.Module):
        """Split lm_head across ranks (column parallel with gather)."""
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            lm_head = model.lm_head
            if lm_head.out_features % self.tp_size == 0:
                parallel_head = ColumnParallelLinear.from_linear(
                    lm_head, self.tp_size, self.tp_rank, self.tp_group,
                    gather_output=True,  # Need full logits for sampling
                )
                model.lm_head = parallel_head
    
    def _fix_attention_config(self, model: nn.Module):
        """
        Update model config AND attention modules for per-rank head counts.
        
        After TP, each rank only has num_heads // tp_size heads.
        Both the config and each attention module need to reflect this.
        """
        config = getattr(model, "config", None)
        if config is None:
            return
        
        # Track original values for fixing modules
        orig_num_heads = getattr(config, "num_attention_heads", None)
        orig_kv_heads = getattr(config, "num_key_value_heads", None)
        
        new_num_heads = None
        new_kv_heads = None
        
        if orig_num_heads is not None and orig_num_heads % self.tp_size == 0:
            new_num_heads = orig_num_heads // self.tp_size
            config.num_attention_heads = new_num_heads
        
        if orig_kv_heads is not None:
            if orig_kv_heads % self.tp_size == 0:
                new_kv_heads = orig_kv_heads // self.tp_size
                config.num_key_value_heads = new_kv_heads
            elif orig_kv_heads == 1:
                # MQA: keep single KV head, replicate on each rank
                new_kv_heads = 1
        
        # Fix cached head counts in each attention module
        # HuggingFace transformers caches these in __init__
        for name, module in model.named_modules():
            # Match attention modules by common attribute patterns
            if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                if new_num_heads is not None and getattr(module, 'num_heads', None) == orig_num_heads:
                    module.num_heads = new_num_heads
                if new_kv_heads is not None and hasattr(module, 'num_key_value_heads'):
                    if module.num_key_value_heads == orig_kv_heads:
                        module.num_key_value_heads = new_kv_heads
                # Also fix num_key_value_groups (used in GQA repeat)
                if hasattr(module, 'num_key_value_groups') and new_num_heads is not None and new_kv_heads is not None:
                    module.num_key_value_groups = new_num_heads // new_kv_heads

    def get_stats(self) -> Dict[str, Any]:
        """Get TP configuration stats."""
        return {
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "has_nccl_group": self.tp_group is not None,
        }


def _replace_module(model: nn.Module, target_name: str, new_module: nn.Module):
    """Replace a named module in the model."""
    parts = target_name.split(".")
    parent = model
    
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def apply_tensor_parallel(
    model: nn.Module,
    tp_size: int,
    tp_rank: int,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """
    Convenience function to apply tensor parallelism.
    
    Args:
        model: The transformer model
        tp_size: Number of GPU shards
        tp_rank: This GPU's rank (0-indexed)
        tp_group: NCCL process group (None for single-process mode)
    
    Returns:
        Model with parallel layers (modified in-place)
    """
    tp = TensorParallel(tp_size, tp_rank, tp_group)
    return tp.apply(model)
