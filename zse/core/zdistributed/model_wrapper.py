"""
Tensor Parallel Model Wrapper

Makes TPCoordinator look like a standard nn.Module / HuggingFace model
so it plugs transparently into TextGenerator and the generation pipeline.

The existing code does:
    output = model(input_ids=..., past_key_values=..., use_cache=True)
    output.logits  # [batch, seq, vocab]
    output.past_key_values  # KV cache

    output_ids = model.generate(input_ids, max_new_tokens=128, ...)

This wrapper provides exactly that interface, delegating to TP workers.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class TPModelOutput:
    """Mimics transformers CausalLMOutput."""

    logits: torch.Tensor
    past_key_values: Any = None


class TPModelWrapper(nn.Module):
    """
    Wraps TPCoordinator to provide nn.Module-compatible interface.

    The real model runs across multiple GPU processes.
    This wrapper serializes forward() calls and routes them.
    """

    def __init__(self, coordinator, config=None):
        super().__init__()
        self.coordinator = coordinator
        self.config = config
        self._device = torch.device("cpu")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        **kwargs,
    ) -> TPModelOutput:
        """Forward pass delegated to TP workers."""
        fwd_kwargs = {"use_cache": use_cache}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask.cpu()
        if past_key_values is not None:
            fwd_kwargs["past_key_values"] = past_key_values

        result = self.coordinator.forward(input_ids, **fwd_kwargs)

        return TPModelOutput(
            logits=result["logits"],
            past_key_values=result.get("past_key_values"),
        )

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generation delegated to TP workers (uses HF generate internally)."""
        return self.coordinator.generate(input_ids, **kwargs)

    def parameters(self):
        """No local parameters — model is distributed."""
        return iter([])

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        return self._device


# PPModelWrapper is the same as TPModelWrapper — both coordinators
# share the same interface (forward, generate, shutdown)
PPModelWrapper = TPModelWrapper
