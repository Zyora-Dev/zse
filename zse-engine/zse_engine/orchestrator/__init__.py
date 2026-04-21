"""ZSE Orchestrator — Inference engine that ties everything together.

Components:
    - VRAMAllocator: Unified GPU memory budget
    - WeightLoader: .zse → GPU weight transfer
    - InferenceKernels: All LLM GPU kernels
    - ModelRunner: Transformer forward pass
    - Sampler: Token sampling (greedy, top-p, top-k)
    - ZSEEngine: Public API
"""

def __getattr__(name):
    if name == "ZSEEngine":
        from zse_engine.orchestrator.engine import ZSEEngine
        return ZSEEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ZSEEngine"]
