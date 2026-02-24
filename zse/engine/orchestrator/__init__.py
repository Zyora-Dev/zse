"""
zOrchestrator - Intelligence Orchestrator

The key differentiator of ZSE: Smart memory management and recommendations.

Components:
- HardwareProfiler: Detect GPUs, CPU, available memory
- MemoryAdvisor: Estimate memory needs, recommend settings
- PrecisionRecommender: Suggest optimal quantization per layer
- AutoConfig: "Fit in X GB" auto-configuration
- IntelligenceOrchestrator: Main orchestrator for loading models

Key Innovation:
- Uses FREE memory, not total memory
- Provides recommendations BEFORE loading
- Allows sharing GPU with other applications
- Auto-optimizes for target VRAM constraints

Memory Targets for 7B models:
- INT4: ~3.5 GB (minimum memory, slower inference)
- INT8: ~7.5 GB (balanced)
- FP16: ~14 GB (maximum speed, full precision)

Usage:
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    # Auto-detect best config
    orchestrator = IntelligenceOrchestrator.auto("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    # Or specify preference
    orchestrator = IntelligenceOrchestrator.for_vram(4.0, "model_name")
    orchestrator = IntelligenceOrchestrator.min_memory("model_name")
    orchestrator = IntelligenceOrchestrator.max_speed("model_name")
"""

from zse.engine.orchestrator.core import (
    OptimizationMode,
    ModelConfig,
    InferenceStats,
    IntelligenceOrchestrator,
    load_model,
    estimate_requirements,
)

__all__ = [
    # Core orchestrator
    "OptimizationMode",
    "ModelConfig",
    "InferenceStats",
    "IntelligenceOrchestrator",
    "load_model",
    "estimate_requirements",
    # Lazy imports for future components
    "HardwareProfiler",
    "MemoryAdvisor",
    "PrecisionRecommender",
    "AutoConfig",
]
