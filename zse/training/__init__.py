"""
ZSE Training Module - QLoRA Fine-tuning for .zse Models

Fine-tune large language models with minimal VRAM using QLoRA:
- Base model stays INT4 quantized (frozen)
- Only LoRA adapters are trained (FP16)
- 32B model training in ~24GB VRAM

Example:
    from zse.format.reader_v2 import load_zse_model
    from zse.training import LoRAConfig, ZSETrainer
    
    # Load base model
    model, tokenizer, info = load_zse_model("qwen32b.zse")
    
    # Add LoRA adapters
    lora_config = LoRAConfig(rank=64, alpha=128)
    model = add_lora_to_model(model, lora_config)
    
    # Train
    trainer = ZSETrainer(model, tokenizer, train_dataset)
    trainer.train(epochs=3)
    
    # Save adapter
    save_lora_adapter(model, "my_adapter.safetensors")
    
    # Later: Load with adapter
    model, tok, _ = load_zse_model("qwen32b.zse", lora="my_adapter.safetensors")

Author: ZSE Team
"""

from .lora import (
    LoRAConfig,
    add_lora_to_model,
    get_lora_state_dict,
    set_lora_trainable,
)

from .adapter_io import (
    save_lora_adapter,
    load_lora_adapter,
)

from .trainer import (
    ZSETrainer,
    TrainingConfig,
    TrainingOutput,
)

from .dataset import (
    ZSEDataset,
    ChatDataset,
    InstructionDataset,
    TextDataset,
    DatasetConfig,
    prepare_dataset,
    load_dataset_from_hub,
)


__all__ = [
    # LoRA
    "LoRAConfig",
    "add_lora_to_model",
    "get_lora_state_dict",
    "set_lora_trainable",
    
    # Adapter I/O
    "save_lora_adapter",
    "load_lora_adapter",
    
    # Trainer
    "ZSETrainer",
    "TrainingConfig",
    "TrainingOutput",
    
    # Dataset
    "ZSEDataset",
    "ChatDataset",
    "InstructionDataset",
    "TextDataset",
    "DatasetConfig",
    "prepare_dataset",
    "load_dataset_from_hub",
]
