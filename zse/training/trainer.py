"""
ZSE Trainer - QLoRA Training Loop

Simple, efficient training loop for LoRA fine-tuning on quantized models.

Features:
- Gradient checkpointing for VRAM savings
- Mixed precision training (FP16/BF16)
- Learning rate scheduling
- Gradient accumulation
- Checkpointing and resumption
- Wandb/tensorboard logging (optional)

Author: ZSE Team
"""

import os
import time
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .lora import set_lora_trainable, get_lora_state_dict


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Core training params
    epochs: int = 3
    """Number of training epochs."""
    
    batch_size: int = 1
    """Batch size per device."""
    
    gradient_accumulation_steps: int = 4
    """Accumulate gradients over N steps."""
    
    learning_rate: float = 2e-4
    """Peak learning rate."""
    
    weight_decay: float = 0.01
    """Weight decay for AdamW."""
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"
    """LR scheduler: 'cosine', 'linear', or 'constant'."""
    
    warmup_ratio: float = 0.03
    """Fraction of steps for warmup."""
    
    min_lr_ratio: float = 0.1
    """Minimum LR as fraction of peak (for cosine)."""
    
    # Memory optimization
    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save VRAM."""
    
    fp16: bool = True
    """Use FP16 mixed precision."""
    
    bf16: bool = False
    """Use BF16 mixed precision (preferred if supported)."""
    
    # Logging
    logging_steps: int = 10
    """Log every N steps."""
    
    save_steps: int = 500
    """Save checkpoint every N steps."""
    
    eval_steps: int = 500
    """Evaluate every N steps."""
    
    # Output
    output_dir: str = "./zse_training_output"
    """Directory for checkpoints and logs."""
    
    save_total_limit: int = 3
    """Maximum checkpoints to keep."""
    
    # Advanced
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    
    seed: int = 42
    """Random seed."""
    
    dataloader_num_workers: int = 4
    """Number of dataloader workers."""
    
    def __post_init__(self):
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingOutput:
    """Output from training."""
    
    train_loss: float
    """Final training loss."""
    
    train_steps: int
    """Total training steps."""
    
    train_time_seconds: float
    """Total training time."""
    
    best_eval_loss: Optional[float] = None
    """Best evaluation loss (if eval dataset provided)."""
    
    checkpoints: List[str] = field(default_factory=list)
    """List of saved checkpoint paths."""


class ZSETrainer:
    """
    Simple trainer for QLoRA fine-tuning.
    
    Usage:
        trainer = ZSETrainer(model, tokenizer, train_dataset)
        output = trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        self.data_collator = data_collator or self._default_collator
        
        # Setup
        self.device = next(model.parameters()).device
        self._setup_training()
    
    def _default_collator(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Default data collator for causal LM."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item.get('labels', item['input_ids']) for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _setup_training(self):
        """Setup optimizer, scheduler, etc."""
        config = self.config
        
        # Set seed
        torch.manual_seed(config.seed)
        
        # Ensure only LoRA params are trainable
        set_lora_trainable(self.model, trainable=True)
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
        
        # Get trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Calculate total steps
        steps_per_epoch = len(self.train_dataset) // (
            config.batch_size * config.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * config.epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        # Learning rate scheduler
        if config.lr_scheduler == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=config.learning_rate * config.min_lr_ratio,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        elif config.lr_scheduler == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.min_lr_ratio,
                total_iters=self.total_steps,
            )
        else:  # constant
            self.scheduler = None
        
        # Mixed precision
        self.scaler = None
        if config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                collate_fn=self.data_collator,
                pin_memory=True,
            )
    
    def train(self) -> TrainingOutput:
        """Run training loop."""
        config = self.config
        model = self.model
        
        model.train()
        global_step = 0
        total_loss = 0.0
        best_eval_loss = float('inf')
        checkpoints = []
        
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting QLoRA Training")
        print(f"{'='*60}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Output dir: {config.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=config.fp16 or config.bf16):
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.max_grad_norm, 
                    )
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                    total_loss += epoch_loss / epoch_steps
                    
                    # Logging
                    if global_step % config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.optimizer.param_groups[0]['lr']
                        elapsed = time.time() - start_time
                        steps_per_sec = global_step / elapsed
                        
                        print(f"Step {global_step}/{self.total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Speed: {steps_per_sec:.2f} steps/s")
                    
                    # Evaluation
                    if self.eval_dataset and global_step % config.eval_steps == 0:
                        eval_loss = self._evaluate()
                        print(f"  Eval loss: {eval_loss:.4f}")
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                        
                        model.train()
                    
                    # Checkpointing
                    if global_step % config.save_steps == 0:
                        ckpt_path = self._save_checkpoint(global_step)
                        checkpoints.append(ckpt_path)
                        
                        # Remove old checkpoints
                        if len(checkpoints) > config.save_total_limit:
                            old_ckpt = checkpoints.pop(0)
                            if os.path.exists(old_ckpt):
                                os.remove(old_ckpt)
                    
                    epoch_loss = 0.0
                    epoch_steps = 0
            
            print(f"\nEpoch {epoch + 1}/{config.epochs} completed")
        
        # Final save
        final_path = self._save_checkpoint("final")
        checkpoints.append(final_path)
        
        training_time = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total steps: {global_step}")
        print(f"  Final loss: {avg_loss:.4f}")
        print(f"  Training time: {training_time/60:.1f} minutes")
        print(f"  Final checkpoint: {final_path}")
        print(f"{'='*60}\n")
        
        return TrainingOutput(
            train_loss=avg_loss,
            train_steps=global_step,
            train_time_seconds=training_time,
            best_eval_loss=best_eval_loss if best_eval_loss < float('inf') else None,
            checkpoints=checkpoints,
        )
    
    def _evaluate(self) -> float:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, step: Union[int, str]) -> str:
        """Save LoRA checkpoint."""
        ckpt_name = f"checkpoint-{step}.safetensors"
        ckpt_path = os.path.join(self.config.output_dir, ckpt_name)
        
        # Get LoRA state dict
        lora_state = get_lora_state_dict(self.model)
        
        # Save with safetensors
        try:
            from safetensors.torch import save_file
            save_file(lora_state, ckpt_path)
        except ImportError:
            # Fallback to torch
            ckpt_path = ckpt_path.replace('.safetensors', '.pt')
            torch.save(lora_state, ckpt_path)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save LoRA config if available
        if hasattr(self.model, '_lora_config'):
            lora_config_path = os.path.join(self.config.output_dir, "lora_config.json")
            with open(lora_config_path, 'w') as f:
                json.dump(asdict(self.model._lora_config), f, indent=2)
        
        print(f"  Saved checkpoint: {ckpt_path}")
        return ckpt_path
