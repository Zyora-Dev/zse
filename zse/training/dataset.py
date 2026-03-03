"""
ZSE Dataset Utilities

Dataset classes and utilities for fine-tuning:
- ChatDataset: Conversational format (user/assistant turns)
- InstructionDataset: Instruction-following format
- prepare_dataset: Helper to create datasets from various formats

Author: ZSE Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    
    max_length: int = 2048
    """Maximum sequence length."""
    
    truncation: bool = True
    """Whether to truncate sequences longer than max_length."""
    
    padding: str = "max_length"
    """Padding strategy: 'max_length', 'longest', or False."""
    
    mask_prompt: bool = True
    """Whether to mask prompt tokens in labels (only train on response)."""


class ZSEDataset(Dataset):
    """
    Base dataset class for ZSE training.
    
    Handles tokenization and formatting.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        config: Optional[DatasetConfig] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def _tokenize(
        self,
        text: str,
        add_eos: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with configured settings."""
        if add_eos and not text.endswith(self.tokenizer.eos_token):
            text = text + self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors="pt",
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }


class ChatDataset(ZSEDataset):
    """
    Dataset for chat/conversation format.
    
    Expected data format:
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    ...
                ]
            },
            ...
        ]
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item['messages']
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize full conversation
        encoded = self._tokenize(text)
        
        # Create labels
        labels = encoded['input_ids'].clone()
        
        if self.config.mask_prompt:
            # Find where assistant response starts and mask everything before
            # This is approximate - works for most chat formats
            labels = self._mask_prompt_tokens(messages, labels)
        
        # Mask padding tokens
        labels[encoded['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }
    
    def _mask_prompt_tokens(
        self,
        messages: List[Dict],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask prompt tokens in labels (only train on assistant responses)."""
        # Tokenize just the prompt parts
        prompt_messages = []
        for msg in messages:
            if msg['role'] == 'assistant':
                break
            prompt_messages.append(msg)
        
        if prompt_messages:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_len = len(self.tokenizer(prompt_text)['input_ids'])
            labels[:prompt_len] = -100
        
        return labels


class InstructionDataset(ZSEDataset):
    """
    Dataset for instruction-following format.
    
    Expected data format:
        [
            {
                "instruction": "Summarize this text:",
                "input": "The quick brown fox...",  # optional
                "output": "A fox jumps over a dog."
            },
            ...
        ]
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        config: Optional[DatasetConfig] = None,
        prompt_template: Optional[str] = None,
    ):
        super().__init__(data, tokenizer, config)
        
        self.prompt_template = prompt_template or (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )
        
        self.prompt_template_no_input = (
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', item.get('response', ''))
        
        # Format prompt
        if input_text:
            full_text = self.prompt_template.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
            prompt_text = self.prompt_template.format(
                instruction=instruction,
                input=input_text,
                output='',
            )
        else:
            full_text = self.prompt_template_no_input.format(
                instruction=instruction,
                output=output,
            )
            prompt_text = self.prompt_template_no_input.format(
                instruction=instruction,
                output='',
            )
        
        # Tokenize
        encoded = self._tokenize(full_text)
        
        # Create labels
        labels = encoded['input_ids'].clone()
        
        if self.config.mask_prompt:
            prompt_len = len(self.tokenizer(prompt_text)['input_ids'])
            labels[:prompt_len] = -100
        
        # Mask padding
        labels[encoded['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }


class TextDataset(ZSEDataset):
    """
    Simple dataset for raw text (continued pretraining).
    
    Expected data format:
        [
            {"text": "First document..."},
            {"text": "Second document..."},
            ...
        ]
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get('text', item.get('content', ''))
        
        encoded = self._tokenize(text)
        
        # For language modeling, labels = input_ids
        labels = encoded['input_ids'].clone()
        labels[encoded['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }


def prepare_dataset(
    data_source: Union[str, Path, List[Dict]],
    tokenizer: Any,
    dataset_type: str = "auto",
    config: Optional[DatasetConfig] = None,
    split: Optional[str] = None,
) -> ZSEDataset:
    """
    Create a dataset from various sources.
    
    Args:
        data_source: Path to JSON/JSONL file, or list of dicts
        tokenizer: Tokenizer to use
        dataset_type: 'chat', 'instruction', 'text', or 'auto' (detect)
        config: Dataset configuration
        split: For HuggingFace datasets, which split to use
        
    Returns:
        Appropriate ZSEDataset subclass
    """
    # Load data if path
    if isinstance(data_source, (str, Path)):
        data_source = Path(data_source)
        
        if data_source.suffix == '.jsonl':
            data = []
            with open(data_source) as f:
                for line in f:
                    data.append(json.loads(line))
        elif data_source.suffix == '.json':
            with open(data_source) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_source.suffix}")
    else:
        data = data_source
    
    # Auto-detect type
    if dataset_type == "auto":
        sample = data[0] if data else {}
        
        if 'messages' in sample:
            dataset_type = "chat"
        elif 'instruction' in sample:
            dataset_type = "instruction"
        elif 'text' in sample or 'content' in sample:
            dataset_type = "text"
        else:
            raise ValueError(f"Cannot auto-detect dataset type from keys: {sample.keys()}")
    
    # Create dataset
    config = config or DatasetConfig()
    
    if dataset_type == "chat":
        return ChatDataset(data, tokenizer, config)
    elif dataset_type == "instruction":
        return InstructionDataset(data, tokenizer, config)
    elif dataset_type == "text":
        return TextDataset(data, tokenizer, config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_dataset_from_hub(
    dataset_name: str,
    tokenizer: Any,
    split: str = "train",
    dataset_type: str = "auto",
    config: Optional[DatasetConfig] = None,
    max_samples: Optional[int] = None,
) -> ZSEDataset:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        split: Dataset split
        dataset_type: 'chat', 'instruction', 'text', or 'auto'
        config: Dataset configuration
        max_samples: Maximum samples to load
        
    Returns:
        ZSEDataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")
    
    hf_dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
    
    # Convert to list of dicts
    data = [dict(item) for item in hf_dataset]
    
    return prepare_dataset(data, tokenizer, dataset_type, config)
