"""
HuggingFace Model Loader

Load models from HuggingFace Hub:
- Auto-download models
- Authentication support
- Cache management
- Revision/branch support
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any
import logging

import torch

try:
    from huggingface_hub import (
        hf_hub_download,
        snapshot_download,
        HfFileSystem,
        login,
        whoami,
    )
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .base import BaseModelLoader, LoadConfig, ModelInfo
from .safetensors_loader import SafetensorsLoader

logger = logging.getLogger(__name__)


class HuggingFaceLoader(BaseModelLoader):
    """
    Load models from HuggingFace Hub.
    
    Downloads models and then delegates to SafetensorsLoader.
    Supports private models with authentication.
    """
    
    def __init__(
        self,
        config: Optional[LoadConfig] = None,
        token: Optional[str] = None,
    ):
        super().__init__(config)
        self.token = token or os.environ.get("HF_TOKEN")
        self._local_loader = SafetensorsLoader(config)
        self._cache_dir = config.cache_dir if config else None
    
    @staticmethod
    def login(token: Optional[str] = None):
        """Login to HuggingFace Hub."""
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        
        if token:
            login(token=token)
        else:
            login()
    
    @staticmethod
    def check_auth() -> bool:
        """Check if authenticated to HuggingFace Hub."""
        if not HF_HUB_AVAILABLE:
            return False
        try:
            whoami()
            return True
        except Exception:
            return False
    
    def _is_local_path(self, model_path: str) -> bool:
        """Check if path is local or HuggingFace repo ID."""
        path = Path(model_path)
        return path.exists() or "/" not in model_path
    
    def _download_model(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Download model from HuggingFace Hub.
        
        Returns local path to downloaded model.
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required")
        
        # Default patterns for model files
        if allow_patterns is None:
            allow_patterns = [
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",  # sentencepiece
                "*.tiktoken",
            ]
        
        logger.info(f"Downloading {repo_id} from HuggingFace Hub...")
        
        local_dir = snapshot_download(
            repo_id,
            revision=revision,
            token=self.token,
            cache_dir=self._cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.bin", "*.h5", "*.ot"],  # Skip PyTorch/TF/ONNX
        )
        
        logger.info(f"Model downloaded to: {local_dir}")
        return local_dir
    
    def _get_model_files(self, repo_id: str) -> List[str]:
        """List files in HuggingFace repo."""
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required")
        
        fs = HfFileSystem(token=self.token)
        files = fs.ls(repo_id, detail=False)
        return [f.split("/")[-1] for f in files]
    
    def load_model_info(self, model_path: str) -> ModelInfo:
        """
        Load model info from HuggingFace Hub or local path.
        
        For Hub models, downloads only config.json first.
        """
        if self._is_local_path(model_path):
            return self._local_loader.load_model_info(model_path)
        
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required")
        
        # Download only config first
        config_path = hf_hub_download(
            model_path,
            filename="config.json",
            token=self.token,
            cache_dir=self._cache_dir,
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        info = ModelInfo.from_config(config, name=model_path)
        info.config_file = config_path
        
        # Get weight files list without downloading
        files = self._get_model_files(model_path)
        info.weight_files = [f for f in files if f.endswith(".safetensors")]
        
        self._model_info = info
        return info
    
    def load_weights(
        self,
        model_path: str,
        model: torch.nn.Module,
        progress_callback=None,
    ) -> torch.nn.Module:
        """
        Load weights from HuggingFace Hub or local path.
        
        Downloads model if necessary, then loads via SafetensorsLoader.
        """
        if self._is_local_path(model_path):
            local_path = model_path
        else:
            local_path = self._download_model(model_path)
        
        return self._local_loader.load_weights(
            local_path,
            model,
            progress_callback,
        )
    
    def iterate_weights(
        self,
        model_path: str,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over weights, downloading if necessary."""
        if self._is_local_path(model_path):
            local_path = model_path
        else:
            local_path = self._download_model(model_path)
        
        yield from self._local_loader.iterate_weights(local_path)
    
    def download(
        self,
        repo_id: str,
        revision: Optional[str] = None,
    ) -> str:
        """
        Download model without loading.
        
        Useful for pre-caching models.
        """
        return self._download_model(repo_id, revision)


class ModelHub:
    """
    Unified interface for loading models from any source.
    
    Automatically detects source type and uses appropriate loader.
    """
    
    def __init__(self, config: Optional[LoadConfig] = None):
        self.config = config or LoadConfig()
        self._hf_loader = HuggingFaceLoader(config)
        self._st_loader = SafetensorsLoader(config)
    
    def load_info(self, model_path: str) -> ModelInfo:
        """Load model info from any source."""
        if self._is_hf_repo(model_path):
            return self._hf_loader.load_model_info(model_path)
        else:
            return self._st_loader.load_model_info(model_path)
    
    def load(
        self,
        model_path: str,
        model: torch.nn.Module,
        progress_callback=None,
    ) -> torch.nn.Module:
        """Load model from any source."""
        if self._is_hf_repo(model_path):
            return self._hf_loader.load_weights(model_path, model, progress_callback)
        else:
            return self._st_loader.load_weights(model_path, model, progress_callback)
    
    def _is_hf_repo(self, path: str) -> bool:
        """Check if path is a HuggingFace repo ID."""
        # Local paths
        if Path(path).exists():
            return False
        # Paths starting with ./ or ../ are local
        if path.startswith("./") or path.startswith("../"):
            return False
        # Absolute paths are local
        if path.startswith("/"):
            return False
        # HF repos have format org/model
        # Must contain exactly one / and not be a path-like pattern
        return "/" in path and path.count("/") == 1
