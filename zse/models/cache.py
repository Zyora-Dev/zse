"""
ZSE Model Cache Manager

Manages local model cache at ~/.zse/models/
Handles downloading, converting, and resolving models.
"""

from __future__ import annotations

import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".zse" / "models"
CONFIG_DIR = Path.home() / ".zse"
MANIFEST_FILE = "manifest.json"

# HuggingFace org for pre-converted .zse models
ZSE_HF_ORG = "zse-zllm"


@dataclass
class CachedModel:
    """Metadata for a cached model."""
    alias: str                    # Short name (e.g., "qwen-7b")
    hf_model_id: str             # HuggingFace repo ID
    zse_path: str                # Path to .zse file
    quantization: str            # int4, int8, fp16
    file_size_gb: float          # File size in GB
    cached_at: str               # ISO timestamp
    source: str                  # "pre-converted" or "local-convert"


class ModelCache:
    """Manages the local ZSE model cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.cache_dir / MANIFEST_FILE
        self._manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load cache manifest."""
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"models": {}}

    def _save_manifest(self) -> None:
        """Save cache manifest."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def get_cached(self, alias: str) -> Optional[CachedModel]:
        """Get a cached model by alias or HF model ID."""
        data = self._manifest["models"].get(alias)
        if data and Path(data["zse_path"]).exists():
            return CachedModel(**data)

        # Also try matching by HF model ID
        for key, data in self._manifest["models"].items():
            if data["hf_model_id"] == alias and Path(data["zse_path"]).exists():
                return CachedModel(**data)

        return None

    def resolve(self, model_ref: str) -> Optional[str]:
        """
        Resolve a model reference to a .zse file path.
        Checks: alias → HF ID → direct path.
        Returns path string or None.
        """
        # 1. Direct .zse file path
        if model_ref.endswith(".zse") and Path(model_ref).exists():
            return model_ref

        # 2. Check cache by alias
        cached = self.get_cached(model_ref)
        if cached:
            return cached.zse_path

        return None

    def add(self, alias: str, hf_model_id: str, zse_path: Path,
            quantization: str = "int4", source: str = "local-convert") -> CachedModel:
        """Register a model in the cache."""
        size_gb = zse_path.stat().st_size / (1024 ** 3)
        model = CachedModel(
            alias=alias,
            hf_model_id=hf_model_id,
            zse_path=str(zse_path),
            quantization=quantization,
            file_size_gb=round(size_gb, 2),
            cached_at=datetime.now().isoformat(),
            source=source,
        )
        self._manifest["models"][alias] = asdict(model)
        self._save_manifest()
        return model

    def remove(self, alias: str) -> bool:
        """Remove a model from cache (deletes file and manifest entry)."""
        data = self._manifest["models"].get(alias)
        if not data:
            return False

        zse_path = Path(data["zse_path"])
        if zse_path.exists():
            zse_path.unlink()

        del self._manifest["models"][alias]
        self._save_manifest()
        return True

    def list_cached(self) -> List[CachedModel]:
        """List all cached models."""
        result = []
        for alias, data in list(self._manifest["models"].items()):
            if Path(data["zse_path"]).exists():
                result.append(CachedModel(**data))
            else:
                # Clean up stale entries
                del self._manifest["models"][alias]
                self._save_manifest()
        return result

    def cache_size_gb(self) -> float:
        """Total cache size in GB."""
        return sum(m.file_size_gb for m in self.list_cached())

    def get_zse_filename(self, alias: str, quantization: str = "int4") -> str:
        """Generate a .zse filename for an alias."""
        safe_alias = alias.replace("/", "--")
        return f"{safe_alias}-{quantization}.zse"


def make_alias(model_id: str) -> str:
    """
    Create a short alias from a HuggingFace model ID.
    "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
    "meta-llama/Llama-3.1-8B-Instruct" → "llama-3.1-8b-instruct"
    """
    # Take the model name part (after /)
    if "/" in model_id:
        name = model_id.split("/", 1)[1]
    else:
        name = model_id
    return name.lower()


def resolve_model_to_hf_id(model_ref: str) -> Optional[str]:
    """
    Resolve a model reference to a HuggingFace model ID.
    Checks registry aliases first, then treats as direct HF ID.
    """
    from zse.models.registry import get_registry

    registry = get_registry()

    # 1. Exact match in registry
    spec = registry.get(model_ref)
    if spec:
        return spec.model_id

    # 2. Search registry by name/tags
    results = registry.search(model_ref)
    if len(results) == 1:
        return results[0].model_id

    # 3. If it looks like a HF ID (has /), use directly
    if "/" in model_ref:
        return model_ref

    # 4. Try common short aliases
    alias_map = _build_alias_map()
    if model_ref.lower() in alias_map:
        return alias_map[model_ref.lower()]

    return None


def _build_alias_map() -> Dict[str, str]:
    """Build short alias → HF model ID mapping from registry."""
    from zse.models.registry import get_registry

    registry = get_registry()
    aliases: Dict[str, str] = {}

    for model in registry.list_all():
        # Full alias: "qwen2.5-7b-instruct"
        full = make_alias(model.model_id)
        aliases[full] = model.model_id

        # Short aliases based on parameters
        param = model.parameters.lower().replace(" ", "")
        provider = model.provider.lower().replace(" ", "")

        # "qwen-7b", "llama-8b", "mistral-7b"
        if provider in ("qwen", "meta", "mistral ai", "google", "microsoft", "deepseek", "tinyllama"):
            short_provider = {
                "qwen": "qwen",
                "meta": "llama",
                "mistral ai": "mistral",
                "google": "gemma",
                "microsoft": "phi",
                "deepseek": "deepseek",
                "tinyllama": "tinyllama",
            }.get(provider, provider)

            short = f"{short_provider}-{param}"
            if short not in aliases:
                aliases[short] = model.model_id

    return aliases


def get_hf_zse_repo(model_id: str, quantization: str = "int4") -> str:
    """
    Get the HuggingFace repo ID for pre-converted .zse model.
    "Qwen/Qwen2.5-7B-Instruct" → "zyora/Qwen2.5-7B-Instruct-zse-int4"
    """
    if "/" in model_id:
        name = model_id.split("/", 1)[1]
    else:
        name = model_id
    return f"{ZSE_HF_ORG}/{name}-zse-{quantization}"


def check_hf_repo_exists(repo_id: str) -> bool:
    """Check if a HuggingFace repo exists (without downloading)."""
    try:
        from huggingface_hub import repo_info
        repo_info(repo_id)
        return True
    except Exception:
        return False


def get_hf_token() -> Optional[str]:
    """Get stored HuggingFace token."""
    token_file = CONFIG_DIR / "hf_token"
    if token_file.exists():
        return token_file.read_text().strip()

    # Also check HF's own token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def save_hf_token(token: str) -> None:
    """Save HuggingFace token."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    token_file = CONFIG_DIR / "hf_token"
    token_file.write_text(token)
    token_file.chmod(0o600)


def remove_hf_token() -> bool:
    """Remove stored HuggingFace token."""
    token_file = CONFIG_DIR / "hf_token"
    if token_file.exists():
        token_file.unlink()
        return True
    return False
