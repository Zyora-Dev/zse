"""
ZSE Model Discovery

Discover compatible models from HuggingFace Hub.
Search, filter, and get information about models.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# Supported architectures for ZSE
SUPPORTED_ARCHITECTURES = [
    "LlamaForCausalLM",
    "MistralForCausalLM", 
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "Gemma2ForCausalLM",
    "GemmaForCausalLM",
    "GPTNeoXForCausalLM",
    "FalconForCausalLM",
    "StableLmForCausalLM",
    "StarCoder2ForCausalLM",
]


@dataclass
class HFModelInfo:
    """Model information from HuggingFace Hub."""
    model_id: str
    author: str
    name: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    architecture: Optional[str]
    library_name: Optional[str]
    license: Optional[str]
    created_at: Optional[str]
    last_modified: Optional[str]
    is_compatible: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "author": self.author,
            "name": self.name,
            "downloads": self.downloads,
            "likes": self.likes,
            "pipeline_tag": self.pipeline_tag,
            "tags": self.tags,
            "architecture": self.architecture,
            "library_name": self.library_name,
            "license": self.license,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "is_compatible": self.is_compatible,
        }


class ModelDiscovery:
    """Discover models from HuggingFace Hub."""
    
    HF_API_URL = "https://huggingface.co/api"
    
    def __init__(self, token: Optional[str] = None):
        """Initialize discovery with optional HF token."""
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._client = None
    
    def _get_client(self):
        """Get or create HTTP client."""
        if not HAS_HTTPX:
            raise ImportError("httpx is required for model discovery. Install with: pip install httpx")
        if self._client is None:
            headers = {"Accept": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(headers=headers, timeout=30.0)
        return self._client
    
    def search(
        self,
        query: str = "",
        author: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        library: str = "transformers",
        pipeline_tag: str = "text-generation",
        sort: str = "downloads",
        direction: str = "-1",  # Descending
        limit: int = 20,
        only_compatible: bool = True,
    ) -> List[HFModelInfo]:
        """
        Search for models on HuggingFace Hub.
        
        Args:
            query: Search query string
            author: Filter by author/organization
            filter_tags: Additional tag filters
            library: Library filter (default: transformers)
            pipeline_tag: Pipeline type (default: text-generation)
            sort: Sort field (downloads, likes, created_at, lastModified)
            direction: Sort direction (-1 for descending)
            limit: Maximum results to return
            only_compatible: Only return ZSE-compatible models
            
        Returns:
            List of HFModelInfo objects
        """
        client = self._get_client()
        
        # Build query parameters
        params = {
            "limit": min(limit * 2 if only_compatible else limit, 100),  # Get more to filter
            "sort": sort,
            "direction": direction,
            "full": "true",  # Get full model info
        }
        
        if query:
            params["search"] = query
        if author:
            params["author"] = author
        if library:
            params["library"] = library
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag
        if filter_tags:
            params["filter"] = ",".join(filter_tags)
        
        # Make request
        try:
            response = client.get(f"{self.HF_API_URL}/models", params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Warning: HuggingFace API error: {e}")
            return []
        
        # Parse results
        results = []
        for item in data:
            model_info = self._parse_model_info(item)
            
            # Filter by compatibility if requested
            if only_compatible and not model_info.is_compatible:
                continue
                
            results.append(model_info)
            
            if len(results) >= limit:
                break
        
        return results
    
    def _parse_model_info(self, data: Dict[str, Any]) -> HFModelInfo:
        """Parse model info from API response."""
        model_id = data.get("modelId", data.get("id", ""))
        
        # Extract architecture from config or tags
        architecture = None
        config = data.get("config", {})
        if config:
            if isinstance(config, dict):
                architecture = config.get("model_type") or config.get("architectures", [None])[0] if config.get("architectures") else None
        
        # Check tags for architecture hints
        tags = data.get("tags", [])
        for tag in tags:
            if tag in SUPPORTED_ARCHITECTURES:
                architecture = tag
                break
        
        # Determine compatibility
        is_compatible = architecture in SUPPORTED_ARCHITECTURES if architecture else False
        
        # Also check by model name patterns
        if not is_compatible:
            model_lower = model_id.lower()
            if any(pattern in model_lower for pattern in ["llama", "mistral", "mixtral", "qwen", "phi-", "gemma", "falcon", "starcoder"]):
                is_compatible = True
        
        # Split model_id into author/name
        parts = model_id.split("/", 1)
        author = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        
        return HFModelInfo(
            model_id=model_id,
            author=author,
            name=name,
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            pipeline_tag=data.get("pipeline_tag"),
            tags=tags,
            architecture=architecture,
            library_name=data.get("library_name"),
            license=data.get("license") or (tags[0] if "license:" in str(tags) else None),
            created_at=data.get("createdAt"),
            last_modified=data.get("lastModified"),
            is_compatible=is_compatible,
        )
    
    def get_model_info(self, model_id: str) -> Optional[HFModelInfo]:
        """Get detailed info for a specific model."""
        client = self._get_client()
        
        try:
            response = client.get(f"{self.HF_API_URL}/models/{model_id}")
            response.raise_for_status()
            data = response.json()
            return self._parse_model_info(data)
        except Exception as e:
            print(f"Warning: Could not fetch model info for {model_id}: {e}")
            return None
    
    def get_model_files(self, model_id: str) -> List[Dict[str, Any]]:
        """Get list of files in a model repository."""
        client = self._get_client()
        
        try:
            response = client.get(f"{self.HF_API_URL}/models/{model_id}/tree/main")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Could not fetch files for {model_id}: {e}")
            return []
    
    def estimate_model_size(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Estimate model size based on files."""
        files = self.get_model_files(model_id)
        
        total_size = 0
        safetensors_size = 0
        pytorch_size = 0
        
        for f in files:
            if f.get("type") != "file":
                continue
            size = f.get("size", 0)
            path = f.get("path", "")
            
            total_size += size
            if path.endswith(".safetensors"):
                safetensors_size += size
            elif path.endswith(".bin"):
                pytorch_size += size
        
        # Prefer safetensors, fall back to pytorch
        model_size = safetensors_size if safetensors_size > 0 else pytorch_size
        
        # Estimate parameters (rough: 2 bytes per param for fp16)
        estimated_params = model_size / 2 / 1e9  # Billions
        
        # Estimate VRAM needs
        vram_fp16 = model_size / 1e9 * 1.2  # Add 20% overhead
        vram_int8 = vram_fp16 / 2 * 1.1
        vram_int4 = vram_fp16 / 4 * 1.1
        
        return {
            "total_size_bytes": total_size,
            "model_size_bytes": model_size,
            "estimated_params_b": round(estimated_params, 2),
            "estimated_vram_fp16_gb": round(vram_fp16, 1),
            "estimated_vram_int8_gb": round(vram_int8, 1),
            "estimated_vram_int4_gb": round(vram_int4, 1),
            "has_safetensors": safetensors_size > 0,
            "has_pytorch": pytorch_size > 0,
        }
    
    def check_compatibility(self, model_id: str) -> Dict[str, Any]:
        """
        Check if a model is compatible with ZSE.
        
        Returns detailed compatibility information.
        """
        info = self.get_model_info(model_id)
        files = self.get_model_files(model_id)
        
        result = {
            "model_id": model_id,
            "compatible": False,
            "architecture": None,
            "issues": [],
            "recommendations": [],
        }
        
        if not info:
            result["issues"].append("Could not fetch model information")
            return result
        
        result["architecture"] = info.architecture
        
        # Check architecture
        if info.architecture in SUPPORTED_ARCHITECTURES:
            result["compatible"] = True
        elif info.is_compatible:
            result["compatible"] = True
            result["issues"].append(f"Architecture '{info.architecture}' not explicitly listed but may work")
        else:
            result["issues"].append(f"Architecture '{info.architecture}' is not supported")
        
        # Check for safetensors
        has_safetensors = any(f.get("path", "").endswith(".safetensors") for f in files if f.get("type") == "file")
        if not has_safetensors:
            result["issues"].append("No safetensors files found (slower loading)")
            result["recommendations"].append("Models with safetensors load faster")
        
        # Check for config.json
        has_config = any(f.get("path") == "config.json" for f in files if f.get("type") == "file")
        if not has_config:
            result["issues"].append("No config.json found")
            result["compatible"] = False
        
        # Check for tokenizer
        has_tokenizer = any(
            f.get("path") in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
            for f in files if f.get("type") == "file"
        )
        if not has_tokenizer:
            result["issues"].append("No tokenizer files found")
            result["compatible"] = False
        
        # Add size estimates
        size_info = self.estimate_model_size(model_id)
        if size_info:
            result["size_info"] = size_info
            
            if size_info["estimated_vram_int8_gb"] > 80:
                result["recommendations"].append("Large model - consider INT4 quantization")
            elif size_info["estimated_vram_int8_gb"] > 24:
                result["recommendations"].append("Consider INT8 quantization for consumer GPUs")
        
        return result
    
    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


# Convenience functions
_discovery: Optional[ModelDiscovery] = None


def get_discovery() -> ModelDiscovery:
    """Get the global discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = ModelDiscovery()
    return _discovery


def search_models(query: str, limit: int = 10, **kwargs) -> List[HFModelInfo]:
    """Search for compatible models on HuggingFace."""
    return get_discovery().search(query=query, limit=limit, **kwargs)


def check_model(model_id: str) -> Dict[str, Any]:
    """Check if a model is compatible with ZSE."""
    return get_discovery().check_compatibility(model_id)
