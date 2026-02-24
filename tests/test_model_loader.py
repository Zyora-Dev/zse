"""
Tests for ZSE Model Loading

Tests:
- Model configuration parsing
- Memory estimation
- Model architecture forward pass
- Weight loading from safetensors
- Multi-GPU device mapping
"""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path

# Import model components
from zse.models import (
    LoadConfig,
    ModelInfo,
    QuantizationType,
    LlamaModel,
    LlamaConfig,
    MistralModel,
    MistralConfig,
    ModelHub,
)
from zse.models.loader.base import BaseModelLoader
from zse.models.architectures.base import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


class TestModelConfig:
    """Test model configuration parsing."""
    
    def test_llama_config_defaults(self):
        """Test default Llama config values."""
        config = LlamaConfig()
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.head_dim == 128
    
    def test_llama_7b_preset(self):
        """Test Llama 7B preset configuration."""
        config = LlamaConfig.llama_7b()
        assert config.hidden_size == 4096
        assert config.intermediate_size == 11008
        assert config.num_key_value_heads == 32  # No GQA in Llama 7B
        assert config.vocab_size == 32000
    
    def test_llama_70b_preset(self):
        """Test Llama 70B preset with GQA."""
        config = LlamaConfig.llama_70b()
        assert config.hidden_size == 8192
        assert config.num_attention_heads == 64
        assert config.num_key_value_heads == 8  # GQA
        assert config.head_dim == 128
    
    def test_llama3_8b_preset(self):
        """Test Llama 3 8B preset."""
        config = LlamaConfig.llama3_8b()
        assert config.vocab_size == 128256  # Larger vocab
        assert config.rope_theta == 500000.0  # Higher theta
        assert config.num_key_value_heads == 8
    
    def test_mistral_config(self):
        """Test Mistral configuration."""
        config = MistralConfig.mistral_7b()
        assert config.sliding_window == 4096
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 14336
    
    def test_config_from_dict(self):
        """Test creating config from HuggingFace-style dict."""
        hf_config = {
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        }
        
        config = LlamaConfig.from_dict(hf_config)
        assert config.hidden_size == 2048
        assert config.num_key_value_heads == 4
        assert config.head_dim == 128  # 2048 / 16


class TestModelInfo:
    """Test model info and memory estimation."""
    
    def test_model_info_from_config(self):
        """Test creating ModelInfo from config dict."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        }
        
        info = ModelInfo.from_config(config, name="test-model")
        
        assert info.name == "test-model"
        assert info.architecture == "LlamaForCausalLM"
        assert info.num_layers == 32
        assert info.num_parameters > 0
    
    def test_memory_estimation(self):
        """Test memory estimation for different quantization."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        
        info = ModelInfo.from_config(config)
        
        # FP16 should be ~2x INT8
        assert info.fp16_memory_gb > info.int8_memory_gb
        assert abs(info.fp16_memory_gb / info.int8_memory_gb - 2.0) < 0.1
        
        # INT8 should be ~2x INT4
        assert info.int8_memory_gb > info.int4_memory_gb
        assert abs(info.int8_memory_gb / info.int4_memory_gb - 2.0) < 0.1
    
    def test_model_info_str(self):
        """Test string representation."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        
        info = ModelInfo.from_config(config, name="llama-7b")
        info_str = str(info)
        
        assert "llama-7b" in info_str
        assert "Parameters" in info_str
        assert "GB" in info_str


class TestRMSNorm:
    """Test RMSNorm layer."""
    
    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        norm = RMSNorm(hidden_size=512, eps=1e-6)
        x = torch.randn(2, 10, 512)
        
        out = norm(x)
        
        assert out.shape == x.shape
        # Output should have unit variance (approximately)
        var = out.float().pow(2).mean(-1)
        assert torch.allclose(var, torch.ones_like(var), atol=0.5)
    
    def test_rmsnorm_dtype_preservation(self):
        """Test that RMSNorm preserves input dtype."""
        norm = RMSNorm(hidden_size=256)
        
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            x = torch.randn(1, 5, 256, dtype=dtype)
            out = norm(x)
            assert out.dtype == dtype


class TestRotaryEmbedding:
    """Test Rotary Position Embeddings."""
    
    def test_rope_basic(self):
        """Test basic RoPE functionality."""
        dim = 64
        max_pos = 512
        
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos)
        
        x = torch.randn(2, 8, 128, dim)  # [batch, heads, seq, dim]
        cos, sin = rope(x)
        
        assert cos.shape[-1] == dim
        assert sin.shape[-1] == dim
    
    def test_rope_position_encoding(self):
        """Test that different positions get different encodings."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=1024)
        
        x = torch.randn(1, 1, 10, 64)
        cos1, sin1 = rope(x)
        
        # Different positions should have different cos/sin
        assert not torch.allclose(cos1[0], cos1[1])
    
    def test_apply_rotary_pos_emb(self):
        """Test applying rotary embeddings to Q/K."""
        dim = 64
        rope = RotaryEmbedding(dim=dim)
        
        q = torch.randn(2, 8, 16, dim)
        k = torch.randn(2, 8, 16, dim)
        
        cos, sin = rope(q)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Values should change after rotation
        assert not torch.allclose(q, q_rot)


class TestLlamaModel:
    """Test Llama model architecture."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=256,
        )
    
    def test_llama_init(self, small_config):
        """Test Llama model initialization."""
        model = LlamaModel(small_config)
        
        assert len(model.layers) == small_config.num_hidden_layers
        assert model.embed_tokens.weight.shape[0] == small_config.vocab_size
        assert model.embed_tokens.weight.shape[1] == small_config.hidden_size
    
    def test_llama_forward(self, small_config):
        """Test Llama forward pass."""
        model = LlamaModel(small_config)
        model.eval()
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, past_kv = model(input_ids, use_cache=True)
        
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(past_kv) == small_config.num_hidden_layers
        assert past_kv[0][0].shape[2] == seq_len  # Key cache seq length
    
    def test_llama_generation(self, small_config):
        """Test Llama text generation."""
        model = LlamaModel(small_config)
        model.eval()
        
        # Prompt
        input_ids = torch.randint(0, small_config.vocab_size, (1, 5))
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        
        assert output.shape[0] == 1
        assert output.shape[1] == 15  # 5 prompt + 10 generated
    
    def test_llama_gqa(self, small_config):
        """Test Grouped Query Attention."""
        # 4 query heads, 2 KV heads = each KV head serves 2 query heads
        model = LlamaModel(small_config)
        
        # Check attention layer has correct dims
        attn = model.layers[0].self_attn
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.num_key_value_groups == 2
    
    def test_llama_kv_cache(self, small_config):
        """Test KV cache management."""
        model = LlamaModel(small_config)
        model.eval()
        
        # First forward (prefill)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        with torch.no_grad():
            _, past_kv = model(input_ids, use_cache=True)
        
        # Second forward (decode with single token)
        next_token = torch.randint(0, small_config.vocab_size, (1, 1))
        with torch.no_grad():
            logits, new_past_kv = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        
        # KV cache should grow by 1
        assert new_past_kv[0][0].shape[2] == 9  # 8 + 1
    
    def test_llama_param_count(self, small_config):
        """Test parameter counting."""
        model = LlamaModel(small_config)
        
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
        
        assert total_params > non_embed_params
        assert non_embed_params > 0


class TestMistralModel:
    """Test Mistral model architecture."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small Mistral config."""
        return MistralConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=256,
            sliding_window=64,
        )
    
    def test_mistral_init(self, small_config):
        """Test Mistral initialization."""
        model = MistralModel(small_config)
        
        assert len(model.layers) == 2
        assert model.layers[0].self_attn.sliding_window == 64
    
    def test_mistral_forward(self, small_config):
        """Test Mistral forward pass."""
        model = MistralModel(small_config)
        model.eval()
        
        input_ids = torch.randint(0, small_config.vocab_size, (1, 32))
        
        with torch.no_grad():
            logits, past_kv = model(input_ids, use_cache=True)
        
        assert logits.shape == (1, 32, small_config.vocab_size)


class TestLoadConfig:
    """Test loading configuration."""
    
    def test_default_config(self):
        """Test default load config."""
        config = LoadConfig()
        
        assert config.device == "cuda"
        assert config.quantization == QuantizationType.NONE
        assert config.use_mmap == True
        assert config.dtype == torch.float16
    
    def test_quantization_types(self):
        """Test different quantization configurations."""
        for quant in ["none", "int8", "int4"]:
            config = LoadConfig(quantization=quant)
            assert config.quantization == QuantizationType(quant)
    
    def test_device_map(self):
        """Test device mapping configuration."""
        # Auto mapping
        config1 = LoadConfig(device_map="auto")
        assert config1.device_map == "auto"
        
        # Manual mapping
        device_map = {"model.embed_tokens": 0, "model.layers.0": 1}
        config2 = LoadConfig(device_map=device_map)
        assert config2.device_map == device_map


class TestQuantization:
    """Test quantization during loading."""
    
    def test_int8_quantization(self):
        """Test INT8 quantization."""
        from zse.models.loader.base import BaseModelLoader, LoadConfig, QuantizationType
        
        config = LoadConfig(quantization=QuantizationType.INT8)
        
        # Create a mock loader to test quantization
        class TestLoader(BaseModelLoader):
            def load_model_info(self, path):
                pass
            def load_weights(self, path, model, callback=None):
                pass
            def iterate_weights(self, path):
                pass
        
        loader = TestLoader(config)
        
        # Test quantization
        tensor = torch.randn(128, 256)
        quantized = loader._quantize_int8(tensor)
        
        assert quantized.shape == tensor.shape
        # Quantized values should be more "discrete"
    
    def test_int4_quantization(self):
        """Test INT4 quantization."""
        from zse.models.loader.base import BaseModelLoader, LoadConfig, QuantizationType
        
        config = LoadConfig(quantization=QuantizationType.INT4)
        
        class TestLoader(BaseModelLoader):
            def load_model_info(self, path):
                pass
            def load_weights(self, path, model, callback=None):
                pass
            def iterate_weights(self, path):
                pass
        
        loader = TestLoader(config)
        
        tensor = torch.randn(128, 256)
        quantized = loader._quantize_int4(tensor)
        
        assert quantized.shape == tensor.shape


class TestSafetensorsLoader:
    """Test safetensors file loading."""
    
    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create a temporary model directory with config."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        
        # Create config.json
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 256,
        }
        
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        return model_dir
    
    def test_load_model_info(self, temp_model_dir):
        """Test loading model info from directory."""
        from zse.models.loader.safetensors_loader import SafetensorsLoader
        
        loader = SafetensorsLoader()
        info = loader.load_model_info(str(temp_model_dir))
        
        assert info.hidden_size == 256
        assert info.num_layers == 2
        assert info.architecture == "LlamaForCausalLM"
    
    @pytest.mark.skipif(
        not pytest.importorskip("safetensors", reason="safetensors not installed"),
        reason="safetensors not installed"
    )
    def test_safetensors_file_detection(self, temp_model_dir):
        """Test finding safetensors files."""
        from safetensors.torch import save_file
        from zse.models.loader.safetensors_loader import SafetensorsLoader
        
        # Create dummy safetensors file
        tensors = {"test": torch.randn(10, 10)}
        save_file(tensors, str(temp_model_dir / "model.safetensors"))
        
        loader = SafetensorsLoader()
        files = loader._find_weight_files(str(temp_model_dir))
        
        assert len(files) == 1
        assert "model.safetensors" in files[0]


class TestModelHub:
    """Test unified ModelHub interface."""
    
    def test_is_hf_repo_detection(self):
        """Test HuggingFace repo ID detection."""
        hub = ModelHub()
        
        # HF repo IDs
        assert hub._is_hf_repo("meta-llama/Llama-2-7b")
        assert hub._is_hf_repo("mistralai/Mistral-7B-v0.1")
        
        # Local paths
        assert not hub._is_hf_repo("/path/to/model")
        assert not hub._is_hf_repo("./local_model")


# GPU-specific tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUFeatures:
    """Tests requiring GPU."""
    
    def test_model_to_cuda(self):
        """Test moving model to CUDA."""
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
        )
        
        model = LlamaModel(config)
        model = model.cuda().half()
        
        input_ids = torch.randint(0, 1000, (1, 16)).cuda()
        
        with torch.no_grad():
            logits, _ = model(input_ids)
        
        assert logits.device.type == "cuda"
        assert logits.dtype == torch.float16
    
    def test_auto_device_map(self):
        """Test automatic device mapping."""
        from zse.models.loader.base import LoadConfig
        from zse.models.loader.safetensors_loader import SafetensorsLoader
        
        config = LoadConfig(device_map="auto")
        loader = SafetensorsLoader(config)
        
        # Create model info for small model
        info = ModelInfo(
            name="test",
            architecture="llama",
            num_parameters=1_000_000,
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=256,
        )
        
        device_map = loader.get_device_map(info)
        
        # For single GPU, should map everything to device 0
        if torch.cuda.device_count() == 1:
            assert device_map == {"": 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
