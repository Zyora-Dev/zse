"""Full pipeline tests: synthetic safetensors → convert → .zse → load → verify.

Tests both Llama and Qwen2 architectures end-to-end through the real
convert_hf_to_zse() pipeline, using synthetic safetensors files we
write ourselves (no downloads needed).
"""

import os
import sys
import json
import struct
import random
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-engine'))

from zse_engine.format.spec import should_quantize
from zse_engine.format.config import ModelConfig
from zse_engine.format.convert import convert_hf_to_zse
from zse_engine.format.loader import ZSELoader
from zse_engine.format.arch.base import detect_architecture, get_adapter
from zse_engine.format.arch.llama import LlamaAdapter
from zse_engine.format.arch.qwen2 import Qwen2Adapter


# --------------------------------------------------------------------------- #
# Synthetic safetensors writer (pure Python, no deps)
# --------------------------------------------------------------------------- #

def _pack_bf16(value: float) -> bytes:
    """Convert float32 → BF16 as 2 bytes."""
    f32_bytes = struct.pack('<f', value)
    # BF16 = top 16 bits of float32
    return f32_bytes[2:4]


def _pack_f16(value: float) -> bytes:
    """Convert float32 → F16 as 2 bytes."""
    return struct.pack('<e', value)


def write_safetensors(path: str, tensors: dict, dtype: str = "BF16"):
    """Write a safetensors file with given tensors.

    Args:
        path: Output file path
        tensors: dict of name -> (shape, flat_float_list)
        dtype: "BF16" or "F16"
    """
    # Build metadata and pack tensor data
    metadata = {}
    data_parts = []
    current_offset = 0

    for name, (shape, values) in tensors.items():
        n_elem = 1
        for d in shape:
            n_elem *= d
        assert len(values) == n_elem, f"{name}: expected {n_elem} vals, got {len(values)}"

        # Pack values
        if dtype == "BF16":
            packed = b''.join(_pack_bf16(v) for v in values)
        elif dtype == "F16":
            packed = b''.join(_pack_f16(v) for v in values)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        metadata[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [current_offset, current_offset + len(packed)],
        }
        data_parts.append(packed)
        current_offset += len(packed)

    # Serialize header
    header_json = json.dumps(metadata).encode('utf-8')
    # Pad header to 8-byte alignment
    pad = (8 - len(header_json) % 8) % 8
    header_json += b' ' * pad

    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


def write_minimal_tokenizer_json(path: str, vocab_size: int = 100):
    """Write a minimal tokenizer.json for testing."""
    vocab = {}
    for i in range(min(vocab_size, 256)):
        # Map printable ASCII chars to IDs
        if 32 <= i < 127:
            vocab[chr(i)] = i
        else:
            vocab[f"<byte_{i:02x}>"] = i

    # Fill remaining vocab with merge tokens
    for i in range(256, vocab_size):
        vocab[f"<tok_{i}>"] = i

    data = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True},
            {"id": 1, "content": "<s>", "special": True},
            {"id": 2, "content": "</s>", "special": True},
        ],
    }
    with open(path, 'w') as f:
        json.dump(data, f)


# --------------------------------------------------------------------------- #
# Generate synthetic model weights
# --------------------------------------------------------------------------- #

def generate_llama_model(model_dir: str, num_layers: int = 2):
    """Generate a synthetic tiny Llama model in HF format."""
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": 2,
        "vocab_size": 200,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }

    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    write_minimal_tokenizer_json(os.path.join(model_dir, "tokenizer.json"), vocab_size=200)

    # Generate tensors
    random.seed(42)
    H = config["hidden_size"]
    I = config["intermediate_size"]
    V = config["vocab_size"]
    KV_H = config["num_key_value_heads"]
    HEAD_DIM = H // config["num_attention_heads"]

    def rand_weights(n):
        return [random.gauss(0, 0.02) for _ in range(n)]

    tensors = {}
    tensors["model.embed_tokens.weight"] = ((V, H), rand_weights(V * H))

    for i in range(num_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = ((H, H), rand_weights(H * H))
        tensors[f"{p}.self_attn.k_proj.weight"] = ((KV_H * HEAD_DIM, H), rand_weights(KV_H * HEAD_DIM * H))
        tensors[f"{p}.self_attn.v_proj.weight"] = ((KV_H * HEAD_DIM, H), rand_weights(KV_H * HEAD_DIM * H))
        tensors[f"{p}.self_attn.o_proj.weight"] = ((H, H), rand_weights(H * H))
        tensors[f"{p}.mlp.gate_proj.weight"] = ((I, H), rand_weights(I * H))
        tensors[f"{p}.mlp.up_proj.weight"] = ((I, H), rand_weights(I * H))
        tensors[f"{p}.mlp.down_proj.weight"] = ((H, I), rand_weights(H * I))
        tensors[f"{p}.input_layernorm.weight"] = ((H,), rand_weights(H))
        tensors[f"{p}.post_attention_layernorm.weight"] = ((H,), rand_weights(H))

    tensors["model.norm.weight"] = ((H,), rand_weights(H))
    tensors["lm_head.weight"] = ((V, H), rand_weights(V * H))

    write_safetensors(os.path.join(model_dir, "model.safetensors"), tensors)

    return config, tensors


def generate_qwen2_model(model_dir: str, num_layers: int = 2):
    """Generate a synthetic tiny Qwen2 model in HF format."""
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": 2,
        "vocab_size": 200,
        "max_position_embeddings": 64,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "sliding_window": 64,
        "max_window_layers": num_layers,
    }

    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    write_minimal_tokenizer_json(os.path.join(model_dir, "tokenizer.json"), vocab_size=200)

    random.seed(99)
    H = config["hidden_size"]
    I = config["intermediate_size"]
    V = config["vocab_size"]
    KV_H = config["num_key_value_heads"]
    HEAD_DIM = H // config["num_attention_heads"]

    def rand_weights(n):
        return [random.gauss(0, 0.02) for _ in range(n)]

    tensors = {}
    tensors["model.embed_tokens.weight"] = ((V, H), rand_weights(V * H))

    for i in range(num_layers):
        p = f"model.layers.{i}"
        # Attention — Qwen2 has biases on QKV
        tensors[f"{p}.self_attn.q_proj.weight"] = ((H, H), rand_weights(H * H))
        tensors[f"{p}.self_attn.q_proj.bias"] = ((H,), rand_weights(H))
        tensors[f"{p}.self_attn.k_proj.weight"] = ((KV_H * HEAD_DIM, H), rand_weights(KV_H * HEAD_DIM * H))
        tensors[f"{p}.self_attn.k_proj.bias"] = ((KV_H * HEAD_DIM,), rand_weights(KV_H * HEAD_DIM))
        tensors[f"{p}.self_attn.v_proj.weight"] = ((KV_H * HEAD_DIM, H), rand_weights(KV_H * HEAD_DIM * H))
        tensors[f"{p}.self_attn.v_proj.bias"] = ((KV_H * HEAD_DIM,), rand_weights(KV_H * HEAD_DIM))
        tensors[f"{p}.self_attn.o_proj.weight"] = ((H, H), rand_weights(H * H))
        # MLP
        tensors[f"{p}.mlp.gate_proj.weight"] = ((I, H), rand_weights(I * H))
        tensors[f"{p}.mlp.up_proj.weight"] = ((I, H), rand_weights(I * H))
        tensors[f"{p}.mlp.down_proj.weight"] = ((H, I), rand_weights(H * I))
        # Norms
        tensors[f"{p}.input_layernorm.weight"] = ((H,), rand_weights(H))
        tensors[f"{p}.post_attention_layernorm.weight"] = ((H,), rand_weights(H))

    tensors["model.norm.weight"] = ((H,), rand_weights(H))
    # No lm_head — tie_word_embeddings=True

    write_safetensors(os.path.join(model_dir, "model.safetensors"), tensors)

    return config, tensors


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_llama_convert_pipeline():
    """Full pipeline: synthetic Llama safetensors → convert → .zse → load → verify."""
    tmpdir = tempfile.mkdtemp(prefix="zse_test_llama_")
    zse_path = os.path.join(tmpdir, "model.zse")

    try:
        # Generate synthetic model
        hf_config, orig_tensors = generate_llama_model(tmpdir, num_layers=2)
        adapter = LlamaAdapter()

        # Convert
        progress = []
        def on_progress(name, cur, total):
            progress.append((name, cur, total))

        convert_hf_to_zse(tmpdir, zse_path, progress_callback=on_progress)

        # Verify progress callback was called for each tensor
        assert len(progress) == len(orig_tensors), \
            f"Progress: {len(progress)} calls, expected {len(orig_tensors)}"

        # Load and verify
        with ZSELoader(zse_path) as loader:
            c = loader.config
            assert c.arch == "llama"
            assert c.num_layers == 2
            assert c.num_heads == 4
            assert c.num_kv_heads == 2
            assert c.hidden_size == 128
            assert c.vocab_size == 200
            assert c.tie_word_embeddings is False

            # Verify tokenizer
            assert loader.tokenizer is not None
            assert loader.tokenizer.vocab_size > 0

            # Verify all tensors present
            wi = loader.weight_index
            assert len(wi) == len(orig_tensors)

            # Verify each tensor
            for hf_name, (shape, orig_vals) in orig_tensors.items():
                zse_name = adapter.map_tensor_name(hf_name)
                entry = wi.find(zse_name)
                assert entry is not None, f"Missing: {zse_name}"
                assert entry.shape == shape, f"Shape mismatch: {zse_name}"

                # Check dtype assignment
                if should_quantize(zse_name):
                    assert entry.dtype == "int4", f"{zse_name} should be int4"
                else:
                    assert entry.dtype == "float16", f"{zse_name} should be float16"

            # Spot-check dequantized values (quantized tensor)
            q_entry = wi.find("layers.0.self_attn.q_proj.weight")
            assert q_entry is not None
            recovered = loader.get_weight_as_float(q_entry)
            hf_q_vals = orig_tensors["model.layers.0.self_attn.q_proj.weight"][1]
            # BF16 loses precision, then INT4 quantization adds more error
            # But should still be in reasonable range
            max_err = max(abs(a - b) for a, b in zip(hf_q_vals, recovered))
            assert max_err < 0.2, f"Quant error too high: {max_err}"

            print(f"\n{loader.summary()}\n")

        file_size = os.path.getsize(zse_path)
        print(f"  [PASS] Llama convert pipeline ({file_size:,} bytes, "
              f"{len(orig_tensors)} tensors)")

    finally:
        shutil.rmtree(tmpdir)


def test_qwen2_convert_pipeline():
    """Full pipeline: synthetic Qwen2 safetensors → convert → .zse → load → verify."""
    tmpdir = tempfile.mkdtemp(prefix="zse_test_qwen2_")
    zse_path = os.path.join(tmpdir, "model.zse")

    try:
        hf_config, orig_tensors = generate_qwen2_model(tmpdir, num_layers=2)
        adapter = Qwen2Adapter()

        # Convert
        convert_hf_to_zse(tmpdir, zse_path)

        with ZSELoader(zse_path) as loader:
            c = loader.config
            assert c.arch == "qwen2"
            assert c.num_layers == 2
            assert c.num_heads == 4
            assert c.num_kv_heads == 2
            assert c.hidden_size == 128
            assert c.vocab_size == 200
            assert c.tie_word_embeddings is True
            assert c.rope_theta == 1000000.0
            assert c.rms_norm_eps == 1e-6

            # Verify tokenizer
            assert loader.tokenizer is not None

            wi = loader.weight_index
            assert len(wi) == len(orig_tensors)

            # Verify Qwen-specific: biases exist and are fp16
            for i in range(2):
                q_bias = wi.find(f"layers.{i}.self_attn.q_proj.bias")
                assert q_bias is not None, f"Missing q_proj.bias for layer {i}"
                assert q_bias.dtype == "float16", f"Bias should be float16, got {q_bias.dtype}"

                k_bias = wi.find(f"layers.{i}.self_attn.k_proj.bias")
                assert k_bias is not None, f"Missing k_proj.bias for layer {i}"
                assert k_bias.dtype == "float16"

                v_bias = wi.find(f"layers.{i}.self_attn.v_proj.bias")
                assert v_bias is not None, f"Missing v_proj.bias for layer {i}"
                assert v_bias.dtype == "float16"

            # Verify no lm_head (tied embeddings)
            assert wi.find("lm_head.weight") is None

            # Verify embed_tokens is fp16
            embed = wi.find("embed_tokens.weight")
            assert embed is not None
            assert embed.dtype == "float16"

            # Verify attention weights are int4 quantized
            q_weight = wi.find("layers.0.self_attn.q_proj.weight")
            assert q_weight is not None
            assert q_weight.dtype == "int4"

            print(f"\n{loader.summary()}\n")

        file_size = os.path.getsize(zse_path)
        print(f"  [PASS] Qwen2 convert pipeline ({file_size:,} bytes, "
              f"{len(orig_tensors)} tensors)")

    finally:
        shutil.rmtree(tmpdir)


def test_architecture_detection():
    """Test arch detection for various HF configs."""
    cases = [
        ({"architectures": ["LlamaForCausalLM"]}, "llama"),
        ({"architectures": ["MistralForCausalLM"]}, "mistral"),
        ({"architectures": ["Qwen2ForCausalLM"]}, "qwen2"),
        ({"architectures": ["Qwen2_5ForCausalLM"]}, "qwen2"),
        ({"model_type": "llama"}, "llama"),
        ({"model_type": "qwen2"}, "qwen2"),
    ]
    for hf_config, expected in cases:
        result = detect_architecture(hf_config)
        assert result == expected, f"Expected {expected}, got {result} for {hf_config}"

    print("  [PASS] architecture detection")


def test_safetensors_roundtrip():
    """Test our synthetic safetensors writer can be read by our safetensors reader."""
    from zse_engine.format.convert import read_safetensors_metadata, read_safetensors_tensor

    tmpdir = tempfile.mkdtemp(prefix="zse_test_st_")
    st_path = os.path.join(tmpdir, "test.safetensors")

    try:
        random.seed(77)
        original = {
            "weight_a": ((4, 4), [random.gauss(0, 1.0) for _ in range(16)]),
            "weight_b": ((8,), [random.gauss(0, 1.0) for _ in range(8)]),
        }

        write_safetensors(st_path, original, dtype="BF16")

        # Read back
        metadata, data_offset = read_safetensors_metadata(st_path)
        assert "weight_a" in metadata
        assert "weight_b" in metadata
        assert metadata["weight_a"]["shape"] == [4, 4]
        assert metadata["weight_b"]["shape"] == [8]

        # Read tensor values
        vals_a, shape_a = read_safetensors_tensor(st_path, metadata["weight_a"], data_offset)
        assert shape_a == (4, 4)
        assert len(vals_a) == 16

        # BF16 precision: ~3 decimal digits
        for orig, read in zip(original["weight_a"][1], vals_a):
            assert abs(orig - read) < 0.05, f"BF16 roundtrip error: {orig} vs {read}"

        print("  [PASS] safetensors roundtrip")
    finally:
        shutil.rmtree(tmpdir)


def test_multi_shard_safetensors():
    """Test conversion with multiple safetensors shards (like real large models)."""
    tmpdir = tempfile.mkdtemp(prefix="zse_test_shard_")
    zse_path = os.path.join(tmpdir, "model.zse")

    try:
        random.seed(55)
        H = 64
        V = 50
        I = 128

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": H,
            "intermediate_size": I,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "vocab_size": V,
            "max_position_embeddings": 32,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(tmpdir, "config.json"), 'w') as f:
            json.dump(config, f)

        write_minimal_tokenizer_json(os.path.join(tmpdir, "tokenizer.json"), V)

        def rand(n):
            return [random.gauss(0, 0.02) for _ in range(n)]

        HEAD_DIM = H // 2
        KV_H = 1

        # Shard 1: embeddings + layer attention
        shard1 = {
            "model.embed_tokens.weight": ((V, H), rand(V * H)),
            "model.layers.0.self_attn.q_proj.weight": ((H, H), rand(H * H)),
            "model.layers.0.self_attn.k_proj.weight": ((KV_H * HEAD_DIM, H), rand(KV_H * HEAD_DIM * H)),
            "model.layers.0.self_attn.v_proj.weight": ((KV_H * HEAD_DIM, H), rand(KV_H * HEAD_DIM * H)),
            "model.layers.0.self_attn.o_proj.weight": ((H, H), rand(H * H)),
        }

        # Shard 2: layer MLP + norms + lm_head
        shard2 = {
            "model.layers.0.mlp.gate_proj.weight": ((I, H), rand(I * H)),
            "model.layers.0.mlp.up_proj.weight": ((I, H), rand(I * H)),
            "model.layers.0.mlp.down_proj.weight": ((H, I), rand(H * I)),
            "model.layers.0.input_layernorm.weight": ((H,), rand(H)),
            "model.layers.0.post_attention_layernorm.weight": ((H,), rand(H)),
            "model.norm.weight": ((H,), rand(H)),
            "lm_head.weight": ((V, H), rand(V * H)),
        }

        write_safetensors(os.path.join(tmpdir, "model-00001-of-00002.safetensors"), shard1)
        write_safetensors(os.path.join(tmpdir, "model-00002-of-00002.safetensors"), shard2)

        # Convert
        convert_hf_to_zse(tmpdir, zse_path)

        with ZSELoader(zse_path) as loader:
            assert loader.config.arch == "llama"
            assert len(loader.weight_index) == len(shard1) + len(shard2)

        file_size = os.path.getsize(zse_path)
        print(f"  [PASS] multi-shard safetensors ({file_size:,} bytes, "
              f"{len(shard1) + len(shard2)} tensors from 2 shards)")

    finally:
        shutil.rmtree(tmpdir)


def test_qwen2_adapter_config_mapping():
    """Test Qwen2 adapter maps real Qwen2.5-0.5B config correctly."""
    # Real Qwen2.5-0.5B config
    hf_config = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "sliding_window": 32768,
    }

    adapter = Qwen2Adapter()
    config = adapter.config_from_hf(hf_config)

    assert config.arch == "qwen2"
    assert config.num_layers == 24
    assert config.num_heads == 14
    assert config.num_kv_heads == 2
    assert config.head_dim == 64  # 896 // 14
    assert config.hidden_size == 896
    assert config.intermediate_size == 4864
    assert config.vocab_size == 151936
    assert config.tie_word_embeddings is True
    assert config.rope_theta == 1000000.0

    # Check expected tensors count
    # embed + 24 layers * 12 tensors (9 weights + 3 biases) + norm = 1 + 288 + 1 = 290
    expected = adapter.expected_tensors(config)
    assert len(expected) == 1 + 24 * 12 + 1  # 290
    assert "lm_head.weight" not in expected  # tied

    warnings = adapter.validate_config(config)
    # 14 heads / 2 kv_heads = 7 (valid), 896 / 14 = 64 (valid)
    assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    print("  [PASS] Qwen2 real config mapping (Qwen2.5-0.5B shape)")


def test_llama_adapter_config_mapping():
    """Test Llama adapter maps real Llama-3.2-1B config correctly."""
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }

    adapter = LlamaAdapter()
    config = adapter.config_from_hf(hf_config)

    assert config.arch == "llama"
    assert config.num_layers == 16
    assert config.num_heads == 32
    assert config.num_kv_heads == 8
    assert config.head_dim == 64
    assert config.hidden_size == 2048
    assert config.tie_word_embeddings is True

    expected = adapter.expected_tensors(config)
    # embed + 16 layers * 9 tensors + norm = 1 + 144 + 1 = 146 (no lm_head, tied)
    assert len(expected) == 146
    assert "lm_head.weight" not in expected

    print("  [PASS] Llama real config mapping (Llama-3.2-1B shape)")


def main():
    print("=" * 60)
    print("ZSE Format — Full Pipeline Tests (Llama + Qwen2)")
    print("=" * 60)

    tests = [
        test_architecture_detection,
        test_safetensors_roundtrip,
        test_llama_adapter_config_mapping,
        test_qwen2_adapter_config_mapping,
        test_llama_convert_pipeline,
        test_qwen2_convert_pipeline,
        test_multi_shard_safetensors,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
