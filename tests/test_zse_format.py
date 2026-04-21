"""End-to-end test: write .zse → read back → verify all sections."""

import os
import sys
import math
import struct
import tempfile
import random

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-engine'))

from zse_engine.format.spec import (
    MAGIC, VERSION, SectionType, Flags, QuantMethod,
    PAGE_SIZE, TENSOR_ALIGN, should_quantize,
)
from zse_engine.format.config import ModelConfig, QuantConfig
from zse_engine.format.header import FileHeader, SectionEntry, read_header_and_sections
from zse_engine.format.weight_index import WeightIndex, WeightEntry
from zse_engine.format.quantize import quantize_tensor, dequantize_tensor
from zse_engine.format.quantize import quantize_tensor_int8, dequantize_tensor_int8
from zse_engine.format.tokenizer import BPETokenizer, SpecialTokens
from zse_engine.format.writer import ZSEWriter
from zse_engine.format.loader import ZSELoader
from zse_engine.format.arch.base import get_adapter, detect_architecture
from zse_engine.format.arch.llama import LlamaAdapter


def test_serializer_roundtrip():
    """Test binary serializer encode/decode."""
    from zse_engine.format import serializer

    cases = [
        None,
        True,
        False,
        42,
        -7,
        3.14,
        "hello",
        b"\x00\x01\x02",
        [1, 2, 3],
        {"key": "value", "num": 123},
        {"nested": {"list": [1, "two", None, True]}},
    ]
    for obj in cases:
        encoded = serializer.encode(obj)
        decoded = serializer.decode(encoded)
        assert decoded == obj, f"Roundtrip failed for {obj}: got {decoded}"

    print("  [PASS] serializer roundtrip")


def test_quantize_roundtrip():
    """Test INT4 quantization and dequantization."""
    random.seed(42)
    weights = [random.gauss(0, 0.5) for _ in range(512)]
    shape = (512,)

    packed, scales, zeros = quantize_tensor(weights, group_size=128)
    recovered = dequantize_tensor(packed, scales, zeros, shape, group_size=128)

    assert len(recovered) == 512
    # Check quantization error is bounded (INT4 = 16 levels)
    max_err = 0.0
    for orig, rec in zip(weights, recovered):
        err = abs(orig - rec)
        if err > max_err:
            max_err = err

    # With 4-bit, max error per group ≈ range/15
    assert max_err < 0.5, f"Max quantization error too high: {max_err}"
    print(f"  [PASS] quantize roundtrip (max_err={max_err:.4f})")


def test_config_roundtrip():
    """Test ModelConfig serialize/deserialize."""
    config = ModelConfig(
        arch="llama",
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        head_dim=64,
        hidden_size=512,
        intermediate_size=1376,
        vocab_size=1000,
        max_seq_len=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    data = config.serialize()
    restored = ModelConfig.deserialize(data)

    assert restored.arch == config.arch
    assert restored.num_layers == config.num_layers
    assert restored.num_heads == config.num_heads
    assert restored.num_kv_heads == config.num_kv_heads
    assert restored.hidden_size == config.hidden_size
    assert restored.vocab_size == config.vocab_size
    print("  [PASS] config roundtrip")


def test_tokenizer_roundtrip():
    """Test BPETokenizer serialize/deserialize."""
    tok = BPETokenizer(
        vocab={"a": 0, "b": 1, "ab": 2, "c": 3},
        merges=[("a", "b")],
        special_tokens=SpecialTokens(bos_id=10, eos_id=11),
        added_tokens={"<pad>": 99},
    )
    data = tok.serialize()
    restored = BPETokenizer.deserialize(data)

    assert restored.vocab == tok.vocab
    assert restored.merges == tok.merges
    assert restored.special_tokens.bos_id == 10
    assert restored.special_tokens.eos_id == 11
    assert restored.added_tokens == {"<pad>": 99}
    print("  [PASS] tokenizer roundtrip")


def test_weight_index_roundtrip():
    """Test WeightIndex serialize/deserialize."""
    index = WeightIndex()
    index.add(WeightEntry(
        name="layers.0.q_proj.weight",
        shape=(512, 512),
        dtype="int4",
        quant_method=QuantMethod.INT4_ASYM,
        group_size=128,
        data_offset=0, data_nbytes=65536,
        scale_offset=65536, scale_nbytes=2048,
        zeros_offset=67584, zeros_nbytes=2048,
    ))
    index.add(WeightEntry(
        name="norm.weight",
        shape=(512,),
        dtype="float16",
        data_offset=70000, data_nbytes=1024,
    ))

    data = index.serialize()
    restored = WeightIndex.deserialize(data)

    assert len(restored) == 2
    e0 = restored.find("layers.0.q_proj.weight")
    assert e0 is not None
    assert e0.shape == (512, 512)
    assert e0.dtype == "int4"
    assert e0.data_nbytes == 65536

    e1 = restored.find("norm.weight")
    assert e1 is not None
    assert e1.dtype == "float16"
    print("  [PASS] weight_index roundtrip")


def test_header_pack_unpack():
    """Test FileHeader and SectionEntry pack/unpack."""
    header = FileHeader(version=1, total_size=123456, flags=5, num_sections=3)
    packed = header.pack()
    assert len(packed) == 64

    restored = FileHeader.unpack(packed)
    assert restored.version == 1
    assert restored.total_size == 123456
    assert restored.flags == 5
    assert restored.num_sections == 3

    entry = SectionEntry(type=SectionType.CONFIG, offset=64, size=200, crc32=0xDEADBEEF)
    packed_entry = entry.pack()
    assert len(packed_entry) == 32

    restored_entry = SectionEntry.unpack(packed_entry)
    assert restored_entry.type == SectionType.CONFIG
    assert restored_entry.offset == 64
    assert restored_entry.size == 200
    assert restored_entry.crc32 == 0xDEADBEEF
    print("  [PASS] header pack/unpack")


def test_llama_adapter():
    """Test Llama architecture adapter."""
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 512,
        "intermediate_size": 1376,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
    }

    arch = detect_architecture(hf_config)
    assert arch == "llama"

    adapter = get_adapter("llama")
    config = adapter.config_from_hf(hf_config)
    assert config.arch == "llama"
    assert config.num_layers == 4
    assert config.num_heads == 8
    assert config.num_kv_heads == 4
    assert config.hidden_size == 512

    # Test tensor name mapping
    assert adapter.map_tensor_name("model.layers.0.self_attn.q_proj.weight") == \
           "layers.0.self_attn.q_proj.weight"
    assert adapter.map_tensor_name("lm_head.weight") == "lm_head.weight"

    # Test expected tensors
    expected = adapter.expected_tensors(config)
    assert "embed_tokens.weight" in expected
    assert "layers.0.self_attn.q_proj.weight" in expected
    assert "norm.weight" in expected
    print("  [PASS] llama adapter")


def test_full_write_read_roundtrip():
    """End-to-end: write a .zse file, read it back, verify everything."""
    random.seed(123)

    # Create a tiny model config
    config = ModelConfig(
        arch="llama",
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=100,
        max_seq_len=64,
    )

    # Create a simple tokenizer
    tok = BPETokenizer(
        vocab={"hello": 0, "world": 1, " ": 2},
        merges=[],
        special_tokens=SpecialTokens(bos_id=98, eos_id=99),
    )

    # Generate fake weights
    tensors = []
    adapter = LlamaAdapter()
    expected_names = adapter.expected_tensors(config)

    for name in expected_names:
        if "norm" in name:
            shape = (config.hidden_size,)
        elif "embed_tokens" in name or "lm_head" in name:
            shape = (config.vocab_size, config.hidden_size)
        elif "q_proj" in name or "o_proj" in name:
            shape = (config.hidden_size, config.hidden_size)
        elif "k_proj" in name or "v_proj" in name:
            shape = (config.num_kv_heads * config.head_dim, config.hidden_size)
        elif "gate_proj" in name or "up_proj" in name:
            shape = (config.intermediate_size, config.hidden_size)
        elif "down_proj" in name:
            shape = (config.hidden_size, config.intermediate_size)
        else:
            shape = (128,)

        n_elem = 1
        for d in shape:
            n_elem *= d
        weights = [random.gauss(0, 0.02) for _ in range(n_elem)]
        quantize = adapter.should_quantize(name)
        tensors.append((name, weights, shape, quantize))

    # Write .zse file
    with tempfile.NamedTemporaryFile(suffix='.zse', delete=False) as f:
        zse_path = f.name

    try:
        with ZSEWriter(zse_path) as writer:
            writer.set_config(config)
            writer.set_tokenizer(tok.serialize())
            writer.begin_weights()
            for name, weights, shape, quantize in tensors:
                writer.add_weight(name, weights, shape, quantize=quantize)

        # Verify file exists and has reasonable size
        file_size = os.path.getsize(zse_path)
        assert file_size > 0, "File is empty"

        # Read back with loader
        with ZSELoader(zse_path) as loader:
            # Verify header
            assert loader.header.version == VERSION
            assert loader.header.total_size == file_size

            # Verify config
            rc = loader.config
            assert rc.arch == "llama"
            assert rc.num_layers == 2
            assert rc.num_heads == 4
            assert rc.hidden_size == 128
            assert rc.vocab_size == 100

            # Verify tokenizer
            rt = loader.tokenizer
            assert rt is not None
            assert rt.vocab == {"hello": 0, "world": 1, " ": 2}
            assert rt.special_tokens.bos_id == 98

            # Verify weight index
            wi = loader.weight_index
            assert len(wi) == len(tensors)

            # Verify each weight can be read and dequantized
            for name, orig_weights, shape, was_quantized in tensors:
                entry = wi.find(name)
                assert entry is not None, f"Missing tensor: {name}"
                assert entry.shape == shape, f"Shape mismatch for {name}"

                # Read and verify data
                raw_data = loader.get_weight_data(entry)
                assert len(raw_data) == entry.data_nbytes

                if was_quantized:
                    assert entry.dtype == "int4"
                    scales = loader.get_weight_scales(entry)
                    zeros = loader.get_weight_zeros(entry)
                    assert len(scales) == entry.scale_nbytes
                    assert len(zeros) == entry.zeros_nbytes

                    # Dequantize and check error bound
                    recovered = loader.get_weight_as_float(entry)
                    assert len(recovered) == len(orig_weights)
                    max_err = max(abs(a - b) for a, b in zip(orig_weights, recovered))
                    assert max_err < 0.1, f"Quant error too high for {name}: {max_err}"
                else:
                    assert entry.dtype == "float16"

            # Print summary
            print(f"\n{loader.summary()}\n")

        print(f"  [PASS] full write/read roundtrip ({file_size:,} bytes, "
              f"{len(tensors)} tensors)")

    finally:
        os.unlink(zse_path)


def test_quantize_int8_roundtrip():
    """Test INT8 symmetric quantization and dequantization."""
    random.seed(42)
    weights = [random.gauss(0, 0.5) for _ in range(512)]
    shape = (512,)

    packed, scales = quantize_tensor_int8(weights, group_size=128)
    recovered = dequantize_tensor_int8(packed, scales, shape, group_size=128)

    assert len(recovered) == 512
    max_err = 0.0
    for orig, rec in zip(weights, recovered):
        err = abs(orig - rec)
        if err > max_err:
            max_err = err

    # INT8 = 256 levels, should be much tighter than INT4
    assert max_err < 0.05, f"Max INT8 quantization error too high: {max_err}"
    print(f"  [PASS] quantize_int8 roundtrip (max_err={max_err:.4f})")


def test_mmap_alignment():
    """Verify weight data section is page-aligned for mmap."""
    random.seed(456)

    config = ModelConfig(arch="llama", num_layers=1, num_heads=2,
                         num_kv_heads=1, head_dim=32, hidden_size=64,
                         intermediate_size=128, vocab_size=50)

    with tempfile.NamedTemporaryFile(suffix='.zse', delete=False) as f:
        zse_path = f.name

    try:
        with ZSEWriter(zse_path) as writer:
            writer.set_config(config)
            writer.begin_weights()
            weights = [random.gauss(0, 0.1) for _ in range(256)]
            writer.add_weight("test.weight", weights, (16, 16), quantize=True)

        with ZSELoader(zse_path) as loader:
            # Weight data section should start at a page boundary
            for section in loader._sections:
                if section.type == SectionType.WEIGHT_DATA:
                    assert section.offset % PAGE_SIZE == 0, \
                        f"Weight data not page-aligned: offset={section.offset}"
                    break

        print("  [PASS] mmap alignment")
    finally:
        os.unlink(zse_path)


def main():
    print("=" * 60)
    print("ZSE Format — End-to-End Tests")
    print("=" * 60)

    tests = [
        test_serializer_roundtrip,
        test_quantize_roundtrip,
        test_config_roundtrip,
        test_tokenizer_roundtrip,
        test_weight_index_roundtrip,
        test_header_pack_unpack,
        test_llama_adapter,
        test_full_write_read_roundtrip,
        test_quantize_int8_roundtrip,
        test_mmap_alignment,
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
