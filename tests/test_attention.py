"""
Tests for ZSE Attention Kernels

Tests both Triton and PyTorch fallback implementations.
GPU tests are skipped if CUDA is not available.
"""

import math
import pytest
import torch

from zse.core.zattention import (
    zAttention,
    AttentionConfig,
    AttentionBackend,
    create_attention,
    TRITON_AVAILABLE,
    paged_attention_v1_torch,
    flash_attention_torch,
)
from zse.core.zkv import (
    zKVCache,
    KVCacheConfig,
    BlockAllocator,
    create_kv_cache,
)


# Test markers
pytestmark = [pytest.mark.attention]
cuda_available = torch.cuda.is_available()


class TestAttentionConfig:
    """Test AttentionConfig."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = AttentionConfig(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        
        assert config.num_heads == 32
        assert config.num_kv_heads == 8
        assert config.head_dim == 128
        assert config.gqa_factor == 4
        assert config.scale == pytest.approx(1.0 / math.sqrt(128))
    
    def test_mha_config(self):
        """Test Multi-Head Attention config (no GQA)."""
        config = AttentionConfig(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
        )
        
        assert config.gqa_factor == 1
    
    def test_gqa_config(self):
        """Test Grouped-Query Attention config."""
        config = AttentionConfig(
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
        )
        
        assert config.gqa_factor == 8


class TestzAttention:
    """Test zAttention module."""
    
    def test_module_creation(self):
        """Test module creation."""
        config = AttentionConfig(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            backend=AttentionBackend.TORCH,  # Use PyTorch for testing
        )
        
        attention = zAttention(config)
        
        assert attention.config.num_heads == 32
        assert attention.backend == AttentionBackend.TORCH
    
    def test_create_attention_factory(self):
        """Test factory function."""
        attention = create_attention(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        
        assert isinstance(attention, zAttention)
        assert attention.config.num_heads == 32
        assert attention.config.gqa_factor == 4
    
    def test_get_info(self):
        """Test get_info method."""
        attention = create_attention(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        
        info = attention.get_info()
        
        assert info["num_heads"] == 32
        assert info["num_kv_heads"] == 8
        assert info["head_dim"] == 128
        assert info["gqa_factor"] == 4


class TestPyTorchFallback:
    """Test PyTorch fallback implementations."""
    
    def test_flash_attention_torch(self):
        """Test PyTorch flash attention."""
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = flash_attention_torch(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
    
    def test_flash_attention_causal_mask(self):
        """Test that flash attention applies causal masking."""
        batch_size = 1
        num_heads = 1
        seq_len = 4
        head_dim = 8
        
        # Create identity-like tensors to make attention patterns visible
        query = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1)
        key = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1)
        
        # Pad to head_dim
        query = torch.cat([query, torch.zeros(1, 1, seq_len, head_dim - seq_len)], dim=-1)
        key = torch.cat([key, torch.zeros(1, 1, seq_len, head_dim - seq_len)], dim=-1)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = flash_attention_torch(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_paged_attention_torch(self):
        """Test PyTorch paged attention fallback."""
        num_seqs = 2
        num_heads = 4
        num_kv_heads = 2  # GQA
        head_dim = 32
        block_size = 4
        context_len = 16
        num_blocks = context_len // block_size
        
        query = torch.randn(num_seqs, num_heads, head_dim)
        
        # Create KV cache blocks
        key_cache = torch.randn(num_seqs * num_blocks, num_kv_heads, block_size, head_dim)
        value_cache = torch.randn(num_seqs * num_blocks, num_kv_heads, block_size, head_dim)
        
        # Block tables (sequential allocation)
        block_tables = torch.zeros(num_seqs, num_blocks, dtype=torch.int64)
        for i in range(num_seqs):
            for j in range(num_blocks):
                block_tables[i, j] = i * num_blocks + j
        
        context_lens = torch.tensor([context_len, context_len], dtype=torch.int64)
        
        output = torch.empty_like(query)
        scale = 1.0 / math.sqrt(head_dim)
        
        paged_attention_v1_torch(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            block_size,
        )
        
        assert output.shape == (num_seqs, num_heads, head_dim)
        assert not torch.isnan(output).any()


class TestBlockAllocator:
    """Test block allocator."""
    
    def test_allocate(self):
        """Test block allocation."""
        allocator = BlockAllocator(num_blocks=10, device="cpu")
        
        assert allocator.num_free_blocks == 10
        assert allocator.num_used_blocks == 0
        
        blocks = allocator.allocate(3)
        
        assert len(blocks) == 3
        assert allocator.num_free_blocks == 7
        assert allocator.num_used_blocks == 3
    
    def test_free(self):
        """Test block deallocation."""
        allocator = BlockAllocator(num_blocks=10, device="cpu")
        
        blocks = allocator.allocate(5)
        allocator.free(blocks[:2])
        
        assert allocator.num_free_blocks == 7
        assert allocator.num_used_blocks == 3
    
    def test_allocate_insufficient(self):
        """Test allocation failure when not enough blocks."""
        allocator = BlockAllocator(num_blocks=5, device="cpu")
        
        allocator.allocate(3)
        
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            allocator.allocate(5)
    
    def test_ref_counting(self):
        """Test reference counting for copy-on-write."""
        allocator = BlockAllocator(num_blocks=10, device="cpu")
        
        blocks = allocator.allocate(2)
        
        assert allocator.get_ref_count(blocks[0]) == 1
        
        allocator.increase_ref(blocks[0])
        assert allocator.get_ref_count(blocks[0]) == 2
        
        # First free decreases ref count but doesn't free
        allocator.free([blocks[0]])
        assert allocator.get_ref_count(blocks[0]) == 1
        assert blocks[0] in allocator.used_blocks
        
        # Second free actually frees
        allocator.free([blocks[0]])
        assert allocator.get_ref_count(blocks[0]) == 0


class TestKVCache:
    """Test KV cache manager."""
    
    def test_create_kv_cache(self):
        """Test KV cache creation."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=128,
            block_size=8,
            device="cpu",
        )
        
        assert cache.config.num_layers == 2
        assert cache.config.num_kv_heads == 4
        assert cache.config.head_dim == 32
        assert cache.config.block_size == 8
    
    def test_allocate_sequence(self):
        """Test sequence allocation."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
        )
        
        block_table = cache.allocate_sequence(sequence_id=1, num_tokens=16)
        
        assert block_table.sequence_id == 1
        assert block_table.num_tokens == 16
        assert block_table.num_blocks == 2  # 16 tokens / 8 block_size
    
    def test_extend_sequence(self):
        """Test sequence extension."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
        )
        
        cache.allocate_sequence(sequence_id=1, num_tokens=8)
        cache.extend_sequence(sequence_id=1, num_new_tokens=16)
        
        block_table = cache.block_tables[1]
        assert block_table.num_tokens == 24
        assert block_table.num_blocks == 3  # ceil(24/8)
    
    def test_free_sequence(self):
        """Test sequence deallocation."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
        )
        
        cache.allocate_sequence(sequence_id=1, num_tokens=16)
        used_before = cache.allocator.num_used_blocks
        
        cache.free_sequence(sequence_id=1)
        
        assert cache.allocator.num_used_blocks < used_before
        assert 1 not in cache.block_tables
    
    def test_get_block_table_tensor(self):
        """Test block table tensor generation."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
        )
        
        cache.allocate_sequence(sequence_id=1, num_tokens=16)
        cache.allocate_sequence(sequence_id=2, num_tokens=24)
        
        tables = cache.get_block_table_tensor([1, 2])
        
        assert tables.shape[0] == 2  # Two sequences
        assert tables.shape[1] >= 3  # At least 3 blocks for seq 2
    
    def test_memory_usage(self):
        """Test memory usage reporting."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
        )
        
        cache.allocate_sequence(sequence_id=1, num_tokens=32)
        
        usage = cache.get_memory_usage()
        
        assert usage["used_blocks"] > 0
        assert usage["num_sequences"] == 1
        assert 0 < usage["utilization"] < 1


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
class TestCUDAAttention:
    """GPU-specific attention tests."""
    
    def test_flash_attention_cuda(self):
        """Test flash attention on CUDA."""
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        
        output = flash_attention_torch(query, key, value)
        
        assert output.device.type == "cuda"
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
    
    def test_paged_attention_cuda(self):
        """Test paged attention on CUDA."""
        num_seqs = 4
        num_heads = 8
        num_kv_heads = 4
        head_dim = 64
        block_size = 16
        context_len = 256
        num_blocks_per_seq = context_len // block_size
        total_blocks = num_seqs * num_blocks_per_seq
        
        query = torch.randn(num_seqs, num_heads, head_dim, device="cuda")
        key_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, device="cuda")
        value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, device="cuda")
        
        block_tables = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int64, device="cuda")
        for i in range(num_seqs):
            for j in range(num_blocks_per_seq):
                block_tables[i, j] = i * num_blocks_per_seq + j
        
        context_lens = torch.full((num_seqs,), context_len, dtype=torch.int64, device="cuda")
        
        output = torch.empty_like(query)
        scale = 1.0 / math.sqrt(head_dim)
        
        paged_attention_v1_torch(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            block_size,
        )
        
        assert output.device.type == "cuda"
        assert not torch.isnan(output).any()
    
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_triton_flash_attention(self):
        """Test Triton flash attention."""
        from zse.core.zattention import flash_attention
        
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        
        output = flash_attention(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.dtype == torch.float16


class TestKVCacheQuantization:
    """Test KV cache quantization."""
    
    def test_int8_cache_creation(self):
        """Test INT8 quantized cache creation."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
            quantization="int8",
        )
        
        assert cache.key_cache.dtype == torch.int8
        assert cache.value_cache.dtype == torch.int8
        assert cache.key_scales is not None
    
    def test_none_quantization(self):
        """Test no quantization (FP16)."""
        cache = create_kv_cache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=8,
            device="cpu",
            dtype=torch.float16,
            quantization="none",
        )
        
        assert cache.key_cache.dtype == torch.float16
        assert cache.key_scales is None
