"""
Tests for ZSE KV Cache and Continuous Batching

Tests cover:
- Simple KV cache operations
- Paged KV cache allocation/deallocation  
- Continuous batching scheduler
- Request queue management
"""

import pytest
import torch
import time
from typing import List

from zse.engine.kv_cache import (
    KVCache,
    KVCacheConfig,
    KVCacheManager,
    PagedKVCache,
)
from zse.engine.scheduler import (
    InferenceRequest,
    RequestStatus,
    GenerationConfig,
    SchedulerConfig,
    RequestQueue,
    SequenceGroup,
    ContinuousBatchingScheduler,
)


class TestKVCacheConfig:
    """Tests for KV cache configuration."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = KVCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.max_seq_len == 2048
        assert config.dtype == torch.float16
    
    def test_bytes_per_token(self):
        """Test memory calculation per token."""
        config = KVCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            dtype=torch.float16,
        )
        # K + V, all layers, all heads, head_dim, 2 bytes
        expected = 2 * 32 * 32 * 128 * 2
        assert config.bytes_per_token == expected
    
    def test_memory_estimation(self):
        """Test memory estimation."""
        config = KVCacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        # For 2048 tokens
        mem_gb = config.estimate_memory_gb(2048)
        # Should be ~1GB for 32 layer model
        assert 0.5 < mem_gb < 2.0


class TestSimpleKVCache:
    """Tests for simple KV cache."""
    
    def test_creation(self):
        """Test KV cache creation."""
        config = KVCacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=256,
            device="cpu",
        )
        cache = KVCache(config, batch_size=2)
        
        assert len(cache.key_cache) == 4
        assert len(cache.value_cache) == 4
        assert cache.seq_len == 0
    
    def test_update(self):
        """Test updating cache."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=128,
            device="cpu",
        )
        cache = KVCache(config, batch_size=1)
        
        # Add first token
        key = torch.randn(1, 4, 1, 32)
        value = torch.randn(1, 4, 1, 32)
        
        full_k, full_v = cache.update(0, key, value)
        cache.advance(1)
        
        assert full_k.shape == (1, 4, 1, 32)
        assert cache.seq_len == 1
        
        # Add more tokens
        key2 = torch.randn(1, 4, 3, 32)
        value2 = torch.randn(1, 4, 3, 32)
        
        full_k, full_v = cache.update(0, key2, value2)
        cache.advance(3)
        
        assert full_k.shape == (1, 4, 4, 32)
        assert cache.seq_len == 4
    
    def test_reset(self):
        """Test cache reset."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=64,
            device="cpu",
        )
        cache = KVCache(config, batch_size=1)
        
        key = torch.randn(1, 4, 10, 32)
        value = torch.randn(1, 4, 10, 32)
        cache.update(0, key, value)
        cache.advance(10)
        
        assert cache.seq_len == 10
        
        cache.reset()
        assert cache.seq_len == 0
    
    def test_memory_usage(self):
        """Test memory tracking."""
        config = KVCacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=256,
            device="cpu",
        )
        cache = KVCache(config, batch_size=1)
        
        mem_bytes = cache.memory_bytes()
        # 2 (K+V) * 4 layers * 1 batch * 8 heads * 256 seq * 64 dim * 2 bytes
        expected = 2 * 4 * 1 * 8 * 256 * 64 * 2
        assert mem_bytes == expected


class TestPagedKVCache:
    """Tests for paged KV cache."""
    
    def test_creation(self):
        """Test paged cache creation."""
        config = KVCacheConfig(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            page_size=16,
            max_pages=256,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        assert cache.num_free_pages() == 256
        assert cache.page_size == 16
    
    def test_sequence_allocation(self):
        """Test allocating sequence."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=64,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        # Allocate sequence with 32 tokens (2 pages)
        success = cache.allocate_sequence(seq_id=0, initial_len=32)
        assert success
        assert cache.num_free_pages() == 62
        assert len(cache.seq_page_tables[0]) == 2
    
    def test_sequence_free(self):
        """Test freeing sequence."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=64,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        cache.allocate_sequence(0, 32)
        assert cache.num_free_pages() == 62
        
        cache.free_sequence(0)
        assert cache.num_free_pages() == 64
    
    def test_append_token(self):
        """Test appending tokens."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=64,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        cache.allocate_sequence(0, 10)
        
        # Append tokens
        key = torch.randn(4, 32)
        value = torch.randn(4, 32)
        
        for layer in range(2):
            success = cache.append_token(0, layer, key, value)
            assert success
        
        assert cache.seq_lengths[0] == 11
    
    def test_get_kv(self):
        """Test retrieving cached KV."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=64,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        cache.allocate_sequence(0, 0)
        
        # Add some tokens
        for i in range(5):
            key = torch.randn(4, 32)
            value = torch.randn(4, 32)
            cache.append_token(0, 0, key, value)
        
        # Retrieve
        keys, values = cache.get_kv(0, 0)
        assert keys.shape == (5, 4, 32)
        assert values.shape == (5, 4, 32)
    
    def test_copy_on_write(self):
        """Test copy-on-write for beam search."""
        config = KVCacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            page_size=16,
            max_pages=64,
            device="cpu",
        )
        cache = PagedKVCache(config)
        
        cache.allocate_sequence(0, 32)
        initial_free = cache.num_free_pages()
        
        # Copy sequence (should share pages)
        success = cache.copy_sequence(0, 1)
        assert success
        
        # Pages should be shared, so free count unchanged
        # (ref counts increased, but no new pages allocated)
        assert cache.num_free_pages() == initial_free
        
        # Both sequences should have same length
        assert cache.seq_lengths[0] == cache.seq_lengths[1]


class TestKVCacheManager:
    """Tests for KV cache manager."""
    
    def test_create_and_free(self):
        """Test creating and freeing caches."""
        manager = KVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu",
            use_paged=False,
        )
        
        seq_id = manager.create_cache(batch_size=1)
        assert seq_id == 0
        
        seq_id2 = manager.create_cache(batch_size=1)
        assert seq_id2 == 1
        
        manager.free_cache(seq_id)
        assert seq_id not in manager.simple_caches
    
    def test_paged_manager(self):
        """Test paged cache manager."""
        manager = KVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu",
            use_paged=True,
            page_size=16,
            max_pages=128,
        )
        
        seq_id = manager.create_cache(batch_size=1, prompt_len=32)
        assert seq_id == 0
        
        stats = manager.memory_stats()
        assert stats['used_pages'] > 0


class TestRequestQueue:
    """Tests for request queue."""
    
    def test_fcfs_ordering(self):
        """Test first-come-first-serve ordering."""
        queue = RequestQueue(policy="fcfs", max_size=100)
        
        # Add requests with different times
        for i in range(5):
            req = InferenceRequest(
                request_id=f"req_{i}",
                prompt_tokens=list(range(10)),
            )
            req.created_at = i  # Explicit ordering
            queue.add(req)
        
        # Should come out in order
        for i in range(5):
            req = queue.pop()
            assert req.request_id == f"req_{i}"
    
    def test_shortest_first_ordering(self):
        """Test shortest-job-first ordering."""
        queue = RequestQueue(policy="shortest_first", max_size=100)
        
        # Add requests with different lengths
        lengths = [50, 10, 30, 5, 20]
        for i, length in enumerate(lengths):
            req = InferenceRequest(
                request_id=f"req_{i}",
                prompt_tokens=list(range(length)),
            )
            queue.add(req)
        
        # Should come out shortest first
        expected_order = [3, 1, 4, 2, 0]  # Indices sorted by length
        for expected_idx in expected_order:
            req = queue.pop()
            assert req.request_id == f"req_{expected_idx}"
    
    def test_max_size(self):
        """Test queue max size."""
        queue = RequestQueue(policy="fcfs", max_size=5)
        
        for i in range(5):
            req = InferenceRequest(f"req_{i}", list(range(10)))
            assert queue.add(req)
        
        # Should reject 6th request
        req = InferenceRequest("req_6", list(range(10)))
        assert not queue.add(req)


class TestInferenceRequest:
    """Tests for inference request."""
    
    def test_creation(self):
        """Test request creation."""
        req = InferenceRequest(
            request_id="test_001",
            prompt_tokens=[1, 2, 3, 4, 5],
        )
        
        assert req.prompt_len == 5
        assert req.output_len == 0
        assert req.total_len == 5
        assert req.status == RequestStatus.PENDING
        assert not req.is_finished
    
    def test_generation_tracking(self):
        """Test tracking generated tokens."""
        req = InferenceRequest(
            request_id="test_001",
            prompt_tokens=[1, 2, 3],
        )
        
        req.generated_tokens = [10, 11, 12]
        
        assert req.output_len == 3
        assert req.total_len == 6
    
    def test_status_transitions(self):
        """Test status transitions."""
        req = InferenceRequest("test", [1, 2, 3])
        
        assert not req.is_finished
        
        req.status = RequestStatus.RUNNING
        assert not req.is_finished
        
        req.status = RequestStatus.COMPLETED
        assert req.is_finished


class TestSequenceGroup:
    """Tests for sequence group."""
    
    def test_prefill_mode(self):
        """Test prefill mode."""
        req = InferenceRequest("test", list(range(100)))
        sg = SequenceGroup(req)
        
        assert sg.is_prefill
        assert sg.prefill_remaining == 100
        
        # Get chunk
        tokens = sg.get_next_tokens(chunk_size=32)
        assert len(tokens) == 32
        assert tokens == list(range(32))
    
    def test_decode_mode(self):
        """Test decode mode after prefill."""
        req = InferenceRequest("test", list(range(10)))
        sg = SequenceGroup(req)
        
        # Simulate prefill completion
        sg.is_prefill = False
        sg.num_computed_tokens = 10
        req.generated_tokens = [100]
        
        tokens = sg.get_next_tokens()
        assert tokens == [100]


class TestContinuousBatchingScheduler:
    """Tests for continuous batching scheduler."""
    
    def setup_method(self):
        """Setup scheduler for tests."""
        self.kv_manager = KVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu",
            use_paged=True,
            page_size=16,
            max_pages=256,
        )
        
        self.scheduler = ContinuousBatchingScheduler(
            config=SchedulerConfig(
                max_batch_size=8,
                max_total_tokens=1024,
            ),
            kv_cache_manager=self.kv_manager,
        )
    
    def test_add_request(self):
        """Test adding requests."""
        req = InferenceRequest("test_1", list(range(50)))
        
        success = self.scheduler.add_request(req)
        assert success
        assert self.scheduler.num_waiting() == 1
    
    def test_schedule_prefill(self):
        """Test scheduling prefill."""
        req = InferenceRequest("test_1", list(range(50)))
        self.scheduler.add_request(req)
        
        prefill, decode = self.scheduler.schedule()
        
        assert len(prefill) == 1
        assert len(decode) == 0
        assert self.scheduler.num_running() == 1
        assert self.scheduler.num_waiting() == 0
    
    def test_schedule_multiple(self):
        """Test scheduling multiple requests."""
        for i in range(5):
            req = InferenceRequest(f"test_{i}", list(range(20)))
            self.scheduler.add_request(req)
        
        prefill, decode = self.scheduler.schedule()
        
        # All should be scheduled
        assert len(prefill) == 5
        assert self.scheduler.num_running() == 5
        assert self.scheduler.num_waiting() == 0
    
    def test_batch_size_limit(self):
        """Test batch size limiting."""
        # Try to add more than max_batch_size
        for i in range(15):
            req = InferenceRequest(f"test_{i}", list(range(20)))
            self.scheduler.add_request(req)
        
        prefill, _ = self.scheduler.schedule()
        
        # Should only schedule max_batch_size
        assert len(prefill) <= 8
        assert self.scheduler.num_running() <= 8
        assert self.scheduler.num_waiting() >= 7
    
    def test_completion(self):
        """Test request completion."""
        req = InferenceRequest("test_1", list(range(10)))
        req.generation_config.max_new_tokens = 5
        
        self.scheduler.add_request(req)
        self.scheduler.schedule()
        
        # Simulate generation
        for i in range(5):
            req.generated_tokens.append(100 + i)
        req.status = RequestStatus.COMPLETED
        
        # Next schedule should remove completed
        self.scheduler.schedule()
        
        assert self.scheduler.num_running() == 0
        assert "test_1" in self.scheduler.completed
    
    def test_update_after_forward(self):
        """Test updating sequences after forward pass."""
        req = InferenceRequest("test_1", list(range(10)))
        req.generation_config.max_new_tokens = 3
        req.generation_config.eos_token_id = 999
        
        self.scheduler.add_request(req)
        prefill, _ = self.scheduler.schedule()
        
        # Simulate prefill completion
        prefill[0].is_prefill = False
        prefill[0].num_computed_tokens = 10
        
        # Update with new tokens
        new_tokens = torch.tensor([100])
        self.scheduler.update_after_forward(prefill, new_tokens, eos_token_id=999)
        
        assert req.generated_tokens == [100]
        assert req.output_len == 1
    
    def test_eos_termination(self):
        """Test termination on EOS token."""
        req = InferenceRequest("test_1", list(range(10)))
        req.generation_config.max_new_tokens = 100
        
        self.scheduler.add_request(req)
        prefill, _ = self.scheduler.schedule()
        
        # Complete prefill
        prefill[0].is_prefill = False
        prefill[0].num_computed_tokens = 10
        
        # Generate EOS token
        new_tokens = torch.tensor([999])  # EOS
        self.scheduler.update_after_forward(prefill, new_tokens, eos_token_id=999)
        
        assert req.status == RequestStatus.COMPLETED
    
    def test_stats(self):
        """Test statistics tracking."""
        for i in range(3):
            req = InferenceRequest(f"test_{i}", list(range(10)))
            self.scheduler.add_request(req)
        
        self.scheduler.schedule()
        
        stats = self.scheduler.stats()
        
        assert stats["total_requests"] == 3
        assert stats["running"] == 3
        assert "kv_cache" in stats


class TestGenerationConfig:
    """Tests for generation configuration."""
    
    def test_defaults(self):
        """Test default config."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 128
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
    
    def test_custom_config(self):
        """Test custom config."""
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_k=40,
            stop_token_ids=[1, 2, 3],
        )
        
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.stop_token_ids == [1, 2, 3]
