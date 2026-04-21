"""ZStreamer — Unit tests for continuous batching engine.

Tests scheduling logic, queue priority, memory budgeting, and request lifecycle.
No GPU needed — uses mock model runner.
"""

import time
import pytest

from zse_engine.zstreamer.request import (
    InferenceRequest, GenerationParams, RequestOutput,
    RequestState, FinishReason,
)
from zse_engine.zstreamer.queue import RequestQueue
from zse_engine.zstreamer.memory_budget import MemoryBudget
from zse_engine.zstreamer.scheduler import Scheduler, SchedulerConfig, StepType


# ======================================================================
# Helpers
# ======================================================================

def make_request(
    request_id="req-1",
    prompt_len=10,
    max_tokens=50,
    priority=0,
    deadline_ms=None,
    temperature=1.0,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_tokens=list(range(prompt_len)),
        params=GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=[2],  # EOS = 2
        ),
        priority=priority,
        deadline_ms=deadline_ms,
    )


# ======================================================================
# Request Tests
# ======================================================================

class TestInferenceRequest:
    def test_basic_properties(self):
        req = make_request(prompt_len=20, max_tokens=100)
        assert req.prompt_len == 20
        assert req.num_generated == 0
        assert req.total_tokens == 20
        assert req.remaining_prefill == 20
        assert req.state == RequestState.WAITING
        assert not req.is_active
        assert not req.is_finished

    def test_state_transitions(self):
        req = make_request()
        assert req.state == RequestState.WAITING

        req.mark_prefilling()
        assert req.state == RequestState.PREFILLING
        assert req.is_active

        req.mark_decoding()
        assert req.state == RequestState.DECODING
        assert req.is_active

        req.mark_finished(FinishReason.LENGTH)
        assert req.state == RequestState.FINISHED
        assert req.is_finished
        assert req.finish_reason == FinishReason.LENGTH
        assert req.finish_time is not None

    def test_add_token(self):
        req = make_request()
        assert req.first_token_time is None

        req.add_token(100)
        assert req.num_generated == 1
        assert req.output_tokens == [100]
        assert req.first_token_time is not None
        assert 100 in req.past_tokens

        req.add_token(200)
        assert req.num_generated == 2

    def test_should_stop_max_tokens(self):
        req = make_request(max_tokens=3)
        assert req.should_stop() is None

        req.add_token(10)
        req.add_token(20)
        assert req.should_stop() is None

        req.add_token(30)
        assert req.should_stop() == FinishReason.LENGTH

    def test_should_stop_eos(self):
        req = make_request()
        req.add_token(10)
        assert req.should_stop() is None

        req.add_token(2)  # EOS token
        assert req.should_stop() == FinishReason.STOP

    def test_urgency_no_deadline(self):
        req = make_request(deadline_ms=None)
        assert req.urgency == float("inf")

    def test_urgency_with_deadline(self):
        req = make_request(deadline_ms=5000)
        # Just created, should have ~5000ms left
        assert 4000 < req.urgency < 5100

    def test_cancel(self):
        req = make_request()
        req.mark_cancelled()
        assert req.state == RequestState.CANCELLED
        assert req.finish_reason == FinishReason.CANCELLED

    def test_preempt(self):
        req = make_request()
        req.mark_decoding()
        req.mark_preempted()
        assert req.state == RequestState.PREEMPTED

    def test_streaming_callback(self):
        tokens_received = []
        req = make_request()
        req.on_token = lambda t: tokens_received.append(t)

        req.add_token(42)
        req.add_token(43)
        assert tokens_received == [42, 43]

    def test_chunked_prefill_tracking(self):
        req = make_request(prompt_len=1000)
        req.chunk_size = 512
        assert req.remaining_prefill == 1000

        req.prefill_offset = 512
        assert req.remaining_prefill == 488


class TestRequestOutput:
    def test_from_request(self):
        req = make_request()
        req.mark_prefilling()
        req.mark_decoding()
        req.add_token(10)
        req.add_token(20)
        req.mark_finished(FinishReason.LENGTH)

        output = RequestOutput.from_request(req)
        assert output.request_id == "req-1"
        assert output.output_tokens == [10, 20]
        assert output.finish_reason == FinishReason.LENGTH
        assert output.ttft_ms >= 0
        assert output.total_time_ms >= 0
        assert output.tokens_per_sec >= 0


# ======================================================================
# Queue Tests
# ======================================================================

class TestRequestQueue:
    def test_add_and_pop(self):
        q = RequestQueue()
        req = make_request("r1")
        assert q.add(req)
        assert q.num_waiting == 1

        batch = q.pop_prefill_batch(max_requests=1)
        assert len(batch) == 1
        assert batch[0].request_id == "r1"
        assert q.num_waiting == 0

    def test_priority_ordering(self):
        q = RequestQueue()
        q.add(make_request("low", priority=0))
        q.add(make_request("high", priority=10))
        q.add(make_request("mid", priority=5))

        batch = q.pop_prefill_batch(max_requests=3)
        # Higher priority first
        assert batch[0].request_id == "high"
        assert batch[1].request_id == "mid"
        assert batch[2].request_id == "low"

    def test_slo_urgency_ordering(self):
        q = RequestQueue()
        q.add(make_request("relaxed", deadline_ms=10000))
        q.add(make_request("urgent", deadline_ms=100))
        q.add(make_request("no_slo"))

        batch = q.pop_prefill_batch(max_requests=3)
        # Most urgent (lowest deadline remaining) first
        assert batch[0].request_id == "urgent"
        assert batch[1].request_id == "relaxed"
        assert batch[2].request_id == "no_slo"

    def test_max_queue_size(self):
        q = RequestQueue(max_queue_size=2)
        assert q.add(make_request("r1"))
        assert q.add(make_request("r2"))
        assert not q.add(make_request("r3"))  # Rejected — queue full
        assert q.num_waiting == 2

    def test_token_budget(self):
        q = RequestQueue()
        q.add(make_request("r1", prompt_len=100))
        q.add(make_request("r2", prompt_len=200))
        q.add(make_request("r3", prompt_len=300))

        # Budget of 250 tokens: should get r1 (100) + r2 (200) = 300 > 250
        # Actually: r1 fits (100 <= 250), r2 would exceed (100+200=300 > 250)
        batch = q.pop_prefill_batch(max_requests=3, max_tokens=250)
        # First always admitted, second checked against remaining budget
        assert len(batch) >= 1
        assert batch[0].request_id == "r1"

    def test_cancel(self):
        q = RequestQueue()
        q.add(make_request("r1"))
        q.add(make_request("r2"))

        cancelled = q.cancel("r1")
        assert cancelled is not None
        assert cancelled.state == RequestState.CANCELLED
        assert q.num_waiting == 1

    def test_cancel_nonexistent(self):
        q = RequestQueue()
        assert q.cancel("nope") is None

    def test_preempted_priority(self):
        q = RequestQueue()
        # Add a normal request
        q.add(make_request("new"))

        # Add a preempted request (should get priority)
        preempted = make_request("resumed")
        preempted.mark_decoding()
        q.add_preempted(preempted)

        batch = q.pop_prefill_batch(max_requests=2)
        # Preempted should come first
        assert batch[0].request_id == "resumed"

    def test_is_empty(self):
        q = RequestQueue()
        assert q.is_empty
        q.add(make_request("r1"))
        assert not q.is_empty

    def test_stats(self):
        q = RequestQueue(max_queue_size=10)
        q.add(make_request("r1", prompt_len=50))
        q.add(make_request("r2", prompt_len=100))

        s = q.stats()
        assert s["num_waiting"] == 2
        assert s["waiting_prompt_tokens"] == 150
        assert not s["queue_full"]


# ======================================================================
# Memory Budget Tests
# ======================================================================

class TestMemoryBudget:
    def test_basic_admission(self):
        budget = MemoryBudget(
            total_blocks=100,
            block_size_tokens=16,
            headroom_ratio=0.5,
            emergency_reserve_blocks=2,
        )
        assert budget.available_blocks == 98  # 100 - 2 emergency

        req = make_request(prompt_len=32, max_tokens=64)
        # Prompt: 32/16 = 2 blocks, headroom: 64*0.5/16 = 2 blocks → total 4
        assert budget.can_admit(req)

    def test_admission_rejected(self):
        budget = MemoryBudget(
            total_blocks=5,
            block_size_tokens=16,
            headroom_ratio=1.0,
            emergency_reserve_blocks=1,
        )
        # Available = 4 blocks
        # Request: prompt 100 tokens → 7 blocks + headroom 50*1.0 → 4 blocks = 11 blocks
        req = make_request(prompt_len=100, max_tokens=50)
        assert not budget.can_admit(req)

    def test_reserve_and_release(self):
        budget = MemoryBudget(
            total_blocks=50,
            block_size_tokens=16,
            headroom_ratio=0.5,
        )
        req = make_request("r1", prompt_len=32, max_tokens=64)

        before = budget.available_blocks
        budget.reserve(req)
        after = budget.available_blocks

        assert after < before
        assert budget.committed_blocks > 0

        budget.release("r1")
        assert budget.available_blocks == before

    def test_utilization(self):
        budget = MemoryBudget(total_blocks=100, block_size_tokens=16)
        assert budget.utilization == 0.0

        req = make_request(prompt_len=160, max_tokens=160)
        budget.reserve(req)
        assert budget.utilization > 0

    def test_decode_progress_shrinks_headroom(self):
        budget = MemoryBudget(
            total_blocks=100,
            block_size_tokens=16,
            headroom_ratio=1.0,
        )
        req = make_request("r1", prompt_len=16, max_tokens=32)
        budget.reserve(req)

        committed_before = budget.committed_blocks
        # Simulate generating 16 tokens (1 block worth)
        budget.update_decode_progress("r1", 16)
        committed_after = budget.committed_blocks

        # Headroom should have shrunk
        assert committed_after <= committed_before

    def test_can_admit_n(self):
        budget = MemoryBudget(
            total_blocks=20,
            block_size_tokens=16,
            headroom_ratio=0.0,
            emergency_reserve_blocks=0,
        )
        # Each request: 16 tokens = 1 block, no headroom
        reqs = [make_request(f"r{i}", prompt_len=16, max_tokens=10) for i in range(30)]
        admitted = budget.can_admit_n(reqs)
        assert admitted == 20  # Exactly 20 blocks available

    def test_preemption_candidates(self):
        budget = MemoryBudget(
            total_blocks=100,
            block_size_tokens=16,
            headroom_ratio=0.0,
        )
        # Reserve 3 requests with different sizes
        for i, plen in enumerate([16, 48, 32]):
            req = make_request(f"r{i}", prompt_len=plen)
            budget.reserve(req)

        # Need 5 blocks
        candidates = budget.preemption_candidates(5)
        assert len(candidates) >= 1
        # Should prefer the largest reservation first
        assert candidates[0] == "r1"  # 48/16 = 3 blocks


# ======================================================================
# Scheduler Tests
# ======================================================================

class TestScheduler:
    def _make_scheduler(self, total_blocks=100, block_size=16, **kwargs):
        config = SchedulerConfig(**kwargs)
        budget = MemoryBudget(
            total_blocks=total_blocks,
            block_size_tokens=block_size,
            headroom_ratio=0.5,
        )
        return Scheduler(config=config, memory_budget=budget)

    def test_idle_when_empty(self):
        sched = self._make_scheduler()
        output = sched.schedule_step()
        assert output.is_idle
        assert output.step_type == StepType.IDLE

    def test_prefill_single_request(self):
        sched = self._make_scheduler()
        req = make_request("r1", prompt_len=20)
        sched.add_request(req)

        output = sched.schedule_step()
        assert output.step_type == StepType.PREFILL
        assert len(output.prefill_requests) == 1
        assert output.prefill_requests[0].request_id == "r1"
        assert output.prefill_requests[0].state == RequestState.PREFILLING

    def test_decode_after_prefill_complete(self):
        sched = self._make_scheduler(decode_per_prefill=1)
        req = make_request("r1", prompt_len=10)
        sched.add_request(req)

        # Step 1: prefill
        output = sched.schedule_step()
        assert output.step_type == StepType.PREFILL

        # Simulate prefill complete + first token
        sched.on_prefill_complete(req)
        assert req.state == RequestState.DECODING
        sched.on_token_generated(req, 42)

        # Step 2: should decode
        output2 = sched.schedule_step()
        assert output2.step_type == StepType.DECODE
        assert len(output2.decode_requests) == 1

    def test_max_prefill_per_step(self):
        sched = self._make_scheduler(max_prefill_per_step=1)
        sched.add_request(make_request("r1"))
        sched.add_request(make_request("r2"))
        sched.add_request(make_request("r3"))

        output = sched.schedule_step()
        # Only 1 prefill per step
        assert len(output.prefill_requests) <= 1

    def test_chunked_prefill(self):
        sched = self._make_scheduler(prefill_chunk_size=50)
        req = make_request("r1", prompt_len=120)
        sched.add_request(req)

        output = sched.schedule_step()
        assert len(output.prefill_requests) == 1
        # Should chunk to 50 tokens
        assert output.prefill_chunk_sizes[0] == 50

    def test_admission_control_rejects(self):
        # Very small memory budget
        sched = self._make_scheduler(total_blocks=3, block_size=16)

        # This request needs more blocks than available
        req = make_request("r1", prompt_len=100, max_tokens=200)
        sched.add_request(req)

        output = sched.schedule_step()
        # Should not admit (put back in queue)
        assert len(output.prefill_requests) == 0

    def test_cancel_waiting_request(self):
        sched = self._make_scheduler()
        sched.add_request(make_request("r1"))
        assert sched.num_waiting == 1

        sched.cancel_request("r1")
        output = sched.schedule_step()
        assert output.is_idle  # Nothing left

    def test_cancel_active_request(self):
        sched = self._make_scheduler()
        req = make_request("r1")
        sched.add_request(req)

        # Prefill it
        output = sched.schedule_step()
        sched.on_prefill_complete(req)
        sched.on_token_generated(req, 10)

        # Cancel while decoding
        sched.cancel_request("r1")
        req.mark_cancelled()

        # Next step should collect it as finished
        output2 = sched.schedule_step()
        assert len(output2.finished_requests) == 1

    def test_finished_request_cleanup(self):
        sched = self._make_scheduler()
        req = make_request("r1", max_tokens=1)
        sched.add_request(req)

        # Prefill
        sched.schedule_step()
        sched.on_prefill_complete(req)
        sched.on_token_generated(req, 10)  # This triggers LENGTH finish

        assert req.is_finished

        # Next step collects finished
        output = sched.schedule_step()
        assert len(output.finished_requests) == 1
        assert sched.num_active == 0

    def test_decode_prefill_ratio(self):
        sched = self._make_scheduler(decode_per_prefill=3)

        # Add initial request and prefill it
        req1 = make_request("r1")
        sched.add_request(req1)
        sched.schedule_step()  # prefill r1
        sched.on_prefill_complete(req1)
        sched.on_token_generated(req1, 10)

        # Add more waiting requests
        sched.add_request(make_request("r2"))
        sched.add_request(make_request("r3"))

        # Should get decode steps before prefill (ratio = 3:1)
        types = []
        for _ in range(5):
            output = sched.schedule_step()
            types.append(output.step_type)
            # Keep generating tokens so r1 stays active
            if output.decode_requests:
                sched.on_token_generated(req1, 11)

        # Should have more decode than prefill steps
        decode_count = sum(1 for t in types if t in (StepType.DECODE, StepType.MIXED))
        assert decode_count >= 2

    def test_multiple_concurrent_decode(self):
        sched = self._make_scheduler(max_batch_seqs=10, decode_per_prefill=0)

        # Prefill 3 requests
        reqs = []
        for i in range(3):
            req = make_request(f"r{i}", prompt_len=8)
            sched.add_request(req)
            output = sched.schedule_step()
            sched.on_prefill_complete(req)
            sched.on_token_generated(req, 100 + i)
            reqs.append(req)

        # Decode step should batch all 3
        output = sched.schedule_step()
        assert output.step_type == StepType.DECODE
        assert len(output.decode_requests) == 3

    def test_stats(self):
        sched = self._make_scheduler()
        sched.add_request(make_request("r1"))
        sched.schedule_step()

        s = sched.stats()
        assert s["total_steps"] == 1
        assert "queue" in s
        assert "memory" in s

    def test_summary(self):
        sched = self._make_scheduler()
        text = sched.summary()
        assert "ZStreamer Scheduler" in text
        assert "Active:" in text


# ======================================================================
# Integration: Scheduler + Queue + MemoryBudget
# ======================================================================

class TestSchedulerIntegration:
    def test_10_requests_priority_order(self):
        """Submit 10 requests with different priorities — verify ordering."""
        config = SchedulerConfig(
            max_batch_tokens=8192,
            max_prefill_per_step=1,
            decode_per_prefill=0,  # Always try prefill
        )
        budget = MemoryBudget(
            total_blocks=200,
            block_size_tokens=16,
            headroom_ratio=0.3,
        )
        sched = Scheduler(config=config, memory_budget=budget)

        # Submit 10 requests with priorities 0-9
        for i in range(10):
            sched.add_request(make_request(f"r{i}", priority=i, prompt_len=8))

        # Pop them one at a time
        prefill_order = []
        for _ in range(10):
            output = sched.schedule_step()
            if output.prefill_requests:
                prefill_order.append(output.prefill_requests[0].request_id)
                req = output.prefill_requests[0]
                sched.on_prefill_complete(req)
                sched.on_token_generated(req, 1)

        # Highest priority first
        assert prefill_order[0] == "r9"  # priority=9
        assert prefill_order[-1] == "r0"  # priority=0

    def test_memory_pressure_no_burst(self):
        """With limited memory, scheduler should NOT admit all at once."""
        config = SchedulerConfig(
            max_prefill_per_step=2,
            max_batch_seqs=10,
        )
        budget = MemoryBudget(
            total_blocks=10,
            block_size_tokens=16,
            headroom_ratio=0.5,
        )
        sched = Scheduler(config=config, memory_budget=budget)

        # Try to submit 20 requests — each needs ~2 blocks (16 prompt + headroom)
        for i in range(20):
            sched.add_request(make_request(f"r{i}", prompt_len=16, max_tokens=16))

        # First step: should admit only what memory allows
        output = sched.schedule_step()
        admitted = len(output.prefill_requests)
        assert admitted <= 5  # Can't admit all 20 with only 10 blocks

    def test_no_starvation(self):
        """Even low-priority requests should eventually get scheduled."""
        config = SchedulerConfig(
            max_prefill_per_step=1,
            decode_per_prefill=1,
        )
        budget = MemoryBudget(
            total_blocks=200,
            block_size_tokens=16,
            headroom_ratio=0.0,
        )
        sched = Scheduler(config=config, memory_budget=budget)

        # High priority request
        sched.add_request(make_request("high", priority=10, prompt_len=8, max_tokens=3))
        # Low priority request
        sched.add_request(make_request("low", priority=0, prompt_len=8, max_tokens=3))

        scheduled_ids = set()
        for _ in range(20):
            output = sched.schedule_step()
            for req in output.prefill_requests:
                scheduled_ids.add(req.request_id)
                sched.on_prefill_complete(req)
                sched.on_token_generated(req, 1)
            for req in output.decode_requests:
                sched.on_token_generated(req, 2)

        # Both should have been scheduled
        assert "high" in scheduled_ids
        assert "low" in scheduled_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
