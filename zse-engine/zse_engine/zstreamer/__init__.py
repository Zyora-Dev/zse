"""ZStreamer — Continuous Batching Engine for ZSE.

Efficient multi-request serving with:
- Disaggregated prefill/decode scheduling
- SLO-aware priority queue
- Predictive memory budgeting (no OOM bursts)
- Chunked prefill (long prompts don't block decode)
- Preemption under memory pressure
- Anti-burst admission control

Usage:
    from zse_engine.zstreamer import ZStreamerEngine, SchedulerConfig

    engine = ZStreamerEngine("model.zse")

    # Submit requests (non-blocking)
    req_id = engine.add_request("Hello world", max_tokens=50)

    # Manual stepping
    result = engine.step()

    # Or background loop
    import threading
    t = threading.Thread(target=engine.run)
    t.start()
    engine.add_request("Another prompt")
    engine.stop()
"""


def __getattr__(name):
    """Lazy imports to avoid cascading import failures."""
    if name == "ZStreamerEngine":
        from zse_engine.zstreamer.engine import ZStreamerEngine
        return ZStreamerEngine
    if name == "SchedulerConfig":
        from zse_engine.zstreamer.scheduler import SchedulerConfig
        return SchedulerConfig
    if name == "Scheduler":
        from zse_engine.zstreamer.scheduler import Scheduler
        return Scheduler
    if name == "MemoryBudget":
        from zse_engine.zstreamer.memory_budget import MemoryBudget
        return MemoryBudget
    if name == "RequestQueue":
        from zse_engine.zstreamer.queue import RequestQueue
        return RequestQueue
    if name == "BatchRunner":
        from zse_engine.zstreamer.batch_runner import BatchRunner
        return BatchRunner
    if name in ("InferenceRequest", "RequestOutput", "GenerationParams",
                "RequestState", "FinishReason"):
        import zse_engine.zstreamer.request as req_mod
        return getattr(req_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ZStreamerEngine",
    "SchedulerConfig",
    "Scheduler",
    "MemoryBudget",
    "RequestQueue",
    "BatchRunner",
    "InferenceRequest",
    "RequestOutput",
    "GenerationParams",
    "RequestState",
    "FinishReason",
]
