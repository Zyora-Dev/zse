"""
zScheduler - Request Scheduling and Batching

Implements high-throughput request handling:
- Continuous batching (dynamic batch composition)
- Priority queues (premium vs standard requests)
- Preemption (pause low-priority when memory tight)
- Fair scheduling across multiple requests
- Memory-aware batch sizing

Throughput Optimization:
- Maximize GPU utilization
- Minimize time-to-first-token
- Balance latency vs throughput
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.core.zscheduler.scheduler import Scheduler
    from zse.core.zscheduler.batch import BatchManager
    from zse.core.zscheduler.request import Request, RequestStatus

__all__ = [
    "Scheduler",
    "BatchManager",
    "Request",
    "RequestStatus",
]
