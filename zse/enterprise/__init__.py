"""
ZSE Enterprise Module

Production-ready features for enterprise deployment.

Components:
- auth: Authentication and authorization (API key, OAuth2, SSO, RBAC)
- monitoring: Metrics, health checks, alerting (Prometheus)
- scaling: Load balancing, auto-scaling
- billing: Usage tracking, cost calculation
- audit: Request/response logging
- multitenancy: Team isolation, quotas

Note: These features are optional and only loaded when
running in enterprise mode (mode: enterprise in config).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.enterprise.auth import AuthManager
    from zse.enterprise.monitoring import MetricsCollector
    from zse.enterprise.scaling import LoadBalancer

__all__ = [
    "AuthManager",
    "MetricsCollector",
    "LoadBalancer",
]
