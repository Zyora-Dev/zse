"""
ZSE Web UI

Web-based interface for ZSE.

Features:
- Playground: Interactive chat interface
- Model Manager: Load, unload, configure models
- Monitoring: Memory usage, throughput, active requests
- Settings: Server configuration

Developer Mode:
- Simple playground for testing

Enterprise Mode:
- Admin dashboard
- User management
- Usage analytics
- Billing integration
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.api.webui.app import create_webui_app

__all__ = ["create_webui_app"]
