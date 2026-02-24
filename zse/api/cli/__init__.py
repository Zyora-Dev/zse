"""
ZSE CLI

Command-line interface for ZSE.

Commands:
- zse serve: Start inference server
- zse chat: Interactive chat session
- zse convert: Convert models to .zse format
- zse info: Show model information
- zse benchmark: Run benchmarks
- zse version: Show version info
"""

from zse.api.cli.main import app

__all__ = ["app"]
