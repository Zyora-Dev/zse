"""ZSE Server — CLI entry point.

Usage:
    python -m zse_engine.server --model model.zse --port 8000 --admin-key mysecret

Options:
    --model PATH       Path to .zse model file (optional for test mode)
    --host HOST        Bind address (default: 0.0.0.0)
    --port PORT        Bind port (default: 8000)
    --admin-key KEY    Admin API key for key management
    --db-path PATH     SQLite database path (default: ~/.zse/server.db)
    --model-name NAME  Model name in /v1/models (default: zse-model)
    --quiet            Suppress output
"""

import argparse
import asyncio
import signal
import sys

from zse_engine.server.app import ZSEServer


def main():
    parser = argparse.ArgumentParser(
        description="ZSE Server — OpenAI-compatible LLM inference server"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .zse model file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Bind port (default: 8000)")
    parser.add_argument("--admin-key", type=str, default=None,
                        help="Admin API key for key management")
    parser.add_argument("--db-path", type=str, default="~/.zse/server.db",
                        help="SQLite database path")
    parser.add_argument("--model-name", type=str, default="zse-model",
                        help="Model name reported in /v1/models")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    if not args.admin_key:
        # Generate a temporary admin key
        import secrets
        args.admin_key = f"sk-zse-admin-{secrets.token_hex(16)}"
        if not args.quiet:
            print(f"[ZSE Server] Generated admin key: {args.admin_key}")
            print(f"  Save this key — it won't be shown again.\n")

    server = ZSEServer(
        model_path=args.model,
        host=args.host,
        port=args.port,
        admin_key=args.admin_key,
        db_path=args.db_path,
        model_name=args.model_name,
        quiet=args.quiet,
    )

    loop = asyncio.new_event_loop()

    # Handle Ctrl+C gracefully
    def shutdown(sig, frame):
        if not args.quiet:
            print("\n[ZSE Server] Shutting down...")
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(server.run_forever())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
