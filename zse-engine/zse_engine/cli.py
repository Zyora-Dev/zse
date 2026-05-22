"""ZSE CLI — Unified command-line interface.

Usage:
    zse serve model.zse [--port 8000] [--host 0.0.0.0]
    zse convert ./hf-model output.zse [--quant int4]
    zse info
    zse version
    zse keys create [--name my-app]
    zse keys list
    zse keys revoke <id>
"""

import argparse
import asyncio
import os
import signal
import sys


def _cmd_serve(args):
    """Serve a .zse model with OpenAI-compatible API."""
    from zse_engine.server.app import ZSEServer

    if args.model and not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    if not args.admin_key:
        import secrets
        args.admin_key = f"sk-zse-admin-{secrets.token_hex(16)}"
        if not args.quiet:
            print(f"  Admin key: {args.admin_key}")
            print(f"  Save this — it won't be shown again.\n")

    server = ZSEServer(
        model_path=args.model,
        host=args.host,
        port=args.port,
        admin_key=args.admin_key,
        db_path=args.db_path,
        model_name=args.model_name,
        quiet=args.quiet,
        tp_size=getattr(args, 'tp_size', 1),
    )

    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        if not args.quiet:
            print("\n[ZSE] Shutting down...")
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


def _cmd_convert(args):
    """Convert HuggingFace model to .zse format."""
    from zse_engine.format.__main__ import main as convert_main
    # Re-inject args into sys.argv for the existing parser
    sys.argv = ["zse-convert", args.model_dir, args.output]
    if args.quant:
        sys.argv += ["--quant", args.quant]
    if args.group_size:
        sys.argv += ["--group-size", str(args.group_size)]
    if args.arch:
        sys.argv += ["--arch", args.arch]
    if args.no_tokenizer:
        sys.argv += ["--no-tokenizer"]
    if args.quiet:
        sys.argv += ["--quiet"]
    convert_main()


def _cmd_info(args):
    """Show GPU and device information."""
    print("ZSE Device Info")
    print("=" * 50)

    try:
        from zse_compiler.runtime.device import get_devices, detect_backend
        backend = detect_backend()
        print(f"  Backend:  {backend}")

        devices = get_devices(backend)
        if not devices:
            print(f"  Devices:  None found")
        for d in devices:
            print(f"\n  Device {d.index}: {d.name}")
            if hasattr(d, 'total_memory') and d.total_memory:
                print(f"    Memory: {d.total_memory // (1024*1024)} MB")
            if hasattr(d, 'compute_capability') and d.compute_capability:
                print(f"    Compute: {d.compute_capability}")
            print(f"    Warp:   {d.warp_size}")
    except Exception as e:
        print(f"  Error detecting devices: {e}")

    print()


def _cmd_version(args):
    """Show version information."""
    print("ZSE — Zero-dependency Server Engine")
    print(f"  zse-engine:   0.1.0")
    print(f"  zse-compiler: 0.1.0")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Platform:     {sys.platform}")


def _cmd_warm(args):
    """Pre-fault a .zse file into the OS page cache for faster cold start.

    Reads the file sequentially so the kernel populates page cache. The next
    `zse serve` on the same host hits the cache and skips network/disk reads
    for weight upload. Works on any storage backend (NFS, NVMe, S3 FUSE, etc).
    """
    import time

    path = args.model
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    size = os.path.getsize(path)
    chunk = 8 * 1024 * 1024  # 8 MB

    print(f"Warming {path} ({size / 1024**3:.2f} GB)...")
    t0 = time.monotonic()
    bytes_read = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            bytes_read += len(buf)
            if bytes_read % (256 * 1024 * 1024) < chunk:
                pct = bytes_read / size * 100
                print(f"  {bytes_read / 1024**3:.2f} / {size / 1024**3:.2f} GB ({pct:.0f}%)")

    elapsed = time.monotonic() - t0
    rate = bytes_read / elapsed / 1024**2 if elapsed > 0 else 0
    print(f"Done in {elapsed:.1f}s ({rate:.0f} MB/s). OS page cache primed.")


def _cmd_keys(args):
    """Manage API keys."""
    from zse_engine.server.database import ServerDatabase

    db_path = os.path.expanduser(args.db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = ServerDatabase(db_path)

    try:
        if args.keys_action == "create":
            name = args.name or "default"
            rpm = args.rate_limit_rpm or 60
            plaintext, key = db.create_key(name=name, rate_limit_rpm=rpm)
            print(f"Created API key:")
            print(f"  Key:    {plaintext}")
            print(f"  Name:   {key.name}")
            print(f"  ID:     {key.id}")
            print(f"\n  Save this key — it cannot be retrieved later.")

        elif args.keys_action == "list":
            keys = db.list_keys()
            if not keys:
                print("No API keys found.")
                return
            print(f"{'ID':>4}  {'Name':<20}  {'Prefix':<16}  {'Active':<7}  {'Requests':>8}")
            print("-" * 65)
            for k in keys:
                print(f"{k.id:>4}  {k.name:<20}  {k.key_prefix:<16}  "
                      f"{'yes' if k.is_active else 'no':<7}  {k.total_requests:>8}")

        elif args.keys_action == "revoke":
            if not args.key_id:
                print("Error: --id is required for revoke", file=sys.stderr)
                sys.exit(1)
            if db.revoke_key(args.key_id):
                print(f"Revoked key ID {args.key_id}")
            else:
                print(f"Key ID {args.key_id} not found", file=sys.stderr)
                sys.exit(1)
        else:
            print("Usage: zse keys {create|list|revoke}", file=sys.stderr)
            sys.exit(1)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="zse",
        description="ZSE — Zero-dependency Server Engine for LLM Inference",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Serve a .zse model")
    p_serve.add_argument("model", nargs="?", default=None,
                         help="Path to .zse model file (omit for test mode)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--admin-key", default=None,
                         help="Admin API key (auto-generated if omitted)")
    p_serve.add_argument("--db-path", default="~/.zse/server.db")
    p_serve.add_argument("--model-name", default="zse-model",
                         help="Model name in /v1/models")
    p_serve.add_argument("--quiet", "-q", action="store_true")
    p_serve.add_argument("--tp", "--tensor-parallel", type=int, default=1,
                         dest="tp_size",
                         help="Tensor parallelism degree (number of GPUs, default: 1)")

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert HuggingFace model to .zse")
    p_conv.add_argument("model_dir", help="HuggingFace model directory")
    p_conv.add_argument("output", help="Output .zse file path")
    p_conv.add_argument("--quant", choices=["int4", "int8", "fp16"], default="int4")
    p_conv.add_argument("--group-size", type=int, default=128)
    p_conv.add_argument("--arch", default=None)
    p_conv.add_argument("--no-tokenizer", action="store_true")
    p_conv.add_argument("--quiet", "-q", action="store_true")

    # --- info ---
    sub.add_parser("info", help="Show GPU and device information")

    # --- version ---
    sub.add_parser("version", help="Show version")

    # --- warm ---
    p_warm = sub.add_parser("warm",
                            help="Pre-fault a .zse into OS page cache (faster cold start)")
    p_warm.add_argument("model", help="Path to .zse model file")

    # --- keys ---
    p_keys = sub.add_parser("keys", help="Manage API keys")
    p_keys.add_argument("keys_action", choices=["create", "list", "revoke"],
                        help="Key action")
    p_keys.add_argument("--name", default=None, help="Key name (for create)")
    p_keys.add_argument("--id", dest="key_id", type=int, help="Key ID (for revoke)")
    p_keys.add_argument("--rate-limit-rpm", type=int, default=60,
                        help="Rate limit requests/minute (for create)")
    p_keys.add_argument("--db-path", default="~/.zse/server.db")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "serve": _cmd_serve,
        "convert": _cmd_convert,
        "info": _cmd_info,
        "version": _cmd_version,
        "warm": _cmd_warm,
        "keys": _cmd_keys,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
