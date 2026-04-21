"""ZSE Convert CLI — Convert HuggingFace models to .zse format.

Usage:
    python -m zse_engine.format /path/to/hf_model output.zse
    python -m zse_engine.format /path/to/hf_model output.zse --quant int8
    python -m zse_engine.format /path/to/hf_model output.zse --quant fp16
"""

import argparse
import os
import sys
import time

from zse_engine.format.convert import convert_hf_to_zse
from zse_engine.format.spec import QuantMethod


def _progress_bar(name: str, current: int, total: int, start_time: float):
    """Simple terminal progress display."""
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0
    pct = current * 100 // total
    bar_len = 30
    filled = bar_len * current // total
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    # Truncate tensor name to fit
    max_name = 40
    short = name if len(name) <= max_name else "..." + name[-(max_name - 3):]

    line = f"\r  [{bar}] {pct:3d}% ({current}/{total}) {elapsed:.0f}s eta {eta:.0f}s | {short}"
    sys.stdout.write(line.ljust(120))
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def main():
    parser = argparse.ArgumentParser(
        prog="zse-convert",
        description="Convert HuggingFace models to .zse format",
    )
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument("output", help="Output .zse file path")
    parser.add_argument(
        "--quant", choices=["int4", "int8", "fp16"], default="int4",
        help="Quantization method (default: int4)",
    )
    parser.add_argument(
        "--group-size", type=int, default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--arch", default=None,
        help="Override architecture detection (llama, qwen2, mistral, etc.)",
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true",
        help="Skip tokenizer embedding",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.isdir(args.model_dir):
        print(f"Error: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    config_json = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_json):
        print(f"Error: No config.json found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    # Map CLI quant option to QuantMethod
    quant_map = {
        "int4": QuantMethod.INT4_ASYM,
        "int8": QuantMethod.INT8_SYM,
        "fp16": QuantMethod.NONE,
    }
    quant_bits = {"int4": 4, "int8": 8, "fp16": 16}

    quant_method = quant_map[args.quant]
    bits = quant_bits[args.quant]

    if not args.quiet:
        print(f"ZSE Convert")
        print(f"  Input:  {args.model_dir}")
        print(f"  Output: {args.output}")
        print(f"  Quant:  {args.quant} (group_size={args.group_size})")
        print()

    start = time.time()

    # Progress callback
    cb = None
    if not args.quiet:
        def cb(name, cur, total):
            _progress_bar(name, cur, total, start)

    convert_hf_to_zse(
        args.model_dir,
        args.output,
        progress_callback=cb,
        quant_method=quant_method,
        quant_bits=bits,
        group_size=args.group_size,
        arch_override=args.arch,
        skip_tokenizer=args.no_tokenizer,
    )

    elapsed = time.time() - start
    size = os.path.getsize(args.output)
    if not args.quiet:
        print(f"\nDone in {elapsed:.1f}s — {size:,} bytes ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
