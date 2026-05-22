"""End-to-end ZSE benchmark on AMD MI300X with v3 MFMA INT4 kernel wired in.

Measures: cold start, VRAM, single-seq throughput, concurrent N=4 throughput.
Run: python3 bench_zse_mi300x_v3.py /root/qwen2.5-32b.zse
"""
import os
import sys
import time
import json
import threading


def _gpu_vram_used_gb() -> float:
    """Read VRAM used via rocm-smi (simple, no torch dep)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["rocm-smi", "--showmemuse", "--json"], stderr=subprocess.DEVNULL
        ).decode()
        d = json.loads(out)
        for v in d.values():
            if isinstance(v, dict):
                for k, val in v.items():
                    if "memory use" in k.lower() and "bytes" in k.lower():
                        return float(val) / 1e9
        return -1.0
    except Exception:
        return -1.0


def main(model_path: str):
    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    print(f"=== ZSE E2E bench on MI300X with v3 MFMA INT4 ===")
    print(f"Model: {model_path}")
    print(f"VRAM before init: {_gpu_vram_used_gb():.2f} GB")

    t_cold0 = time.monotonic()
    cfg = SchedulerConfig(max_batch_seqs=8, max_seq_len=2048)
    engine = ZStreamerEngine(
        model_path=model_path,
        scheduler_config=cfg,
        max_seq_len=2048,
        quiet=False,
    )
    t_cold = time.monotonic() - t_cold0
    vram_after_init = _gpu_vram_used_gb()
    print(f"\n=== Cold start: {t_cold:.2f}s ===")
    print(f"VRAM after init: {vram_after_init:.2f} GB")

    # Start engine background loop
    eng_thread = threading.Thread(target=engine.run, daemon=True)
    eng_thread.start()
    time.sleep(0.2)

    PROMPT = "essay about history of AI"
    MAX_TOK = 100

    # ===== Single-seq throughput =====
    print(f"\n=== Single-seq throughput (max_tokens={MAX_TOK}) ===")
    done_evt = threading.Event()
    result_box = {}

    def on_finish(out):
        result_box["out"] = out
        done_evt.set()

    t0 = time.monotonic()
    rid = engine.add_request(
        prompt=PROMPT,
        max_tokens=MAX_TOK,
        temperature=0.0,
        on_finish=on_finish,
    )
    if rid is None:
        print("ERROR: request rejected")
        return
    done_evt.wait(timeout=120)
    t_single = time.monotonic() - t0
    out = result_box.get("out")
    if out is None:
        print("ERROR: timeout")
        return
    n_tokens = len(out.output_tokens)
    ttft_ms = (out.first_token_time - out.arrival_time) * 1000 if out.first_token_time else -1
    single_tps = n_tokens / t_single
    print(f"  generated {n_tokens} tokens in {t_single:.2f}s")
    print(f"  TTFT: {ttft_ms:.1f} ms")
    print(f"  throughput: {single_tps:.1f} tok/s")

    # ===== Concurrent N=4 throughput =====
    print(f"\n=== Concurrent N=4 throughput (max_tokens={MAX_TOK}) ===")
    prompts = [
        "essay about history of AI",
        "explain quantum computing",
        "write a story about a dragon",
        "describe how photosynthesis works",
    ]
    done_evts = [threading.Event() for _ in prompts]
    results = [None] * len(prompts)

    def make_cb(idx):
        def cb(out):
            results[idx] = out
            done_evts[idx].set()
        return cb

    t0 = time.monotonic()
    rids = []
    for i, p in enumerate(prompts):
        rid = engine.add_request(
            prompt=p, max_tokens=MAX_TOK, temperature=0.0, on_finish=make_cb(i),
        )
        rids.append(rid)
    for evt in done_evts:
        evt.wait(timeout=180)
    t_conc = time.monotonic() - t0
    total_tokens = sum(len(r.output_tokens) if r else 0 for r in results)
    conc_tps = total_tokens / t_conc
    print(f"  generated {total_tokens} total tokens in {t_conc:.2f}s")
    print(f"  aggregate throughput: {conc_tps:.1f} tok/s")
    for i, r in enumerate(results):
        if r:
            print(f"    seq {i}: {len(r.output_tokens)} tok")

    # ===== Summary =====
    print(f"\n=== SUMMARY (v3 MFMA wired) ===")
    print(f"  cold_start_s: {t_cold:.2f}")
    print(f"  vram_used_gb: {vram_after_init:.2f}")
    print(f"  single_tok_s: {single_tps:.1f}")
    print(f"  concurrent_n4_tok_s: {conc_tps:.1f}")

    result = {
        "cold_start_s": round(t_cold, 2),
        "vram_used_gb": round(vram_after_init, 2),
        "single_tok_s": round(single_tps, 1),
        "single_ttft_ms": round(ttft_ms, 1),
        "concurrent_n4_tok_s": round(conc_tps, 1),
        "concurrent_n4_total_tokens": total_tokens,
        "concurrent_n4_wall_s": round(t_conc, 2),
    }
    with open("/root/bench_zse_v3_mi300x.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote /root/bench_zse_v3_mi300x.json")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/qwen2.5-32b.zse"
    main(model_path)
