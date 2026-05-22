"""ZSE vs vLLM Benchmark Suite — head-to-head on Modal.

Compares ZSE and vLLM on the same hardware + model with identical workloads.
Metrics: cold start, VRAM, TTFT, single-request throughput, batched throughput.

Usage:
    # Default: Qwen2.5-14B on A100-80GB
    modal run tests/test_modal_bench_vs_vllm.py

    # Override model + GPU via env vars
    BENCH_MODEL="Qwen/Qwen2.5-7B-Instruct" BENCH_GPU=A100 \\
        modal run tests/test_modal_bench_vs_vllm.py

    # Run only one side
    modal run tests/test_modal_bench_vs_vllm.py::run_zse
    modal run tests/test_modal_bench_vs_vllm.py::run_vllm
"""

import json
import os
import sys
import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-14B-Instruct")
GPU_TYPE = os.environ.get("BENCH_GPU", "A100-80GB")
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "128"))
MAX_SEQ_LEN = int(os.environ.get("BENCH_MAX_SEQ_LEN", "1024"))
CONCURRENT_N = int(os.environ.get("BENCH_CONCURRENT", "4"))
# vLLM quantization: "fp16" (default, native) or "awq" (INT4, for fair vs ZSE-INT4)
VLLM_QUANT = os.environ.get("BENCH_VLLM_QUANT", "fp16").lower()
# When using AWQ, vLLM needs the AWQ-quantized model variant. Override here if
# the base model has a different AWQ repo name.
VLLM_AWQ_MODEL = os.environ.get(
    "BENCH_VLLM_AWQ_MODEL",
    MODEL_ID + "-AWQ" if not MODEL_ID.endswith("-AWQ") else MODEL_ID,
)

PROMPT = (
    "Write a detailed essay about the history of artificial intelligence, "
    "covering its origins, major milestones, and future prospects."
)
CONCURRENT_PROMPTS = [
    "Write about solar energy.",
    "Write about wind energy.",
    "Write about nuclear energy.",
    "Write about hydro energy.",
    "Write about geothermal energy.",
    "Write about tidal energy.",
    "Write about biomass energy.",
    "Write about hydrogen fuel cells.",
]

app = modal.App("zse-vs-vllm-bench")

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm", "torch")
    .env({"BENCH_VLLM_QUANT": VLLM_QUANT, "BENCH_VLLM_AWQ_MODEL": VLLM_AWQ_MODEL,
          "BENCH_MODEL": MODEL_ID, "BENCH_GPU": GPU_TYPE,
          "BENCH_MAX_TOKENS": str(MAX_TOKENS), "BENCH_MAX_SEQ_LEN": str(MAX_SEQ_LEN),
          "BENCH_CONCURRENT": str(CONCURRENT_N)})
)

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
    .env({"BENCH_MODEL": MODEL_ID, "BENCH_GPU": GPU_TYPE,
          "BENCH_MAX_TOKENS": str(MAX_TOKENS), "BENCH_MAX_SEQ_LEN": str(MAX_SEQ_LEN),
          "BENCH_CONCURRENT": str(CONCURRENT_N)})
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# vLLM benchmark
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU_TYPE,
    image=vllm_image,
    timeout=1800,
    volumes={"/root/hf_cache": hf_cache},
)
def run_vllm() -> dict:
    """Benchmark vLLM: cold start, VRAM, single + concurrent throughput."""
    import time
    import torch

    os.environ["HF_HOME"] = "/root/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/root/hf_cache"

    # Resolve model + quant for this run
    use_awq = VLLM_QUANT == "awq"
    vllm_model = VLLM_AWQ_MODEL if use_awq else MODEL_ID
    quant_label = "INT4-AWQ" if use_awq else "FP16"

    out = {"engine": "vLLM", "model": vllm_model, "gpu": GPU_TYPE,
           "quant": quant_label}
    print(f"\n{'='*70}\nvLLM — {vllm_model} on {GPU_TYPE} ({quant_label})\n{'='*70}")

    # Capture baseline free VRAM BEFORE engine init (CUDA context already exists)
    torch.cuda.init()
    free_before, total_dev = torch.cuda.mem_get_info()

    # --- Cold start ---
    t_cold = time.monotonic()
    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=vllm_model,
        dtype="float16",
        max_model_len=MAX_SEQ_LEN,
        download_dir="/root/hf_cache",
        trust_remote_code=True,
        enforce_eager=True,
    )
    if use_awq:
        llm_kwargs["quantization"] = "awq"
    llm = LLM(**llm_kwargs)
    out["cold_start_s"] = round(time.monotonic() - t_cold, 2)
    print(f"  cold_start_s = {out['cold_start_s']}")

    # --- VRAM (device-level, sees subprocess allocations too) ---
    free_after, _ = torch.cuda.mem_get_info()
    used_bytes = max(0, free_before - free_after)
    # Fallback: if subprocess EngineCore makes free_after appear unchanged on
    # this process's view, query nvidia-smi via ctypes/NVML-free method.
    if used_bytes < 1 * 1024**3:  # <1GB is implausible for a 14B FP16 model
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            used_mib = int(r.stdout.strip().splitlines()[0])
            used_bytes = used_mib * 1024 * 1024
        except Exception:
            pass
    out["vram_used_gb"] = round(used_bytes / 1024**3, 2)
    out["vram_total_gb"] = round(total_dev / 1024**3, 1)
    print(f"  vram_used_gb = {out['vram_used_gb']} / {out['vram_total_gb']}")

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Warmup
    _ = llm.generate([PROMPT], sp)

    # --- Single-request throughput (avg of 3) ---
    times = []
    n_tok = 0
    for _ in range(3):
        t0 = time.monotonic()
        r = llm.generate([PROMPT], sp)
        times.append(time.monotonic() - t0)
        n_tok = len(r[0].outputs[0].token_ids)
    avg_t = sum(times) / len(times)
    out["single_tps"] = round(n_tok / avg_t, 1) if avg_t > 0 else 0
    out["single_ttft_ms"] = None  # vLLM doesn't expose per-iter TTFT easily here
    print(f"  single_tps = {out['single_tps']} ({n_tok} tok in {avg_t*1000:.0f}ms)")

    # --- Concurrent throughput ---
    prompts = CONCURRENT_PROMPTS[:CONCURRENT_N]
    sp_c = SamplingParams(temperature=0.0, max_tokens=100)
    _ = llm.generate(prompts, sp_c)  # warmup
    t0 = time.monotonic()
    outs = llm.generate(prompts, sp_c)
    t_c = time.monotonic() - t0
    total_c = sum(len(o.outputs[0].token_ids) for o in outs)
    out["concurrent_n"] = CONCURRENT_N
    out["concurrent_tps"] = round(total_c / t_c, 1) if t_c > 0 else 0
    print(f"  concurrent_tps = {out['concurrent_tps']} ({total_c} tok in {t_c*1000:.0f}ms)")

    print(f"\nvLLM result: {json.dumps(out, indent=2)}")
    return out


# ---------------------------------------------------------------------------
# ZSE benchmark
# ---------------------------------------------------------------------------

def _zse_path_for(model_id: str) -> str:
    safe = model_id.replace("/", "_").replace("-", "_").lower()
    return f"/root/zse_cache/{safe}.zse"


@app.function(
    gpu=GPU_TYPE,
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def run_zse() -> dict:
    """Benchmark ZSE: cold start, VRAM, single + concurrent throughput."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import struct
    import threading
    import time

    out = {"engine": "ZSE", "model": MODEL_ID, "gpu": GPU_TYPE}
    print(f"\n{'='*70}\nZSE — {MODEL_ID} on {GPU_TYPE} (INT4)\n{'='*70}")

    # --- Init CUDA context ---
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    rc = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert rc == 0, f"cuCtxCreate failed: {rc}"

    # --- Ensure .zse file exists ---
    zse_path = _zse_path_for(MODEL_ID)
    if not (os.path.exists(zse_path) and os.path.getsize(zse_path) > 10_000_000):
        from huggingface_hub import snapshot_download
        from zse_engine.format.convert import convert_hf_to_zse
        print(f"  converting {MODEL_ID} → {zse_path} ...")
        hf_dir = snapshot_download(
            MODEL_ID, cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        t0 = time.time()
        convert_hf_to_zse(hf_dir, zse_path)
        print(f"  converted in {time.time()-t0:.1f}s")
        try:
            modal.Volume.lookup("zse-model-cache").commit()
        except Exception:
            pass
    out["zse_size_gb"] = round(os.path.getsize(zse_path) / 1024**3, 2)
    print(f"  .zse size = {out['zse_size_gb']} GB")

    # --- Cold start ---
    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    t_cold = time.monotonic()
    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=max(2048, MAX_SEQ_LEN * 2),
            max_batch_seqs=max(8, CONCURRENT_N),
        ),
        max_seq_len=MAX_SEQ_LEN,
        quiet=False,
    )
    out["cold_start_s"] = round(time.monotonic() - t_cold, 2)
    print(f"  cold_start_s = {out['cold_start_s']}")

    # --- VRAM (device-level for apples-to-apples with vLLM) ---
    plan = engine._vram_plan
    # Query actual device free memory via CUDA driver (matches vLLM measurement)
    free_bytes = ctypes.c_size_t(0)
    total_bytes = ctypes.c_size_t(0)
    libcuda.cuMemGetInfo_v2(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
    device_used = total_bytes.value - free_bytes.value
    out["vram_weights_gb"] = round(plan.weight_bytes / 1024**3, 2)
    out["vram_kv_gb"] = round(plan.kv_cache_bytes / 1024**3, 2)
    out["vram_used_gb"] = round(device_used / 1024**3, 2)
    out["vram_total_gb"] = round(total_bytes.value / 1024**3, 1)
    print(f"  vram_used_gb = {out['vram_used_gb']} / {out['vram_total_gb']} "
          f"(weights={out['vram_weights_gb']}, kv={out['vram_kv_gb']})")

    tokenizer = engine._tokenizer

    # --- Helper: run a single request through the engine, return (tokens, elapsed, ttft) ---
    def _run_request(prompt: str, max_tok: int):
        tokens = []
        done = threading.Event()
        first = [None]

        def on_token(tid):
            if first[0] is None:
                first[0] = time.monotonic()
            tokens.append(tid)

        def on_finish(_):
            done.set()

        t0 = time.monotonic()
        engine.add_request(
            prompt=prompt, max_tokens=max_tok, temperature=0.0,
            on_token=on_token, on_finish=on_finish,
        )
        for _ in range(max_tok + 200):
            engine.step()
            if done.is_set():
                break
        elapsed = time.monotonic() - t0
        ttft_ms = (first[0] - t0) * 1000 if first[0] else 0
        return tokens, elapsed, ttft_ms

    # Warmup
    _run_request(PROMPT, 32)

    # --- Single-request throughput (avg of 3) ---
    tps_runs = []
    ttft_runs = []
    n_tok = 0
    for _ in range(3):
        toks, elapsed, ttft = _run_request(PROMPT, MAX_TOKENS)
        n_tok = len(toks)
        if elapsed > 0 and n_tok > 1:
            tps_runs.append(n_tok / elapsed)
            ttft_runs.append(ttft)
    out["single_tps"] = round(sum(tps_runs) / len(tps_runs), 1) if tps_runs else 0
    out["single_ttft_ms"] = round(sum(ttft_runs) / len(ttft_runs), 1) if ttft_runs else 0
    print(f"  single_tps = {out['single_tps']}, ttft_ms = {out['single_ttft_ms']}")

    # --- Concurrent throughput ---
    prompts = CONCURRENT_PROMPTS[:CONCURRENT_N]
    conc_tokens = [[] for _ in prompts]
    conc_done = [threading.Event() for _ in prompts]

    def _cb(idx):
        def on_t(tid): conc_tokens[idx].append(tid)
        def on_f(_): conc_done[idx].set()
        return on_t, on_f

    for i, p in enumerate(prompts):
        ot, of = _cb(i)
        engine.add_request(
            prompt=p, max_tokens=100, temperature=0.0,
            on_token=ot, on_finish=of,
        )
    t0 = time.monotonic()
    for _ in range(2000):
        engine.step()
        if all(e.is_set() for e in conc_done):
            break
    t_conc = time.monotonic() - t0
    total_conc = sum(len(t) for t in conc_tokens)
    out["concurrent_n"] = CONCURRENT_N
    out["concurrent_tps"] = round(total_conc / t_conc, 1) if t_conc > 0 else 0
    print(f"  concurrent_tps = {out['concurrent_tps']} "
          f"({total_conc} tok in {t_conc*1000:.0f}ms)")

    print(f"\nZSE result: {json.dumps(out, indent=2)}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_winner(a: float, b: float, higher_is_better: bool) -> str:
    """Return 'ZSE Nx' / 'vLLM Nx' / '=' string."""
    if a == 0 and b == 0:
        return "—"
    if higher_is_better:
        if a > b:
            return f"ZSE ({a/b:.2f}x)" if b > 0 else "ZSE (∞)"
        if b > a:
            return f"vLLM ({b/a:.2f}x)" if a > 0 else "vLLM (∞)"
        return "tie"
    else:
        if a < b:
            return f"ZSE ({b/a:.2f}x)" if a > 0 else "ZSE (∞)"
        if b < a:
            return f"vLLM ({a/b:.2f}x)" if b > 0 else "vLLM (∞)"
        return "tie"


def _print_table(z: dict, v: dict) -> None:
    rows = [
        ("Cold start (s)", z["cold_start_s"], v["cold_start_s"], False),
        ("VRAM used (GB)", z["vram_used_gb"], v["vram_used_gb"], False),
        ("Single tok/s", z["single_tps"], v["single_tps"], True),
        (f"Concurrent tok/s ({z.get('concurrent_n','?')} req)",
         z["concurrent_tps"], v["concurrent_tps"], True),
    ]
    print("\n" + "=" * 78)
    print(f"  ZSE vs vLLM — {z['model']} on {z['gpu']}")
    print("=" * 78)
    print(f"  {'Metric':<32} {'ZSE (INT4)':>14} {'vLLM (FP16)':>14} {'Winner':>14}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
    for label, zv, vv, hib in rows:
        print(f"  {label:<32} {zv:>14} {vv:>14} {_fmt_winner(zv, vv, hib):>14}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print(f"Benchmark: {MODEL_ID} on {GPU_TYPE}")
    print("Launching ZSE and vLLM in parallel...")
    z_handle = run_zse.spawn()
    v_handle = run_vllm.spawn()

    z = z_handle.get()
    v = v_handle.get()

    _print_table(z, v)

    # Persist a JSON report next to the script
    report = {"zse": z, "vllm": v}
    out_file = os.path.join(
        os.path.dirname(__file__),
        f"bench_{MODEL_ID.replace('/', '_')}_{GPU_TYPE}.json",
    )
    try:
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {out_file}")
    except Exception as e:
        print(f"(could not save report: {e})")
