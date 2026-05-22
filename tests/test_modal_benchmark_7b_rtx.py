"""ZSE vs vLLM — Qwen2.5-7B INT4 on consumer-class NVIDIA GPUs (T4, L4, A10G).

Three GPUs, one model (7B INT4 both sides), all six runs parallel on Modal.
- T4   16 GB  (~RTX 2080 Ti class)
- L4   24 GB  (Ada — closest to RTX 4090)
- A10G 24 GB  (Ampere — closest to RTX 3090)

vLLM uses the AWQ INT4 build of the same model so the comparison is apples-to-apples.
"""

import sys
import modal

app = modal.App("zse-vs-vllm-7b-rtx")

# ---------- Images ----------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm", "torch", "autoawq")
)

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine",  remote_path="/root/zse-engine",  copy=True)
    .pip_install("huggingface_hub")
)

hf_cache  = modal.Volume.from_name("zse-hf-cache",    create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

PROMPT     = "Write a detailed essay about the history of artificial intelligence, covering its origins, major milestones, and future prospects."
MAX_TOKENS = 128

MODEL_FP    = "Qwen/Qwen2.5-7B-Instruct"
MODEL_AWQ   = "Qwen/Qwen2.5-7B-Instruct-AWQ"


# ============================================================================
# vLLM (AWQ INT4)
# ============================================================================
def _bench_vllm(gpu_label: str):
    import time, os, torch
    os.environ["HF_HOME"] = "/root/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/root/hf_cache"

    results = {"engine": "vLLM-AWQ-INT4", "model": MODEL_AWQ, "gpu": gpu_label}
    print("=" * 70)
    print(f"vLLM (AWQ INT4) — Qwen2.5-7B on {gpu_label}")
    print("=" * 70)

    print("\n[1] Cold start...")
    t = time.monotonic()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_AWQ,
        quantization="awq",
        dtype="float16",
        max_model_len=2048,
        download_dir="/root/hf_cache",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )
    results["cold_start_s"] = round(time.monotonic() - t, 2)
    print(f"   {results['cold_start_s']}s")

    print("\n[2] VRAM...")
    results["vram_allocated_gb"] = round(torch.cuda.memory_allocated()/1024**3, 2)
    results["vram_reserved_gb"]  = round(torch.cuda.memory_reserved()/1024**3, 2)
    results["vram_total_gb"]     = round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)
    print(f"   allocated={results['vram_allocated_gb']} GB  reserved={results['vram_reserved_gb']} GB  total={results['vram_total_gb']} GB")

    print(f"\n[3] Single-seq throughput ({MAX_TOKENS} greedy)...")
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    _  = llm.generate([PROMPT], sp)               # warmup
    t  = time.monotonic()
    out = llm.generate([PROMPT], sp)
    dt = time.monotonic() - t
    n  = len(out[0].outputs[0].token_ids)
    results["num_tokens"] = n
    results["tps"]        = round(n / dt, 1) if dt > 0 else 0
    results["output"]     = out[0].outputs[0].text[:200]
    print(f"   {n} tokens / {dt:.2f}s = {results['tps']} tok/s")

    # avg of 3 more
    times = []
    for _ in range(3):
        t = time.monotonic()
        _ = llm.generate([PROMPT], sp)
        times.append(time.monotonic() - t)
    results["avg_tps"] = round(n / (sum(times)/len(times)), 1)
    print(f"   avg {results['avg_tps']} tok/s")

    print("\n[4] Concurrent N=4 throughput...")
    prompts = [
        "Write about solar energy.",
        "Write about wind energy.",
        "Write about nuclear energy.",
        "Write about hydro energy.",
    ]
    sp4 = SamplingParams(temperature=0.0, max_tokens=100)
    _   = llm.generate(prompts, sp4)
    t   = time.monotonic()
    out4 = llm.generate(prompts, sp4)
    dt  = time.monotonic() - t
    tot = sum(len(o.outputs[0].token_ids) for o in out4)
    results["concurrent_4_tps"] = round(tot/dt, 1) if dt > 0 else 0
    print(f"   {tot} tokens / {dt:.2f}s = {results['concurrent_4_tps']} tok/s")

    print("\nvLLM:", results)
    return results


# ============================================================================
# ZSE (INT4 .zse)
# ============================================================================
def _bench_zse(gpu_label: str):
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import os, time, threading, ctypes

    results = {"engine": "ZSE-INT4", "model": MODEL_FP, "gpu": gpu_label}
    print("=" * 70)
    print(f"ZSE (INT4) — Qwen2.5-7B on {gpu_label}")
    print("=" * 70)

    # CUDA init
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    assert libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0) == 0

    # ---- ensure cached .zse ----
    zse_path = "/root/zse_cache/qwen2_7b.zse"
    if not (os.path.exists(zse_path) and os.path.getsize(zse_path) > 1_000_000_000):
        progress = zse_path + ".progress"
        if os.path.exists(progress):
            os.remove(progress)
        from huggingface_hub import snapshot_download
        print("[0] Downloading + converting Qwen2.5-7B...")
        hf_dir = snapshot_download(
            MODEL_FP, cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        from zse_engine.format.convert import convert_hf_to_zse
        t = time.time()
        convert_hf_to_zse(hf_dir, zse_path)
        print(f"   converted in {time.time()-t:.1f}s")
        try:
            modal.Volume.lookup("zse-model-cache").commit()
        except Exception:
            pass
    else:
        print(f"[CACHE] using {os.path.getsize(zse_path)/1024**3:.2f} GB .zse")

    print("\n[1] Cold start...")
    t = time.monotonic()
    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig
    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(max_batch_tokens=2048, max_batch_seqs=8),
        max_seq_len=512,
        quiet=False,
    )
    results["cold_start_s"] = round(time.monotonic() - t, 2)
    print(f"   {results['cold_start_s']}s")

    print("\n[2] VRAM...")
    plan = engine._vram_plan
    try:
        used = engine._allocator.utilization()['used_bytes'] / 1024**3
    except Exception:
        used = (plan.weight_bytes + plan.kv_cache_bytes + plan.scratch_bytes) / 1024**3
    results["vram_weights_gb"] = round(plan.weight_bytes/1024**3, 2)
    results["vram_used_gb"]    = round(used, 2)
    results["vram_total_gb"]   = round(plan.total_vram/1024**3, 1)
    print(f"   weights={results['vram_weights_gb']} GB  total used={results['vram_used_gb']} GB  gpu={results['vram_total_gb']} GB")

    tokenizer = engine._tokenizer

    def run(prompt, max_toks):
        toks = []
        done = threading.Event()
        first = [None]
        def on_t(tid):
            if first[0] is None: first[0] = time.monotonic()
            toks.append(tid)
        def on_f(_): done.set()
        t0 = time.monotonic()
        engine.add_request(prompt=prompt, max_tokens=max_toks, temperature=0.0,
                           on_token=on_t, on_finish=on_f)
        for _ in range(max_toks + 200):
            engine.step()
            if done.is_set(): break
        return toks, time.monotonic() - t0, (first[0]-t0)*1000 if first[0] else 0

    print(f"\n[3] Single-seq throughput ({MAX_TOKENS} greedy)...")
    run(PROMPT, 32)                                 # warmup
    toks, dt, ttft = run(PROMPT, MAX_TOKENS)
    n = len(toks)
    results["num_tokens"] = n
    results["ttft_ms"]    = round(ttft, 1)
    results["tps"]        = round(n/dt, 1) if dt > 0 and n > 1 else 0
    try:
        results["output"] = tokenizer.decode(toks)[:200]
    except Exception:
        results["output"] = "?"
    print(f"   {n} tokens / {dt:.2f}s = {results['tps']} tok/s  TTFT={results['ttft_ms']}ms")

    # avg of 2 more
    tps_list = [results["tps"]]
    for _ in range(2):
        t2, d2, _ = run(PROMPT, MAX_TOKENS)
        n2 = len(t2)
        if d2 > 0 and n2 > 1:
            tps_list.append(n2/d2)
    results["avg_tps"] = round(sum(tps_list)/len(tps_list), 1)
    print(f"   avg {results['avg_tps']} tok/s")

    print("\n[4] Concurrent N=4 throughput...")
    prompts = [
        "Write about solar energy.",
        "Write about wind energy.",
        "Write about nuclear energy.",
        "Write about hydro energy.",
    ]
    conc_toks = [[] for _ in range(4)]
    conc_done = [threading.Event() for _ in range(4)]
    for i, p in enumerate(prompts):
        def mk(i):
            def on_t(tid): conc_toks[i].append(tid)
            def on_f(_):   conc_done[i].set()
            return on_t, on_f
        on_t, on_f = mk(i)
        engine.add_request(prompt=p, max_tokens=100, temperature=0.0,
                           on_token=on_t, on_finish=on_f)
    t = time.monotonic()
    for _ in range(800):
        engine.step()
        if all(e.is_set() for e in conc_done): break
    dt = time.monotonic() - t
    tot = sum(len(t) for t in conc_toks)
    results["concurrent_4_tps"] = round(tot/dt, 1) if dt > 0 else 0
    print(f"   {tot} tokens / {dt:.2f}s = {results['concurrent_4_tps']} tok/s")

    print("\nZSE:", results)
    return results


# ============================================================================
# Per-GPU wrappers (Modal needs decorated funcs)
# ============================================================================
@app.function(gpu="T4",   image=vllm_image, timeout=2400, volumes={"/root/hf_cache": hf_cache})
def vllm_t4():   return _bench_vllm("T4")

@app.function(gpu="L4",   image=vllm_image, timeout=2400, volumes={"/root/hf_cache": hf_cache})
def vllm_l4():   return _bench_vllm("L4")

@app.function(gpu="A10G", image=vllm_image, timeout=2400, volumes={"/root/hf_cache": hf_cache})
def vllm_a10g(): return _bench_vllm("A10G")

@app.function(gpu="T4",   image=zse_image,  timeout=3600,
              volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache})
def zse_t4():    return _bench_zse("T4")

@app.function(gpu="L4",   image=zse_image,  timeout=3600,
              volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache})
def zse_l4():    return _bench_zse("L4")

@app.function(gpu="A10G", image=zse_image,  timeout=3600,
              volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache})
def zse_a10g():  return _bench_zse("A10G")


# ============================================================================
# Entry point — fan out, collect, print matrix
# ============================================================================
@app.local_entrypoint()
def main():
    print("=" * 70)
    print("ZSE vs vLLM (both INT4) — Qwen2.5-7B on T4 / L4 / A10G")
    print("=" * 70)

    handles = {
        ("vLLM", "T4"):   vllm_t4.spawn(),
        ("vLLM", "L4"):   vllm_l4.spawn(),
        ("vLLM", "A10G"): vllm_a10g.spawn(),
        ("ZSE",  "T4"):   zse_t4.spawn(),
        ("ZSE",  "L4"):   zse_l4.spawn(),
        ("ZSE",  "A10G"): zse_a10g.spawn(),
    }

    results = {}
    for key, h in handles.items():
        engine, gpu = key
        print(f"\n>>> waiting on {engine} on {gpu} ...")
        try:
            results[key] = h.get()
            print(f"<<< {engine}/{gpu} done")
        except Exception as e:
            print(f"!!! {engine}/{gpu} FAILED: {type(e).__name__}: {e}")
            results[key] = {"engine": engine, "gpu": gpu, "error": str(e)}

    # ---------- matrix ----------
    print("\n" + "=" * 90)
    print("FINAL MATRIX — Qwen2.5-7B INT4")
    print("=" * 90)

    hdr = f"{'GPU':<8}{'Engine':<14}{'Cold(s)':<10}{'VRAM(GB)':<12}{'Single tps':<14}{'N=4 tps':<10}"
    print(hdr)
    print("-" * len(hdr))

    for gpu in ("T4", "L4", "A10G"):
        for engine in ("ZSE", "vLLM"):
            r = results.get((engine, gpu), {})
            if "error" in r:
                print(f"{gpu:<8}{engine:<14}ERROR: {r['error'][:50]}")
                continue
            cold = r.get("cold_start_s", "—")
            vram = r.get("vram_used_gb", r.get("vram_allocated_gb", "—"))
            tps  = r.get("avg_tps", r.get("tps", "—"))
            c4   = r.get("concurrent_4_tps", "—")
            print(f"{gpu:<8}{engine:<14}{str(cold):<10}{str(vram):<12}{str(tps):<14}{str(c4):<10}")
        # gap row
        z = results.get(("ZSE", gpu), {})
        v = results.get(("vLLM", gpu), {})
        if "error" not in z and "error" not in v:
            zc, vc = z.get("cold_start_s", 0), v.get("cold_start_s", 0)
            zv, vv = z.get("vram_used_gb", 0), v.get("vram_allocated_gb", 0) or v.get("vram_reserved_gb", 0)
            zt, vt = z.get("avg_tps", 0), v.get("avg_tps", 0)
            cold_x = f"{vc/zc:.1f}x" if zc else "—"
            vram_x = f"{vv/zv:.1f}x" if zv else "—"
            tps_x  = f"{zt/vt:.2f}x" if vt else "—"
            print(f"{'':<8}{'(ZSE/vLLM)':<14}{cold_x+' faster':<10}{vram_x+' less':<12}{tps_x:<14}")
        print()

    # save raw json
    import json, os
    out_path = "/tmp/bench_7b_rtx.json"
    with open(out_path, "w") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in results.items()}, f, indent=2)
    print(f"Raw JSON: {out_path}")
