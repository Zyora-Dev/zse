"""ZSE — Qwen2.5-14B-Instruct on A100-80GB: pre-convert + cold start + VRAM.

Focused test (no throughput, no vLLM comparison):
  1. Pre-convert Qwen2.5-14B-Instruct to .zse INT4 (cached in volume)
  2. Cold start the engine from the pre-converted .zse
  3. Report: cold start seconds, VRAM weights / total used / GPU total

Run:
  modal run tests/test_modal_14b_coldstart.py                # full (convert if missing + cold start)
  modal run tests/test_modal_14b_coldstart.py::convert_only  # just convert
  modal run tests/test_modal_14b_coldstart.py::coldstart_only # just measure (needs cached .zse)
"""

import sys
import modal

app = modal.App("zse-14b-coldstart")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
ZSE_PATH = "/root/zse_cache/qwen2_14b.zse"


# ============================================================================
# Step 1: Pre-convert HF → .zse (idempotent, cached)
# ============================================================================

@app.function(
    gpu="A100-80GB",
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def convert_only():
    """Download + convert Qwen2.5-14B to .zse INT4. Skips if already cached."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import os
    import time

    print("=" * 70)
    print("ZSE PRE-CONVERT — Qwen2.5-14B-Instruct → INT4 .zse")
    print("=" * 70)

    # Reuse cached .zse if valid
    if os.path.exists(ZSE_PATH) and os.path.getsize(ZSE_PATH) > 1_000_000_000:
        with open(ZSE_PATH, "rb") as f:
            magic = f.read(4)
        if magic == b"ZSE\x01":
            size_gb = os.path.getsize(ZSE_PATH) / 1024**3
            print(f"[CACHED] {ZSE_PATH} ({size_gb:.2f} GB) — skipping convert")
            return {"cached": True, "zse_size_gb": round(size_gb, 2)}
        print("[CACHE] Corrupt magic — reconverting")
        os.remove(ZSE_PATH)

    progress_file = ZSE_PATH + ".progress"
    if os.path.exists(progress_file):
        os.remove(progress_file)

    from huggingface_hub import snapshot_download
    print(f"\n[1] Downloading {MODEL_ID} from HuggingFace...")
    t0 = time.time()
    hf_dir = snapshot_download(
        MODEL_ID,
        cache_dir="/root/hf_cache",
        allow_patterns=["*.safetensors", "*.json"],
    )
    print(f"     Downloaded in {time.time()-t0:.1f}s → {hf_dir}")

    print(f"\n[2] Converting → INT4 .zse ...")
    from zse_engine.format.convert import convert_hf_to_zse
    t0 = time.time()
    convert_hf_to_zse(hf_dir, ZSE_PATH)
    convert_s = time.time() - t0

    size_gb = os.path.getsize(ZSE_PATH) / 1024**3
    print(f"\n[OK] Converted in {convert_s:.1f}s")
    print(f"     .zse size: {size_gb:.2f} GB")

    # Commit volume so next run sees the file
    try:
        modal.Volume.lookup("zse-model-cache").commit()
    except Exception:
        pass

    return {"cached": False, "convert_s": round(convert_s, 1), "zse_size_gb": round(size_gb, 2)}


# ============================================================================
# Step 2: Cold start + VRAM measurement
# ============================================================================

@app.function(
    gpu="A100-80GB",
    image=zse_image,
    timeout=900,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def coldstart_only():
    """Measure cold start + VRAM on cached .zse (requires convert_only ran first).

    Runs TWO inits back-to-back in the same container:
      - Run 1 = "cold" Modal NFS (first read, no OS page cache)
      - Run 2 = "warm" Modal NFS (page cache hit — the real ZSE engine number)
    """
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time

    assert os.path.exists(ZSE_PATH), f".zse not found at {ZSE_PATH} — run convert_only first"
    size_gb = os.path.getsize(ZSE_PATH) / 1024**3

    print("=" * 70)
    print("ZSE COLD START — Qwen2.5-14B-Instruct (INT4) on A100-80GB")
    print("=" * 70)
    print(f"\n.zse file: {ZSE_PATH} ({size_gb:.2f} GB)")

    # Init CUDA context (matches production load path)
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    rc = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert rc == 0, f"cuCtxCreate failed: {rc}"

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    def _one_run(label):
        print(f"\n--- {label} ---")
        t0 = time.monotonic()
        eng = ZStreamerEngine(
            model_path=ZSE_PATH,
            scheduler_config=SchedulerConfig(max_batch_tokens=2048, max_batch_seqs=8),
            max_seq_len=512,
            quiet=False,
        )
        cold = time.monotonic() - t0
        print(f"[OK] {label}: {cold:.2f}s")
        # VRAM info
        plan = eng._vram_plan
        try:
            util = eng._allocator.utilization()
            used_gb = util["used_bytes"] / 1024**3
        except Exception:
            used_gb = (plan.weight_bytes + plan.kv_cache_bytes + plan.scratch_bytes) / 1024**3
        # Release for next run
        try:
            eng._weights.destroy(eng._gpu_mem)
        except Exception:
            pass
        del eng
        return cold, used_gb

    # Run 1 — cold Modal NFS (first read)
    cold1, vram1 = _one_run("RUN 1 (Modal NFS cold)")

    # Run 2 — warm OS page cache (file already in RAM from run 1)
    cold2, vram2 = _one_run("RUN 2 (warm page cache — real engine number)")

    results = {
        "model": MODEL_ID,
        "zse_size_gb": round(size_gb, 2),
        "cold_run1_s": round(cold1, 2),
        "cold_run2_warm_s": round(cold2, 2),
        "vram_used_gb": round(vram1, 2),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:<22} {v}")

    return results


# ============================================================================
# Local entrypoint: convert (if needed) → cold start
# ============================================================================

@app.local_entrypoint()
def main():
    print("[Step 1] Pre-converting Qwen2.5-14B → .zse INT4 (if not cached)...")
    convert_info = convert_only.remote()
    print(f"  → {convert_info}\n")

    print("[Step 2] Measuring cold start + VRAM...")
    results = coldstart_only.remote()
    print(f"\n[DONE]")
    print(f"  Cold start: {results['cold_start_s']}s")
    print(f"  VRAM used:  {results['vram_used_gb']} GB / {results['vram_total_gb']} GB")
    print(f"  Weights:    {results['vram_weights_gb']} GB (INT4)")
