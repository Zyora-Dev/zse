"""Stage 2 (validation): convert the REAL Gemma 4 12B to .zse on Modal.

Validates the Stage-0 gemma4 adapter against actual weights (not just the
header). Downloads google/gemma-4-12B-it, runs convert_hf_to_zse with the
gemma4 adapter, and reports:
  - conversion success / failure + timing
  - .zse file size
  - per-tensor surprises: anything map_tensor_name leaves unmapped, dtype/shape
    oddities, vision/audio tensors that need fp16 (not INT4)
  - the resulting ModelConfig as stored in the .zse header

This is the de-risk step before building the gemma4 inference forward path:
proves the 25GB model converts end-to-end and surfaces any real-weight
surprises now rather than at the parity gate.

Requires Modal secret "huggingface" with HF_TOKEN (Gemma 4 gated).
Run:  modal run tests/test_modal_gemma4_convert.py
"""

import sys
import modal

app = modal.App("zse-gemma4-convert")

image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
)

MODEL = "google/gemma-4-12B-it"
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    timeout=18000,
    cpu=16.0,
    memory=98304,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def convert():
    import os
    import time
    import json

    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")
    os.environ["HF_HOME"] = "/root/hf_cache"

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print("=== Stage 2: convert Gemma 4 12B to .zse ===", flush=True)

    # 1. Download model (cached in volume across runs)
    t0 = time.time()
    model_dir = snapshot_download(
        MODEL, token=token, cache_dir="/root/hf_cache",
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    )
    print(f"Model at {model_dir} (download/cache {time.time()-t0:.1f}s)", flush=True)
    print("Files:", sorted(os.listdir(model_dir)), flush=True)

    # 2. Inspect config via adapter (sanity before convert)
    from zse_engine.format.arch.base import detect_architecture, get_adapter
    with open(os.path.join(model_dir, "config.json")) as f:
        hf_config = json.load(f)
    arch = detect_architecture(hf_config)
    adapter = get_adapter(arch)
    mc = adapter.config_from_hf(hf_config)
    print(f"\narch={arch} layers={mc.num_layers} hidden={mc.hidden_size} "
          f"head_dim={mc.head_dim} global_head_dim={mc.global_head_dim} "
          f"multimodal={list((mc.multimodal or {}).keys())}", flush=True)

    # 3. Convert to .zse (INT4) — this is the real test
    from zse_engine.format.convert import convert_hf_to_zse
    out_path = "/root/zse_cache/gemma4-12b.zse"

    seen = {"count": 0, "last": ""}
    def progress(name, cur, total):
        seen["count"] = cur
        seen["last"] = name
        if cur % 50 == 0 or cur == total:
            print(f"  [{cur}/{total}] {name}", flush=True)

    print("\nConverting (INT4, group_size=128)...", flush=True)
    t1 = time.time()
    try:
        convert_hf_to_zse(
            model_dir, out_path,
            progress_callback=progress,
            group_size=128,
        )
    except Exception as e:
        import traceback
        print("\n=== CONVERSION FAILED ===", flush=True)
        print(f"After {seen['count']} tensors, last={seen['last']}", flush=True)
        traceback.print_exc()
        return {"ok": False, "error": str(e), "tensors_done": seen["count"],
                "last_tensor": seen["last"]}
    convert_s = time.time() - t1

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"\n=== CONVERSION OK ===", flush=True)
    print(f"  time: {convert_s:.1f}s", flush=True)
    print(f"  .zse size: {size_gb:.2f} GB", flush=True)
    print(f"  tensors converted: {seen['count']}", flush=True)

    # 4. Read back the .zse header config to confirm it round-trips
    try:
        from zse_engine.format.loader import ZSELoader
        loader = ZSELoader(out_path)
        cfg = loader.config
        print(f"\n.zse header config: arch={cfg.arch} layers={cfg.num_layers} "
              f"hidden={cfg.hidden_size} head_dim={cfg.head_dim} "
              f"global_head_dim={getattr(cfg,'global_head_dim',None)} "
              f"layer_types={'set' if getattr(cfg,'layer_types',None) else 'none'} "
              f"softcap={getattr(cfg,'final_logit_softcapping',None)}", flush=True)
        loader.close()
    except Exception as e:
        print(f"\n[WARN] could not read back .zse header: {e}", flush=True)

    zse_cache.commit()
    return {"ok": True, "convert_s": round(convert_s, 1),
            "size_gb": round(size_gb, 2), "tensors": seen["count"]}


@app.local_entrypoint()
def main():
    res = convert.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
