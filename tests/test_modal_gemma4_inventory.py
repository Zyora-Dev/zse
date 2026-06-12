"""Gemma 4 12B — Stage 0 tensor inventory on Modal.

Downloads ONLY the config + safetensors index (a few KB, not the 25 GB of
weights) for google/gemma-4-12B-it, then:

  1. Parses the config via the gemma4 adapter -> ModelConfig.
  2. Lists every real tensor name + shape from the safetensors index.
  3. Runs the adapter's map_tensor_name over each real name and reports:
       - text-backbone tensors (mapped to flat layout)
       - vision / audio modality tensors
       - any UNMAPPED / ambiguous names (these tell us what to fix before
         Stage-1 inference).

This is the ground-truth step: no blind coding. The exact prefixes the model
ships with drive the final map_tensor_name + the Stage-1 inference plan.

Requires a HuggingFace token with Gemma 4 access, set as a Modal secret:
    modal secret create huggingface HF_TOKEN=hf_xxx

Run:  modal run tests/test_modal_gemma4_inventory.py
"""

import sys
import modal

app = modal.App("zse-gemma4-inventory")

image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("huggingface_hub")
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
)

MODEL = "google/gemma-4-12B-it"


@app.function(
    image=image,
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface")],
)
def inventory():
    import os
    import json

    sys.path.insert(0, "/root/zse-engine")
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"=== Gemma 4 12B inventory ({MODEL}) ===", flush=True)

    # --- config.json ---
    cfg_path = hf_hub_download(MODEL, "config.json", token=token)
    with open(cfg_path) as f:
        hf_config = json.load(f)

    from zse_engine.format.arch.base import detect_architecture, get_adapter
    arch = detect_architecture(hf_config)
    adapter = get_adapter(arch)
    mc = adapter.config_from_hf(hf_config)
    print(f"\narch detected: {arch}", flush=True)
    print(f"layers={mc.num_layers} hidden={mc.hidden_size} heads={mc.num_heads} "
          f"kv={mc.num_kv_heads} head_dim={mc.head_dim} "
          f"global_head_dim={mc.global_head_dim} global_kv={mc.global_num_kv_heads}", flush=True)
    print(f"rope sliding={mc.rope_theta} global={mc.global_rope_theta} "
          f"partial_rotary={mc.partial_rotary_factor}", flush=True)
    print(f"act={mc.hidden_activation} softcap={mc.final_logit_softcapping} "
          f"k_eq_v={mc.attention_k_eq_v} multimodal={list((mc.multimodal or {}).keys())}", flush=True)

    # --- safetensors index (tensor names + shapes, no weights) ---
    try:
        idx_path = hf_hub_download(MODEL, "model.safetensors.index.json", token=token)
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        names = sorted(weight_map.keys())
        shapes = {}
        print(f"\ntotal tensors in index: {len(names)}", flush=True)
    except Exception as e:
        print(f"\nNo index file (single-file model). Reading safetensors header "
              f"via HTTP range request...", flush=True)
        import struct
        import urllib.request
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(MODEL, "model.safetensors")
        # safetensors layout: [8-byte little-endian header length][JSON header][data]
        hdr = {"Authorization": f"Bearer {token}"} if token else {}
        # 1) fetch first 8 bytes -> header length
        req = urllib.request.Request(url, headers={**hdr, "Range": "bytes=0-7"})
        n_bytes = urllib.request.urlopen(req).read(8)
        header_len = struct.unpack("<Q", n_bytes)[0]
        # 2) fetch the JSON header
        req2 = urllib.request.Request(
            url, headers={**hdr, "Range": f"bytes=8-{8 + header_len - 1}"})
        header_json = urllib.request.urlopen(req2).read(header_len)
        meta = json.loads(header_json)
        meta.pop("__metadata__", None)
        names = sorted(meta.keys())
        shapes = {k: meta[k].get("shape") for k in names}
        print(f"header_len={header_len} bytes  total tensors: {len(names)}", flush=True)

    # --- classify via adapter mapping ---
    text_layer0, vision, audio, other, unmapped = [], [], [], [], []
    prefixes_seen = {}
    for hf_name in names:
        zse = adapter.map_tensor_name(hf_name)
        top = hf_name.split(".")[0:3]
        prefixes_seen[".".join(top)] = prefixes_seen.get(".".join(top), 0) + 1
        if zse.startswith("vision."):
            vision.append((hf_name, zse))
        elif zse.startswith("audio."):
            audio.append((hf_name, zse))
        elif zse.startswith("layers.0.") or zse in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
            text_layer0.append((hf_name, zse))
        elif zse.startswith("layers."):
            other.append((hf_name, zse))  # other layers (don't spam)
        else:
            unmapped.append((hf_name, zse))

    print("\n--- TOP-LEVEL PREFIX COUNTS ---", flush=True)
    for p, c in sorted(prefixes_seen.items(), key=lambda x: -x[1])[:25]:
        print(f"  {c:4d}  {p}", flush=True)

    print("\n--- TEXT BACKBONE (layer 0 + globals) ---", flush=True)
    for hf, zse in text_layer0:
        print(f"  {hf}  {shapes.get(hf)}\n      -> {zse}", flush=True)

    print(f"\n--- VISION tensors ({len(vision)}) ---", flush=True)
    for hf, zse in vision[:40]:
        print(f"  {hf}  {shapes.get(hf)}\n      -> {zse}", flush=True)

    print(f"\n--- AUDIO tensors ({len(audio)}) ---", flush=True)
    for hf, zse in audio[:40]:
        print(f"  {hf}  {shapes.get(hf)}\n      -> {zse}", flush=True)

    print(f"\n--- UNMAPPED / AMBIGUOUS ({len(unmapped)}) ---", flush=True)
    for hf, zse in unmapped[:60]:
        print(f"  {hf}  {shapes.get(hf)}\n      -> {zse}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  text(layer0+glob)={len(text_layer0)}  other-text-layers={len(other)}  "
          f"vision={len(vision)}  audio={len(audio)}  UNMAPPED={len(unmapped)}", flush=True)
    if unmapped:
        print("  ACTION: update map_tensor_name for the unmapped prefixes above.", flush=True)
    else:
        print("  All tensors mapped. Ready for Stage-1 inference planning.", flush=True)

    return {
        "arch": arch,
        "num_tensors": len(names),
        "vision": len(vision),
        "audio": len(audio),
        "unmapped": len(unmapped),
        "unmapped_names": [h for h, _ in unmapped[:60]],
    }


@app.local_entrypoint()
def main():
    res = inventory.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
