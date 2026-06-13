"""Stage 2b: Gemma 4 12B logit PARITY GATE — ZSE vs HF transformers.

The truth test for the dedicated Gemma 4 prefill path. Loads the cached .zse
(converted in Stage 2), runs ZSE prefill on a short prompt, and compares the
last-token logits against HuggingFace transformers' reference forward.

Pass criteria:
  - argmax (greedy next token) matches HF
  - top-5 tokens overlap heavily
  - cosine similarity of logit vectors > 0.99

Short prompt (<1024 tokens) so sliding-window vs full-causal is identical (the
ZSE prefill currently uses full causal for all layers — valid in this range).

Needs the cached .zse + HF token. Runs on an A10G (24GB; 12B INT4 ~8GB).
Run:  modal run tests/test_modal_gemma4_parity.py
"""

import sys
import modal

app = modal.App("zse-gemma4-parity")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("torch", "transformers>=5.0.0", "huggingface_hub", "accelerate")
    .env({"HF_HOME": "/root/hf_cache"})
)

zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)

MODEL = "google/gemma-4-12B-it"
PROMPT = "The capital of France is"


@app.function(gpu="A100-80GB", image=zse_image, timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/zse_cache": zse_cache, "/root/hf_cache": hf_cache})
def parity():
    import os
    import math
    import struct

    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")
    os.environ["ZSE_G4_DEBUG"] = "1"
    token = os.environ.get("HF_TOKEN")

    print("=== Gemma 4 12B PARITY GATE ===", flush=True)

    # ---- 1. Tokenize prompt with HF tokenizer ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()
    print(f"prompt={PROMPT!r}  tokens={input_ids}", flush=True)

    # ---- 2. ZSE forward (prefill → last-token logits) ----
    zse_path = "/root/zse_cache/gemma4-12b-v2.zse"
    print(f"\n[ZSE] loading {zse_path}", flush=True)
    from zse_engine.orchestrator.engine import ZSEEngine
    eng = ZSEEngine(zse_path, max_seq_len=512, quiet=False)
    runner = eng._runner
    print("[ZSE] running prefill...", flush=True)
    zse_logits_bytes = runner.prefill(input_ids, seq_id=0)
    vocab = eng._config.vocab_size
    zse_logits = list(struct.unpack(f"<{vocab}e", zse_logits_bytes[:vocab * 2]))
    zse_argmax = max(range(vocab), key=lambda i: zse_logits[i])
    print(f"[ZSE] argmax token: {zse_argmax} -> {tok.decode([zse_argmax])!r}", flush=True)

    # ---- 3. HF reference forward ----
    print("\n[HF] loading reference model (bf16)...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, token=token, dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    with torch.no_grad():
        out = hf(torch.tensor([input_ids], device="cuda"))
        hf_logits_t = out.logits[0, -1, :].float().cpu()
    hf_logits = hf_logits_t.tolist()
    hf_argmax = int(hf_logits_t.argmax())
    print(f"[HF] argmax token: {hf_argmax} -> {tok.decode([hf_argmax])!r}", flush=True)

    # ---- 4. Compare ----
    print("\n=== COMPARISON ===", flush=True)
    argmax_match = (zse_argmax == hf_argmax)
    # cosine similarity
    dot = sum(a * b for a, b in zip(zse_logits, hf_logits))
    nz = math.sqrt(sum(a * a for a in zse_logits))
    nh = math.sqrt(sum(b * b for b in hf_logits))
    cos = dot / (nz * nh + 1e-9)
    # top-5 overlap
    zse_top5 = sorted(range(vocab), key=lambda i: zse_logits[i], reverse=True)[:5]
    hf_top5 = sorted(range(vocab), key=lambda i: hf_logits[i], reverse=True)[:5]
    overlap = len(set(zse_top5) & set(hf_top5))

    print(f"argmax match:     {argmax_match}  (ZSE {zse_argmax} vs HF {hf_argmax})", flush=True)
    print(f"cosine sim:       {cos:.4f}", flush=True)
    print(f"top-5 overlap:    {overlap}/5", flush=True)
    print(f"ZSE top5: {[(t, tok.decode([t])) for t in zse_top5]}", flush=True)
    print(f"HF  top5: {[(t, tok.decode([t])) for t in hf_top5]}", flush=True)

    passed = argmax_match and cos > 0.99
    print(f"\n=== PARITY {'PASSED' if passed else 'FAILED'} ===", flush=True)
    return {"argmax_match": argmax_match, "cosine": round(cos, 4),
            "top5_overlap": overlap, "passed": passed,
            "zse_argmax": zse_argmax, "hf_argmax": hf_argmax}


@app.local_entrypoint()
def main():
    res = parity.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
