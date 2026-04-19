"""
ZSE Speculative Decoding - .zse Format Test

Tests speculative decoding with pre-converted .zse format models
downloaded from the zse-zllm HuggingFace organization.

Usage:
    modal run deploy/test_speculative_zse.py
"""

import modal
import os
import sys
import time

app = modal.App("zse-spec-zse-test")

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

test_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.20.0",
        "bitsandbytes>=0.41.0",
        "pynvml",
        "rich",
        "typer",
        "sentencepiece",
        "protobuf",
        "psutil",
    )
)

test_image_with_code = test_image.add_local_dir(ZSE_ROOT, remote_path="/root/zse")


@app.function(
    image=test_image_with_code,
    gpu="a10g",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_speculative_zse():
    """Test speculative decoding with .zse format models."""
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd="/root/zse",
    )
    if "/root/zse" not in sys.path:
        sys.path.insert(0, "/root/zse")

    import torch
    from huggingface_hub import hf_hub_download

    print(f"\n{'=' * 60}")
    print(f"ZSE Speculative Decoding - .zse Format Test")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'=' * 60}\n")

    TARGET_REPO = "zse-zllm/Qwen2.5-7B-Instruct-zse-int4"
    TARGET_FILE = "Qwen2.5-7B-Instruct-zse-int4.zse"
    DRAFT_REPO = "zse-zllm/Qwen2.5-0.5B-Instruct-zse-int4"
    DRAFT_FILE = "Qwen2.5-0.5B-Instruct-zse-int4.zse"
    PROMPT = "Explain how black holes form in 3 sentences."
    MAX_TOKENS = 128

    # ── Download .zse files ─────────────────────────────────────
    print(f"Downloading target: {TARGET_REPO}")
    target_path = hf_hub_download(repo_id=TARGET_REPO, filename=TARGET_FILE)
    print(f"  Saved: {target_path}")
    print(f"  Size: {os.path.getsize(target_path) / 1024**3:.2f} GB")

    print(f"Downloading draft: {DRAFT_REPO}")
    draft_path = hf_hub_download(repo_id=DRAFT_REPO, filename=DRAFT_FILE)
    print(f"  Saved: {draft_path}")
    print(f"  Size: {os.path.getsize(draft_path) / 1024**3:.2f} GB")
    print()

    # ── Load from .zse format ───────────────────────────────────
    from zse.format.reader_v2 import load_zse_model

    print("Loading target model from .zse...")
    target_model, tokenizer, target_info = load_zse_model(target_path, device="cuda")
    target_model.eval()
    target_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Architecture: {target_info.get('architecture', 'unknown')}")
    print(f"  VRAM: {target_vram:.2f} GB")

    print("Loading draft model from .zse...")
    draft_model, draft_tokenizer, draft_info = load_zse_model(draft_path, device="cuda")
    draft_model.eval()
    total_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Architecture: {draft_info.get('architecture', 'unknown')}")
    print(f"  VRAM: {total_vram - target_vram:.2f} GB")
    print(f"  Total VRAM: {total_vram:.2f} GB")
    print()

    # ── Verify KV cache support ─────────────────────────────────
    print("Verifying .zse model supports use_cache=True...")
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {prompt_len}")

    with torch.no_grad():
        out = target_model(input_ids=input_ids, use_cache=True)
        assert hasattr(out, "logits"), "Model output missing .logits"
        assert hasattr(out, "past_key_values"), "Model output missing .past_key_values"
        assert out.past_key_values is not None, "past_key_values is None"
        print(f"  .logits shape: {out.logits.shape}")
        # Handle DynamicCache (transformers 5.x .layers, 4.x .key_cache, or legacy tuple)
        pv = out.past_key_values
        if hasattr(pv, "layers"):
            num_layers = len(pv.layers)
            kv_shape = pv.layers[0].keys.shape
        elif hasattr(pv, "key_cache"):
            num_layers = len(pv.key_cache)
            kv_shape = pv.key_cache[0].shape
        else:
            num_layers = len(pv)
            kv_shape = pv[0][0].shape
        print(f"  .past_key_values: {num_layers} layers")
        print(f"  KV shape per layer: {kv_shape}")
    print("  KV cache support: OK\n")

    # ── Test 1: Baseline (.zse model, standard decode) ──────────
    print(f"{'─' * 60}")
    print("TEST 1: Standard Decoding (.zse target model)")
    print(f"{'─' * 60}")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output_ids = target_model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - start
    baseline_tokens = output_ids.shape[1] - prompt_len
    baseline_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    baseline_tps = baseline_tokens / baseline_time

    print(f"  Tokens:  {baseline_tokens}")
    print(f"  Time:    {baseline_time:.2f}s")
    print(f"  Speed:   {baseline_tps:.1f} tok/s")
    print(f"  Output:  {baseline_text[:200]}...")
    print()

    # ── Test 2: Speculative (.zse target + .zse draft) ──────────
    print(f"{'─' * 60}")
    print("TEST 2: Speculative Decoding (.zse target + .zse draft)")
    print(f"{'─' * 60}")

    from zse.core.zspec import SpeculativeDecoder, SpeculativeConfig

    spec_config = SpeculativeConfig(
        num_speculative_tokens=5,
        acceptance_threshold=0.0,
        adaptation_window=50,
    )

    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        config=spec_config,
        target_device="cuda:0",
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    all_tokens = []
    for step_output in decoder.generate(
        input_ids,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop_token_id=tokenizer.eos_token_id,
    ):
        tokens = step_output.token_ids[0].tolist()
        all_tokens.extend(tokens)

    torch.cuda.synchronize()
    spec_time = time.perf_counter() - start
    spec_tokens = len(all_tokens)
    spec_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    spec_tps = spec_tokens / spec_time

    stats = decoder.get_stats()

    print(f"  Tokens:       {spec_tokens}")
    print(f"  Time:         {spec_time:.2f}s")
    print(f"  Speed:        {spec_tps:.1f} tok/s")
    print(f"  Speedup:      {spec_tps / baseline_tps:.2f}x vs baseline")
    print(f"  Accept rate:  {stats['avg_acceptance_rate']:.1%}")
    print(f"  Avg tok/step: {stats['avg_tokens_per_step']:.1f}")
    print(f"  Draft time:   {stats['draft_time_ms']:.0f}ms total")
    print(f"  Verify time:  {stats['verify_time_ms']:.0f}ms total")
    print(f"  Current K:    {stats['current_k']}")
    print(f"  Output:       {spec_text[:200]}...")
    print()

    # ── Test 3: Full Pipeline with .zse ─────────────────────────
    print(f"{'─' * 60}")
    print("TEST 3: Full Pipeline (SpeculativeTextGenerator + .zse)")
    print(f"{'─' * 60}")

    from zse.engine.generation import SpeculativeTextGenerator, SamplingParams

    spec_gen = SpeculativeTextGenerator(
        model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device="cuda:0",
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    chunks = []
    for chunk in spec_gen.generate_stream(
        prompt_text,
        SamplingParams(max_new_tokens=MAX_TOKENS, temperature=0.0),
    ):
        chunks.append(chunk)

    torch.cuda.synchronize()
    pipeline_time = time.perf_counter() - start
    pipeline_tokens = len(chunks)
    pipeline_text = "".join(c.text for c in chunks)
    pipeline_tps = pipeline_tokens / pipeline_time

    pipeline_stats = spec_gen.get_speculation_stats()

    print(f"  Tokens:       {pipeline_tokens}")
    print(f"  Time:         {pipeline_time:.2f}s")
    print(f"  Speed:        {pipeline_tps:.1f} tok/s")
    print(f"  Speedup:      {pipeline_tps / baseline_tps:.2f}x vs baseline")
    print(f"  Accept rate:  {pipeline_stats['avg_acceptance_rate']:.1%}")
    print(f"  Output:       {pipeline_text[:200]}...")
    print()

    # ── Summary ─────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("SUMMARY (.zse Format)")
    print(f"{'=' * 60}")
    print(f"  Target:       .zse INT4 (Qwen 7B)")
    print(f"  Draft:        .zse INT4 (Qwen 0.5B)")
    print(f"  Baseline:     {baseline_tps:.1f} tok/s")
    print(f"  Speculative:  {spec_tps:.1f} tok/s ({spec_tps / baseline_tps:.2f}x)")
    print(f"  Pipeline:     {pipeline_tps:.1f} tok/s ({pipeline_tps / baseline_tps:.2f}x)")
    print(f"  Accept rate:  {stats['avg_acceptance_rate']:.1%}")
    print(f"  VRAM used:    {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"{'=' * 60}")

    passed = spec_tokens > 0 and spec_tps > 0
    status = "PASS" if passed else "FAIL"
    print(f"\n{'✅' if passed else '❌'} .zse Speculative Decoding: {status}\n")

    return {
        "status": status,
        "format": ".zse",
        "baseline_tps": round(baseline_tps, 1),
        "speculative_tps": round(spec_tps, 1),
        "pipeline_tps": round(pipeline_tps, 1),
        "speedup": round(spec_tps / baseline_tps, 2),
        "acceptance_rate": round(stats["avg_acceptance_rate"], 3),
        "avg_tokens_per_step": round(stats["avg_tokens_per_step"], 1),
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }


@app.local_entrypoint()
def main():
    """Run .zse format speculative decoding test on Modal GPU."""
    print("🚀 Launching .zse format speculative decoding test...")
    print("   Target: zse-zllm/Qwen2.5-7B-Instruct-zse-int4 (.zse)")
    print("   Draft:  zse-zllm/Qwen2.5-0.5B-Instruct-zse-int4 (.zse)")
    print()

    result = test_speculative_zse.remote()

    print(f"\n{'=' * 60}")
    print("RESULT FROM MODAL:")
    print(f"{'=' * 60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
