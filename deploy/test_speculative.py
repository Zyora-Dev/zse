"""
ZSE Speculative Decoding - Modal GPU Test

Tests speculative decoding end-to-end on a real GPU:
1. Loads target model (Qwen 7B) and draft model (Qwen 0.5B)
2. Generates with standard decoding (baseline)
3. Generates with speculative decoding
4. Compares speed, output quality, and acceptance rates

Usage:
    modal run deploy/test_speculative.py
"""

import modal
import os
import sys
import time

app = modal.App("zse-spec-test")

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

test_image_with_code = test_image.add_local_dir(
    ZSE_ROOT, remote_path="/root/zse"
)


@app.function(
    image=test_image_with_code,
    gpu="a10g",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_speculative_decoding():
    """Test speculative decoding on GPU with real models."""
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd="/root/zse",
    )
    if "/root/zse" not in sys.path:
        sys.path.insert(0, "/root/zse")

    import torch
    print(f"\n{'='*60}")
    print(f"ZSE Speculative Decoding Test")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    # ── Load Models ──────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    PROMPT = "Explain how black holes form in 3 sentences."
    MAX_TOKENS = 128

    print(f"Target: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
    print(f"Prompt: {PROMPT}")
    print(f"Max tokens: {MAX_TOKENS}\n")

    # Load tokenizer (shared — same tokenizer family)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)

    # Load target model (INT4 to fit in VRAM alongside draft)
    print("Loading target model (INT4)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    target_model.eval()

    target_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Target VRAM: {target_vram:.2f} GB")

    # Load draft model (FP16 — 0.5B is only ~1GB, and FP16 forward passes are
    # much faster than INT4/bnb which has high per-call dequant overhead)
    print("Loading draft model (FP16)...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    draft_model.eval()

    total_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Draft VRAM:  {total_vram - target_vram:.2f} GB")
    print(f"  Total VRAM:  {total_vram:.2f} GB\n")

    # Prepare input
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_len}\n")

    # ── Test 1: Standard Decoding (baseline) ────────────────────
    print(f"{'─'*60}")
    print("TEST 1: Standard Decoding (baseline)")
    print(f"{'─'*60}")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output_ids = target_model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,  # Greedy for deterministic comparison
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

    # ── Test 2: Speculative Decoding ────────────────────────────
    print(f"{'─'*60}")
    print("TEST 2: Speculative Decoding (draft=0.5B, target=7B)")
    print(f"{'─'*60}")

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

    # Generate
    torch.cuda.synchronize()
    start = time.perf_counter()

    all_tokens = []
    for step_output in decoder.generate(
        input_ids,
        max_tokens=MAX_TOKENS,
        temperature=0.0,  # Greedy
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

    # ── Test 3: Via SpeculativeTextGenerator (pipeline test) ────
    print(f"{'─'*60}")
    print("TEST 3: Full Pipeline (SpeculativeTextGenerator)")
    print(f"{'─'*60}")

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
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:     {baseline_tps:.1f} tok/s")
    print(f"  Speculative:  {spec_tps:.1f} tok/s ({spec_tps/baseline_tps:.2f}x)")
    print(f"  Pipeline:     {pipeline_tps:.1f} tok/s ({pipeline_tps/baseline_tps:.2f}x)")
    print(f"  Accept rate:  {stats['avg_acceptance_rate']:.1%}")
    print(f"  VRAM used:    {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"{'='*60}")

    passed = spec_tokens > 0 and spec_tps > 0
    status = "PASS" if passed else "FAIL"
    print(f"\n{'✅' if passed else '❌'} Speculative Decoding: {status}\n")

    return {
        "status": status,
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
    """Run speculative decoding test on Modal GPU."""
    print("🚀 Launching speculative decoding test on Modal GPU...")
    print("   Target: Qwen2.5-7B-Instruct (INT4)")
    print("   Draft:  Qwen2.5-0.5B-Instruct (INT4)")
    print()

    result = test_speculative_decoding.remote()

    print(f"\n{'='*60}")
    print("RESULT FROM MODAL:")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
