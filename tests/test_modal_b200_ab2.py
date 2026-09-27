"""ZSE — VALID A/B of the medium-M fix on B200 (graph captured per-config).

The earlier A/B was invalid: CUDA graphs capture kernels ONCE, so flipping the
Python flag after capture replayed the same graph. Here each config gets its OWN
engine, and the flag is set BEFORE the graph is captured (first decode).

Run: modal run tests/test_modal_b200_ab2.py --batch-size 12
"""

import sys
import statistics
import time
import modal

app = modal.App("zse-b200-ab2")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

N = 12
MAX_TOKENS = 100


def measure_batch(engine, prompts, max_tokens, clock=time.monotonic,
                  temperature=0.0, repetition_penalty=1.0, output_tokens=None,
                  top_k=50, top_p=0.9):
    token_times = [[] for _ in prompts]
    submitted = [0.0 for _ in prompts]
    finished = [False for _ in prompts]
    failures = []

    def callbacks(index):
        def on_token(token_id):
            token_times[index].append(clock())
            if output_tokens is not None:
                output_tokens[index].append(token_id)

        def on_finish(output):
            finished[index] = True
            if output.finish_reason.name not in {"STOP", "LENGTH"}:
                failures.append(str(output.finish_reason))

        return on_token, on_finish

    started = clock()
    for index, prompt in enumerate(prompts):
        on_token, on_finish = callbacks(index)
        submitted[index] = clock()
        engine.add_request(prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                   repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p,
                           on_token=on_token, on_finish=on_finish)
    for _ in range(max_tokens + 300):
        result = engine.step()
        if result.errors:
            raise RuntimeError(f"Benchmark request errors: {result.errors}")
        if all(finished):
            break
    elapsed = clock() - started
    if not all(finished) or failures or not all(token_times):
        raise RuntimeError(f"Incomplete or failed benchmark: finished={finished}, errors={failures}")
    if elapsed <= 0:
        raise RuntimeError("Benchmark elapsed time must be positive")

    ttft = [times[0] - submitted[index] for index, times in enumerate(token_times)]
    intervals = [later - earlier for times in token_times
                 for earlier, later in zip(times, times[1:])]
    import math

    def p95(values):
        return sorted(values)[math.ceil(len(values) * 0.95) - 1] * 1000 if values else None

    return {
        "aggregate_tps": sum(map(len, token_times)) / elapsed,
        "ttft_median_ms": statistics.median(ttft) * 1000,
        "itl_median_ms": statistics.median(intervals) * 1000 if intervals else None,
        "ttft_p95_ms": p95(ttft),
        "itl_p95_ms": p95(intervals),
        "completed_requests": sum(finished),
        "failed_requests": len(failures),
        "generated_tokens": sum(map(len, token_times)),
        "elapsed_s": elapsed,
    }


def benchmark_overall(build_engine, prompts, memory_snapshot, persist,
                      runs=3, max_tokens=100):
    import hashlib
    import json

    results = {
        "configuration": dict(batch_sizes=sorted({1, min(4, len(prompts)), len(prompts)}),
                              runs=runs, max_tokens=max_tokens, top_k=50, top_p=0.9,
                              repetition_penalty=1.0, seed=20260927),
        "cases": [],
    }
    for concurrency in results["configuration"]["batch_sizes"]:
        for mode, temperature in (("greedy", 0.0), ("sampled", 0.8)):
            reference_hash = None
            for graph_enabled in (False, True):
                case = dict(batch_size=concurrency, mode=mode, graphs=graph_enabled,
                            status="running", runs=[])
                results["cases"].append(case)
                engine = None
                try:
                    case["memory_before_load"] = memory_snapshot(None)
                    started = time.monotonic()
                    engine = build_engine(False, max_prefill_per_step=concurrency)
                    case["engine_init_s"] = time.monotonic() - started
                    runner = engine._model_runner
                    if graph_enabled:
                        assert runner._graph_runner is not None, "CUDA graphs unavailable"
                    else:
                        runner.destroy_graph()
                        assert runner._graph_runner is None
                    case["memory_after_load"] = memory_snapshot(engine)
                    batch_prompts = prompts[:concurrency]
                    params = dict(temperature=temperature, repetition_penalty=1.0,
                                  top_k=50, top_p=0.9)
                    engine._sampler._rng.seed(20260927)
                    case["first_batch"] = measure_batch(engine, batch_prompts, 2, **params)
                    measure_batch(engine, batch_prompts, 16, **params)
                    if graph_enabled:
                        assert concurrency in runner._batched_graph_runners, "Requested graph not captured"
                    for trial in range(runs):
                        engine._sampler._rng.seed(20260927)
                        tokens = [[] for _ in batch_prompts]
                        metrics = measure_batch(engine, batch_prompts, max_tokens,
                                                output_tokens=tokens, **params)
                        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                        if not graph_enabled and reference_hash is None:
                            reference_hash = digest
                        metrics["token_sha256"] = digest
                        metrics["matches_graph_off"] = digest == reference_hash
                        metrics["full_length"] = metrics["generated_tokens"] == concurrency * max_tokens
                        case["runs"].append(metrics)
                        if trial == 0:
                            case["output_sample"] = engine._tokenizer.decode(tokens[0])
                        print(f"OVERALL N={concurrency} {mode} graphs={graph_enabled} trial={trial}: {metrics}",
                              flush=True)
                    case["memory_after_generation"] = memory_snapshot(engine)
                    case["captured_batch_sizes"] = sorted(runner._batched_graph_runners)
                    case["median"] = {
                        metric: statistics.median(sample[metric] for sample in case["runs"]
                                                  if sample[metric] is not None)
                        for metric in ("aggregate_tps", "ttft_median_ms", "itl_median_ms",
                                       "ttft_p95_ms", "itl_p95_ms")
                        if any(sample[metric] is not None for sample in case["runs"])
                    }
                    case["status"] = "completed"
                except Exception as error:
                    case["status"] = "failed"
                    case["error"] = f"{type(error).__name__}: {error}"
                    print(f"OVERALL FAILURE: {case['error']}", flush=True)
                finally:
                    if engine is not None:
                        engine.destroy()
                    persist(results)
                print(f"OVERALL CASE: {case}", flush=True)
    return results


class StageTimings:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.seconds = {}
        self.calls = {}
        self.originals = []

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            started = self.clock()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[stage] = self.seconds.get(stage, 0.0) + self.clock() - started
                self.calls[stage] = self.calls.get(stage, 0) + 1

        self.originals.append((owner, name, original))
        setattr(owner, name, timed)

    def reset(self):
        self.seconds.clear()
        self.calls.clear()

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()


def compare_samplers(build_engine, prompts, runs=3, max_tokens=24):
    results = {}
    reference_tokens = None
    for label in ("reference", "compact"):
        engine = build_engine(False, max_prefill_per_step=len(prompts))
        try:
            if label == "reference":
                engine._sampler._sample_top_k = lambda *args: None
            params = dict(temperature=0.8, repetition_penalty=1.0, top_k=50, top_p=0.9)
            measure_batch(engine, prompts, 2, **params)
            assert len(prompts) in engine._model_runner._batched_graph_runners
            samples = []
            for trial in range(runs):
                engine._sampler._rng.seed(20260927)
                tokens = [[] for _ in prompts]
                metrics = measure_batch(engine, prompts, max_tokens, output_tokens=tokens, **params)
                assert metrics["generated_tokens"] == len(prompts) * max_tokens
                if reference_tokens is None:
                    reference_tokens = tokens
                assert tokens == reference_tokens, f"Sampler parity failed: {label} trial={trial}"
                samples.append(metrics)
                print(f"SAMPLER {label} trial={trial}: {metrics}", flush=True)
            results[label] = {
                "runs": samples,
                "median_tps": statistics.median(sample["aggregate_tps"] for sample in samples),
            }
        finally:
            engine.destroy()
    results["token_parity"] = True
    results["configuration"] = dict(batch_size=len(prompts), max_tokens=max_tokens, runs=runs,
                                    temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.0)
    print(f"SAMPLER RESULT: {results}", flush=True)
    return results


def profile_sampling(build_engine, prompts, runs=3, max_tokens=24, profile_only=False):
    import cProfile
    import io
    import pstats

    results = {}
    modes = (("sampled", 0.8),) if profile_only else (("greedy", 0.0), ("sampled", 0.8))
    for label, temperature in modes:
        engine = build_engine(False, max_prefill_per_step=len(prompts))
        timings = StageTimings()
        try:
            runner = engine._model_runner
            assert runner._graph_runner is not None, "CUDA graphs unavailable"
            params = dict(temperature=temperature, repetition_penalty=1.0,
                          top_k=50, top_p=0.9)
            engine._sampler._rng.seed(20260927)
            reference = [[] for _ in prompts]
            measure_batch(engine, prompts, 2 if profile_only else max_tokens,
                          output_tokens=reference, **params)
            assert len(prompts) in runner._batched_graph_runners
            timings.wrap(runner, "batched_decode_graph", "graph_decode_inclusive")
            timings.wrap(runner, "_bulk_download_logits", "logits_download_and_slice")
            timings.wrap(engine._sampler, "sample", "sampling")
            for graph, stream in runner._batched_graph_runners.values():
                timings.wrap(graph, "replay", "graph_launch")
                timings.wrap(graph, "sync", "graph_wait")
            samples = []
            for trial in range(0 if profile_only else runs):
                engine._sampler._rng.seed(20260927)
                timings.reset()
                tokens = [[] for _ in prompts]
                metrics = measure_batch(engine, prompts, max_tokens, output_tokens=tokens, **params)
                assert tokens == reference, f"Instrumentation token mismatch: {label} trial {trial}"
                assert metrics["generated_tokens"] == len(prompts) * max_tokens, "Early termination"
                metrics["stage_seconds"] = dict(timings.seconds)
                metrics["stage_calls"] = dict(timings.calls)
                samples.append(metrics)
                print(f"PROFILE {label} trial={trial}: {metrics}", flush=True)
            results[label] = {
                "runs": samples,
                "median_tps": statistics.median(sample["aggregate_tps"] for sample in samples) if samples else None,
                "instrumentation_token_parity": bool(samples),
            }
            timings.restore()
            if label == "sampled":
                profiler = cProfile.Profile()
                engine._sampler._rng.seed(20260927)
                profiler.enable()
                try:
                    measure_batch(engine, prompts, 8, **params)
                finally:
                    profiler.disable()
                output = io.StringIO()
                pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(25)
                results["sampled_cprofile"] = output.getvalue()
                print(output.getvalue(), flush=True)
        finally:
            timings.restore()
            engine.destroy()
    results["configuration"] = dict(batch_size=len(prompts), max_tokens=max_tokens,
                                    runs=runs, top_k=50, top_p=0.9, repetition_penalty=1.0)
    results["timing_notes"] = (
        "Wall-clock attribution, not CUDA-event kernel timings. Graph decode includes launch, "
        "wait, metadata and download; do not sum overlapping stages. Sampling includes prefill. "
        "cProfile is a separate short run, excluded from throughput medians.")
    print(f"PROFILE RESULT: {results}", flush=True)
    return results


@app.function(
    gpu="B200", image=zse_image, timeout=2400,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def ab2(batch_size: int = N, validate_sampling: bool = False, sampling_profile: bool = False,
    profile_only: bool = False, sampler_ab: bool = False, model: str = "qwen2.5-14b",
    overall: bool = False, smoke_only: bool = False):
    import os
    import json
    from pathlib import Path

    if model not in {"qwen2.5-14b", "qwen2.5-7b", "llama3.1-8b"}:
        raise ValueError(f"Unsupported benchmark model: {model}")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    print("=" * 70, flush=True)
    print("ZSE B200 performance benchmark (per-config engines and graphs)", flush=True)
    print("=" * 70, flush=True)

    zse_path = "/root/zse_cache/qwen2_14b.zse"
    model_revision = None
    if model in {"llama3.1-8b", "qwen2.5-7b"}:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from zse_engine.format.convert import convert_hf_to_zse
        from zse_engine.format.header import FileHeader

        model_id = ("Qwen/Qwen2.5-7B-Instruct" if model == "qwen2.5-7b"
                    else "meta-llama/Llama-3.1-8B-Instruct")
        token = False if model == "qwen2.5-7b" else os.environ.get("HF_TOKEN")
        if model == "llama3.1-8b" and not token:
            raise RuntimeError("Modal huggingface secret must contain HF_TOKEN")
        model_revision = HfApi(token=token).model_info(model_id).sha
        hf_hub_download(model_id, "config.json", revision=model_revision,
                        cache_dir="/root/hf_cache", token=token)
        print(f"MODEL access OK: {model_id}@{model_revision}", flush=True)
        cache_name = "llama3_1_8b" if model == "llama3.1-8b" else model
        zse_path = f"/root/zse_cache/{cache_name}_{model_revision}_int4_rowmajor.zse"
        if not os.path.exists(zse_path):
            started = time.monotonic()
            hf_dir = snapshot_download(model_id, revision=model_revision, token=token,
                                       cache_dir="/root/hf_cache",
                                       allow_patterns=["*.safetensors", "*.json"])
            hf_cache.commit()
            print(f"MODEL downloaded in {time.monotonic() - started:.2f}s", flush=True)
            partial_path = zse_path + ".converting"
            convert_hf_to_zse(hf_dir, partial_path)
            os.replace(partial_path, zse_path)
            zse_cache.commit()
        with open(zse_path, "rb") as source:
            header = FileHeader.unpack(source.read(64))
        assert header.total_size == os.path.getsize(zse_path), "Truncated cached model"
        print(f"MODEL current-format INT4 ready: {os.path.getsize(zse_path)} bytes", flush=True)
    assert os.path.exists(zse_path), "cached .zse missing"

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    base_prompts = [f"Write a short essay number {index} about renewable energy."
                    for index in range(batch_size)]
    if model == "llama3.1-8b":
        base_prompts = [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for prompt in base_prompts
        ]
    elif model == "qwen2.5-7b":
        base_prompts = [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            for prompt in base_prompts
        ]

    def build_engine(disable_medium, max_prefill_per_step=2):
        eng = ZStreamerEngine(
            model_path=zse_path,
            scheduler_config=SchedulerConfig(max_batch_tokens=4096, max_batch_seqs=max(16, batch_size),
                                             max_prefill_per_step=max_prefill_per_step),
            max_seq_len=512, quiet=True,
        )
        # Set BEFORE any decode so the captured graph uses the chosen kernel.
        eng._model_runner._disable_medium_gemv = disable_medium
        return eng

    if overall:
        import subprocess

        def memory_snapshot(engine):
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"], text=True)
            used, total = [float(value.strip()) for value in output.strip().splitlines()[0].split(",")]
            snapshot = dict(device_used_bytes=int(used * 1024**2),
                            device_total_bytes=int(total * 1024**2))
            if engine is not None:
                snapshot.update(weights_bytes=engine._weights.total_bytes,
                                kv_allocated_bytes=engine._kv_cache.stats().total_bytes,
                                scratch_planned_bytes=engine._vram_plan.scratch_bytes)
            return snapshot

        result_kind = "rowmajor_overall" if model in {"qwen2.5-7b", "llama3.1-8b"} else "overall"
        result_path = Path(f"/root/zse_cache/{model}_{result_kind}_results.json")

        def persist(results):
            results.update(model=model, revision=model_revision, model_path=zse_path,
                           gpu="B200", quality_status=("limited_prior_rowmajor_smoke" if model == "qwen2.5-7b"
                                                       else "not_independently_validated"),
                           notes=["Performance only; token parity is not semantic correctness.",
                                  "Startup times are constructor-only, excluding imports, container startup and download.",
                                  "First engine uses existing model volume; page-cache state is uncontrolled. Later engines reuse warmed caches.",
                                  "First batch includes demand graph capture; measured trials follow warmup.",
                                  "VRAM snapshots are device-wide, not peak or isolated process allocation.",
                                  "Latency summaries are medians of trial statistics; p95 uses nearest rank."])
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=2) + "\n")
            temporary.replace(result_path)
            zse_cache.commit()

        results = benchmark_overall(build_engine, base_prompts, memory_snapshot, persist)
        print(f"OVERALL RESULT: {json.dumps(results)}", flush=True)
        return results
    if sampler_ab or smoke_only:
        smoke_results = []
        if model in {"llama3.1-8b", "qwen2.5-7b"}:
            engine = build_engine(False)
            try:
                for temperature in (0.0, 0.8):
                    engine._sampler._rng.seed(20260927)
                    tokens = [[]]
                    metrics = measure_batch(engine, base_prompts[:1], 48,
                                            temperature=temperature, output_tokens=tokens)
                    text = engine._tokenizer.decode(tokens[0])
                    assert text.strip(), f"Empty decoded {model} output"
                    smoke = dict(temperature=temperature, text=text, metrics=metrics)
                    smoke_results.append(smoke)
                    print(f"MODEL SMOKE ({model}): {smoke}", flush=True)
            finally:
                engine.destroy()
        results = {} if smoke_only else compare_samplers(build_engine, base_prompts)
        results.update(model=model, revision=model_revision, smoke=smoke_results)
        result_kind = "rowmajor_smoke" if smoke_only else "sampler_ab"
        result_path = Path(f"/root/zse_cache/{model}_{result_kind}_results.json")
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        zse_cache.commit()
        print(f"Persisted sampler results: {result_path}", flush=True)
        return results
    if sampling_profile or profile_only:
        return profile_sampling(build_engine, base_prompts, profile_only=profile_only)

    sampling_results = {}
    if validate_sampling:
        reference_tokens = {}
        for graph_enabled in (False, True):
            engine = build_engine(False, max_prefill_per_step=12)
            try:
                runner = engine._model_runner
                if not graph_enabled:
                    runner.destroy_graph()
                    assert runner._graph_runner is None
                else:
                    assert runner._graph_runner is not None, "CUDA graphs unavailable"
                    graph_decode = runner.batched_decode_graph
                    graph_calls = []

                    def checked_graph(token_ids, seq_ids, positions, **kwargs):
                        assert kwargs.get("return_logits") is True
                        previous = set(runner._batched_graph_runners)
                        rows = graph_decode(token_ids, seq_ids, positions, **kwargs)
                        assert set(runner._batched_graph_runners) - previous <= {len(token_ids)}
                        assert len(rows) == len(token_ids)
                        graph_calls.append(len(token_ids))
                        return rows

                    runner.batched_decode_graph = checked_graph

                for concurrency in (1, 4, 12):
                    engine._sampler._rng.seed(20260927)
                    tokens = [[] for _ in range(concurrency)]
                    prompts = [f"Write a short essay number {index} about renewable energy."
                               for index in range(concurrency)]
                    metrics = measure_batch(engine, prompts, 24, temperature=0.8,
                                            repetition_penalty=1.1, output_tokens=tokens)
                    key = f"graph_{graph_enabled}_n{concurrency}"
                    sampling_results[key] = dict(metrics, tokens=tokens)
                    if graph_enabled:
                        assert tokens == reference_tokens[concurrency], f"Sampled token mismatch at N={concurrency}"
                        assert concurrency in graph_calls, f"Graph path not used at N={concurrency}"
                    else:
                        reference_tokens[concurrency] = tokens
                    print(f"SAMPLED {key}: {metrics}", flush=True)
                if graph_enabled:
                    sampling_results["graph_batch_sizes"] = sorted(set(graph_calls))
                    sampling_results["token_parity"] = True
                    print("PASS: sampled token parity and demand-only graph capture at N=1,4,12", flush=True)
            finally:
                engine.destroy()

    def measure(disable, label, runs=3):
        eng = build_engine(disable)
        try:
            measure_batch(eng, base_prompts, 16)
            samples = [measure_batch(eng, base_prompts, MAX_TOKENS) for _ in range(runs)]
            summary = {}
            for metric in ("aggregate_tps", "ttft_median_ms", "itl_median_ms"):
                values = [sample[metric] for sample in samples if sample[metric] is not None]
                summary[metric] = statistics.median(values) if values else None
            summary["runs"] = samples
            print(f"    {label}: {summary}", flush=True)
            return summary
        finally:
            eng.destroy()

    print(f"\n[A/B] N={batch_size} median aggregate (separate engine and graphs per config):", flush=True)
    fix_on = measure(False, "fix ON (bgemv M16)")
    fix_off = measure(True, "fix OFF (tiled)")
    tps_on = fix_on["aggregate_tps"]
    tps_off = fix_off["aggregate_tps"]

    delta = tps_on - tps_off
    pct = (delta / tps_off * 100) if tps_off > 0 else 0.0
    results = {
        "batch_size": batch_size,
        "sampling_validation": sampling_results,
        "fix_on": fix_on,
        "fix_off": fix_off,
        "delta_tps": round(delta, 1),
        "delta_pct": round(pct, 1),
    }
    print("\n" + "=" * 70, flush=True)
    print(f"    fix ON : {tps_on:.1f} tok/s", flush=True)
    print(f"    fix OFF: {tps_off:.1f} tok/s", flush=True)
    print(f"    DELTA  : {delta:+.1f} tok/s ({pct:+.1f}%)", flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 70, flush=True)
    return results


@app.local_entrypoint()
def main(batch_size: int = N, validate_sampling: bool = False,
         report: str = "tests/modal_b200_serving_results.json", sampling_profile: bool = False,
         profile_only: bool = False, sampler_ab: bool = False, spawn: bool = False,
         model: str = "qwen2.5-14b", overall: bool = False, smoke_only: bool = False):
    import json
    import traceback
    from pathlib import Path

    try:
        if model not in {"qwen2.5-14b", "qwen2.5-7b", "llama3.1-8b"}:
            raise ValueError(f"Unsupported benchmark model: {model}")
        if sum((validate_sampling, sampling_profile, profile_only, sampler_ab, overall, smoke_only)) > 1:
            raise ValueError("Choose only one benchmark mode")
        if spawn:
            call = ab2.spawn(batch_size, validate_sampling, sampling_profile, profile_only, sampler_ab, model, overall, smoke_only)
            print(f"Spawned call {call.object_id}; results will be printed in app logs.", flush=True)
            return
        results = ab2.remote(batch_size, validate_sampling, sampling_profile, profile_only, sampler_ab, model, overall, smoke_only)
        Path(report).write_text(json.dumps(results, indent=2) + "\n")
        print(f"Results saved to {report}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
