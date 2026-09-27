# B200 Serving Validation - 2026-09-27

## Corrected Qwen7B Overall Benchmark (Latest)

[Modal app](https://modal.com/apps/zyoralabsai/main/ap-XbfLyxYBoDzqceW0W1huc7) completed normally. The locally verified [raw results](modal_b200_qwen7b_rowmajor_overall.json) use the corrected row-major Qwen2.5-7B INT4 artifact, revision `a09a35458c702b33eeacc393d103063234e8bc28`. These results are separate from the earlier corrupt-artifact measurements below.

Fresh engines per configuration, N=1/4/12, greedy/sampled, graphs off/on, three measured trials each, 100 tokens per request. Sampling configuration: top_k=50, top_p=0.9, repetition_penalty=1.0, seed=20260927. All 12 configurations and 36 trials completed: 204 requests, 20,400 measured tokens, zero failures, full-length output and graph-off token-hash parity throughout. Graph-on captures were exactly [1], [4], and [12]; graph-off cases captured none.

| Concurrency | Mode | Graph OFF tok/s | Graph ON tok/s | ON TTFT ms | ON inter-token ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Greedy | 61.76 | 110.98 | 115.54 | 7.96 |
| 1 | Sampled | 13.33 | 14.75 | 161.38 | 67.02 |
| 4 | Greedy | 127.58 | 134.03 | 287.46 | 25.54 |
| 4 | Sampled | 14.96 | 14.92 | 403.31 | 262.87 |
| 12 | Greedy | 167.97 | 173.55 | 749.10 | 55.85 |
| 12 | Sampled | 15.40 | 15.41 | 1048.58 | 766.36 |

Throughput is median aggregate tokens/sec across three trials. Latencies are medians of per-trial statistics. Graphs materially help single-request greedy throughput, give modest concurrent greedy gains, and show no meaningful concurrent sampled gain in this run. The sampled performance difference from earlier allocations is not a controlled regression measurement; its cause has not been profiled on this corrected artifact.

First constructor: 5.1788s; later constructor median: 2.6884s. These exclude imports, container startup and downloads; initial page-cache state is uncontrolled and later engines reuse warm caches. First batches include demand capture and are excluded from measured trials. Device-wide used-memory snapshots after loading/generation span 6.6309-6.6484 GiB, not peak or isolated process VRAM. Weight bytes: 5,647,215,616; KV allocation: 704,643,072; planned scratch: 96,010,240.

Recorded output samples are coherent and on-topic, with repetition and token-limit truncation visible. Quality status remains `limited_prior_rowmajor_smoke`: this workload and exact token parity do not establish comprehensive semantic correctness. No production inference changes, dependencies, threshold changes or ROCm work were made for this benchmark closeout.

## Qwen7B Correctness Fix

Root cause: the fast converter wrote INT4 matrices in 64x16 tiled order, while production GEMV/matmul readers expected row-major order. The GPU loader preserves file bytes without relayout. A deterministic conversion regression reproduced corruption at element 16 (expected 2, recovered 1).

Conversion now preserves row-major bytes and metadata. GPU loading rejects explicitly tiled artifacts with a reconversion message. Legacy conversion checkpoints restart instead of mixing layouts. Existing affected artifacts must be reconverted from source; changing metadata alone does not repair them.

Validation: 246 scoped conversion, format, orchestrator, ZStreamer and speculative tests pass. [B200 smoke app](https://modal.com/apps/zyoralabsai/main/ap-V0vN6yTOB7AsledenPJmU7) freshly converted official Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` to a separate `_int4_rowmajor.zse` cache artifact (339 tensors, 5,654,453,970 bytes). App completed normally.

Both single-request checks completed 48 tokens with zero failures, using the same renewable-energy chat prompt:

- Greedy: "Renewable energy, often referred to as green energy, is a critical component in the global effort to transition away from fossil fuels and mitigate the impacts of climate change."
- Sampled (temperature 0.8): "Renewable energy, a term that encompasses a variety of energy sources that are naturally replenished and are replenished on a human timescale, has become increasingly important in the modern"

Raw results: [modal_b200_qwen7b_rowmajor_smoke.json](modal_b200_qwen7b_rowmajor_smoke.json). Outputs are coherent and on-topic, with sampled repetition and token-limit truncation visible. This is a limited smoke check, not full model-quality evaluation or a new throughput benchmark. Historical failed-quality measurements below remain unchanged and must not be presented as validated useful-answer throughput.

## Earlier Benchmark Results

Source: [Modal app logs](https://modal.com/apps/zyoralabsai/main/ap-RmfjkbZLuIIq56r2RGEOP6), recovered with `modal app logs`. Logs report `RESULT` and `Stopping app - local entrypoint completed` at 00:49:16 +05:30. The expected local JSON file was not found; this report records the logged results, not a recovered JSON artifact.

Hardware/model: NVIDIA B200, cached Qwen2.5-14B INT4, CUDA 12.8.
Harness: `tests/test_modal_b200_ab2.py --batch-size 12 --validate-sampling`.

## Greedy Medium-M A/B

Fresh engine and graphs per configuration; dispatch flag set before capture. Default gradual admission, 16-token warmup, three trials of 12 requests with 100 generated tokens each. All six trials generated 1,200 tokens and passed completion, error and nonempty-output gates.

| Metric (median of three trials) | Fix OFF | Fix ON |
| --- | ---: | ---: |
| Aggregate throughput (tok/s) | 48.08088842876458 | 97.70541777864695 |
| Per-trial median TTFT (ms) | 2082.8904705000186 | 1234.4214184999914 |
| Per-trial median inter-token latency (ms) | 236.1917240000082 | 105.44027649999066 |

Throughput improvement: **2.03x (+103.2%)**. This compares medium-M dispatch, not graph-on versus graph-off.

| Configuration | Trial | Throughput (tok/s) | Elapsed (s) |
| --- | ---: | ---: | ---: |
| OFF | 1 | 47.885837827705295 | 25.059601219 |
| OFF | 2 | 48.08088842876458 | 24.95794148599998 |
| OFF | 3 | 48.08148240156543 | 24.95763316900002 |
| ON | 1 | 96.56578618756937 | 12.426761562000024 |
| ON | 2 | 97.70541777864695 | 12.281816374999977 |
| ON | 3 | 97.71036011765375 | 12.281195142000001 |

## Sampled Graph Correctness

Exact graph-on/off token equality passed at N=1,4,12, with actual graph execution at M=1,4,12, logits-return routing and demand-only capture. Each configuration produced 408 tokens total (24 per request), with temperature 0.8, repetition penalty 1.1 and RNG seed 20260927. Parity engines admit up to 12 prefills per step so the short workload reaches M=12.

| Concurrency | Tokens per configuration | Graph OFF tok/s | Graph ON tok/s |
| --- | ---: | ---: | ---: |
| 1 | 24 | 3.968767288405063 | 4.244993052052727 |
| 4 | 96 | 4.293084410010148 | 4.4566235883293475 |
| 12 | 288 | 4.07275924245569 | 4.405991430628007 |

These are single short correctness trials, not repeated sampling performance benchmarks. Token parity establishes equivalence to graph-off execution, not independent model-quality validation.

ROCm sampled parity/performance, wave64 medium-M GPU parity and a measured dispatch sweep around M=16 remain unvalidated. No dispatch threshold changed.

## Sampling Attribution

Matched N=12, 24 tokens/request, full admission of 12 prefills, top_k=50,
top_p=0.9, repetition_penalty=1.0. Greedy temperature=0; sampled=0.8.
App `ap-aAHxuQkznP24PtVFz4JRXv` completed three timed trials per mode,
with exact instrumentation-token parity and 288 tokens per trial, before
interruption ahead of cProfile. Median greedy throughput was 75.204875 tok/s;
sampled was 4.364602 tok/s. Sampling consumed about 94% of sampled wall time.
These are instrumented wall-clock measurements, not CUDA-event timings.

Separate cProfile recovery completed in app `ap-zOIHdFeiInIXwUT7BbXWsR`:
96 sampler calls consumed 16.192 seconds, top-p 6.635 seconds, top-k 4.686
seconds, and softmax 5.265 seconds (overlapping cumulative times).
cProfile was excluded from throughput medians.

## Compact Sampler A/B

Source: completed `SAMPLER RESULT` from [spawned recovery app logs](https://modal.com/apps/zyoralabsai/main/ap-mCB0VYCyzqjswUsClZYhdn)
at 01:19:05 +05:30. Spawn mode prints results remotely without writing the
local JSON report; this section records log-derived results.

Hardware/model: B200, Qwen2.5-14B INT4. Separate engines and graphs for
reference and compact samplers; medium-M optimization enabled for both.
Reference disables only the compact helper, retaining the old sampling path.
Each engine receives a two-token warmup. Each of three measured trials uses
N=12, 24 tokens/request, temperature=0.8, top_k=50, top_p=0.9,
repetition_penalty=1.0 and seed=20260927, with full prefill admission.
No instrumentation wrappers or cProfile in these trials.

| Metric | Reference | Compact |
| --- | ---: | ---: |
| Trial 1 throughput (tok/s) | 9.815504 | 33.877601 |
| Trial 2 throughput (tok/s) | 9.159273 | 33.842254 |
| Trial 3 throughput (tok/s) | 8.992072 | 33.834126 |
| Median throughput (tok/s) | 9.159273 | 33.842254 |
| Median of trial-median TTFT (ms) | 1260.045 | 794.760 |
| Median of trial-median inter-token latency (ms) | 1258.921 | 305.356 |

Median throughput improved **3.69x**. All six trials produced 288 tokens,
passed completion/error/nonempty gates and matched the reference token IDs
exactly (864 measured tokens per sampler). This establishes equivalence to
the reference sampler, not independent model-quality validation.

The optimization uses stdlib heap selection for the top-k cutoff, retaining
ties and original token order, then performs top-p/softmax on compact
candidates. Float32 temperature/penalty updates and RNG semantics are
preserved; nonfinite logits fall back to the original path. Local
orchestrator/ZStreamer/speculative suites: **221 passed**. Harness checks
after adding spawn mode: **9 passed**. No dependencies or threshold changes.

Earlier app `ap-rpHWLqGa7pwWHUyMUbsva8` was canceled after three reference
trials (median 4.684327 tok/s) and one compact trial (14.480404 tok/s,
exact token parity). Those partial measurements are not pooled with this
completed comparison. Absolute throughput varied substantially between
allocations; the cause was not established. Reported speedup is specific
to the completed sequential, within-run A/B, not a hardware-wide guarantee.
These short sampled trials are distinct from the 100-token greedy benchmark.

## Qwen 7B Fresh Conversion: Quality Failure

App: [Qwen 7B run](https://modal.com/apps/zyoralabsai/main/ap-NKbtxyGgJM7qWcy783CneB).
Command: `modal run --detach tests/test_modal_b200_ab2.py --model qwen2.5-7b --batch-size 12 --sampler-ab --report tests/modal_b200_qwen7b_sampler_ab.json`.

Official public source: `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Anonymous repository access
succeeded. Snapshot retrieval took 0.38s with the existing HF cache; this
is not an uncached network-download benchmark. The C-accelerated converter
wrote a current-format INT4 model with 28 layers, hidden size 3584 and
339 tensors. Header parsing and file-size checks passed (5,654,453,970 bytes).
The converted artifact was committed to `zse-model-cache` at
`/root/zse_cache/qwen2.5-7b_a09a35458c702b33eeacc393d103063234e8bc28_int4.zse`.

Both smoke tests used a Qwen-formatted chat prompt asking for an essay about
renewable energy, seed 20260927 and a 48-token limit. Greedy (temperature 0)
and sampled (0.8) each produced 48 tokens, but visual inspection found
incoherent multilingual/code fragments, not meaningful answers. The greedy
output began `entifulilmingtonilmington`, with later fragments such as
`BuilderInterfacevelte`; sampled output also mixed unrelated text and code.
The harness's nonempty-output assertion passed but does not establish quality.

The remaining timing trials were canceled after inspecting these samples.
Modal confirmed the app stopped at 07:26:38 +05:30 with zero tasks. No complete
sampler A/B or independent model-correctness result is claimed. The cause
has not been isolated between conversion, tokenization and inference.
The earlier 14B parity/performance results above remain distinct evidence,
not validation of this fresh 7B artifact's output quality.

Harness changes only: added Qwen model selection, anonymous pinned download,
separate cache and Qwen chat formatting, reusing existing smoke/A/B logic.
Eleven focused benchmark tests pass; no editor diagnostics in touched Python
files. No production inference changes, dependencies or ROCm work.

## Qwen 7B Overall Performance Benchmark

Performance measurement was subsequently authorized despite the known quality
failure above. These timings do not establish useful-answer throughput or
production readiness of this artifact.

### Method

- Same pinned Qwen2.5-7B-Instruct INT4 artifact on one Modal B200.
- Concurrency 1, 4 and 12; greedy temperature 0 and sampled temperature 0.8;
	CUDA graphs off and on. Medium-M optimization enabled in all cases.
- Fresh engine per configuration; initial 2-token batch, 16-token warmup,
	then three measured trials of 100 generated tokens per request.
- Sampled settings: top-k 50, top-p 0.9, repetition penalty 1.0;
	RNG seed 20260927 reset before every measured trial.
- Qwen chat-formatted renewable-energy prompts; max sequence length 512,
	max batch sequences 16, batch token budget 4096, prefill admission limit
	equal to the tested concurrency.
- Throughput includes prefill and request lifecycle, not just decode.
	Latencies are medians of three trial statistics; p95 uses nearest rank
	within each trial, not a pooled percentile across trials.
- Startup is engine-constructor time only, excluding imports, container
	scheduling, download and conversion. Page-cache state is uncontrolled;
	later engines reuse warmed caches. First-batch timing includes demand
	graph capture; measured trials follow warmup.
- VRAM values are device-wide snapshots, not peak or isolated-process usage.
	HTTP/network latency, long-context workloads and competitor performance
	were not measured.

### Run Provenance

The initial [performance run](https://modal.com/apps/zyoralabsai/main/ap-abcDFTwYQTKAl26pkwXjRV)
was canceled at 07:50:22 +05:30 during N=12 sampled graph-off trial 2.
Its [saved checkpoint](modal_b200_qwen7b_overall.json) contains ten completed
configurations and one completed trial in the unfinished N=12 sampled graph-off
case (44.6466 tok/s); this is not a three-trial median. The interrupted trial
is excluded. Cancellation cause was not established.

A separate [spawned recovery](https://modal.com/apps/zyoralabsai/main/ap-gELHxZI432NyMQlbSwUegU)
repeats the full matrix using the existing harness, without production code
changes. Measurements from different allocations must not be pooled.
Recovery completed all 12 configurations and is the sole source for the tables
below. [Complete raw results](modal_b200_qwen7b_overall_recovery.json).
Call ID: `fc-01M3GAQ632Y8DDDE9ZJ55PCR5B`.

### Throughput

Aggregate generated tokens per second, median of three measured trials:

| Concurrent requests | Mode | Graphs off | Graphs on | Change |
|---|---|---:|---:|---:|
| 1 | Greedy | 80.39 | 113.43 | +41.10% |
| 1 | Sampled | 28.46 | 31.91 | +12.12% |
| 4 | Greedy | 129.92 | 136.15 | +4.79% |
| 4 | Sampled | 33.17 | 33.49 | +0.96% |
| 12 | Greedy | 171.35 | 176.67 | +3.10% |
| 12 | Sampled | 35.37 | 35.41 | +0.10% |

### Latency

Milliseconds; each entry is median / p95 (median of trial statistics).
TTFT is time to first token; ITL is the interval between successive tokens
for the same request.

| N | Mode | Graphs off TTFT | Graphs on TTFT | Graphs off ITL | Graphs on ITL |
|---|---|---:|---:|---:|---:|
| 1 | Greedy | 107.16 / 107.16 | 107.18 / 107.18 | 11.47 / 11.67 | 7.85 / 8.53 |
| 1 | Sampled | 124.34 / 124.34 | 124.14 / 124.14 | 34.15 / 35.22 | 30.44 / 31.61 |
| 4 | Greedy | 268.03 / 428.57 | 267.86 / 428.42 | 26.76 / 27.51 | 25.37 / 26.06 |
| 4 | Sampled | 310.31 / 495.87 | 310.51 / 496.47 | 116.68 / 118.66 | 115.69 / 118.36 |
| 12 | Greedy | 697.33 / 1290.54 | 696.78 / 1289.41 | 57.65 / 58.41 | 55.60 / 56.30 |
| 12 | Sampled | 808.63 / 1498.18 | 809.61 / 1494.98 | 327.05 / 333.30 | 327.16 / 334.62 |

At N=1, each trial has only one TTFT observation, so its median and p95
coincide. These short trials do not establish production tail-latency SLOs.

### Startup And Memory

- First engine constructor: **4.4493 s**. Subsequent constructors:
	**2.4254 s median**, range **2.2117-2.6336 s**. Not deployment cold-start.
- First engine's initial two-token batch: **309.79 ms TTFT**, **321.67 ms**
	total batch time, before warmup (greedy, N=1, graphs off).
- Post-generation device VRAM snapshots: **6.6309-6.6484 GiB**.
- Weight allocation: **5,647,215,616 bytes**; KV allocation:
	**704,643,072 bytes**; planned scratch: **96,010,240 bytes**.
- Device-wide memory includes runtime/context allocations. The slight growth
	between sequential engines is observed, not diagnosed; this is not a
	memory-leak or peak-memory benchmark.

### Reliability And Interpretation

All **36 measured trials / 204 requests / 20,400 generated tokens** completed
with zero reported request errors and every request reaching 100 tokens.
All measured token hashes matched the corresponding graph-off reference;
graph-on cases captured exactly the requested batch size (1, 4 or 12).
Warmup and initial batches are excluded from these totals.

Decoded samples remain incoherent in both modes. Graph consistency is not
semantic correctness, and zero errors in this short test is not long-running
reliability certification. This artifact is not quality-validated for serving.

Greedy throughput increases with concurrency, while sampled aggregate
throughput changes little from N=1 to N=12 and per-request latency rises.
Graphs help single-request throughput most in this allocation; their sampled
benefit at N=12 is negligible. No new profiling was performed here to assign
the remaining time to individual stages.

Absolute timings varied across allocations: the interrupted run measured
N=1 greedy off/on 101.50/113.76 tok/s and N=4 sampled off/on 41.27/41.43,
versus 80.39/113.43 and 33.17/33.49 in the complete recovery. The cause is
unknown; do not combine these runs or generalize the graph speedups to all
allocations. No competitor or different-GPU comparison is claimed.

Local verification: **14 focused harness tests** and **226 scoped serving
tests** passed. Final artifact checks verified matrix coverage, trial counts,
completion status, full-length output, parity flags and captured batch sizes.
No production inference changes, dependencies, dispatch threshold changes,
ROCm work or commits were made for this benchmark.