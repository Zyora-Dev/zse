# Qwen2.5-32B-Instruct — ZSE vs vLLM on AMD MI300X (192GB, gfx942, ROCm 7.2)

Single-GPU, max_seq_len=1024, prompt = "essay about history of AI", max_new_tokens=100, concurrent N=4 (4 distinct prompts).

| Metric                | ZSE (INT4)       | vLLM (FP16)       | Winner            |
|-----------------------|------------------|-------------------|-------------------|
| Cold start            | **4.34 s**       | 42.65 s           | **ZSE 9.83×**     |
| VRAM used             | **22.07 GB**     | 161.77 GB         | **ZSE 7.33×**     |
| Single tok/s          | 32.9             | **56.4**          | vLLM 1.71×        |
| Single TTFT           | 356.7 ms         | n/a (vLLM agg)    | —                 |
| Concurrent N=4 tok/s  | 50.9             | **197.2**         | vLLM 3.87×        |

ZSE breakdown: weights 16.89 GB + KV 3.0 GB + scratch 0.23 GB. Init: load 0.0s, plan 0.0s, weights 3.9s, compile 3.0s.
vLLM breakdown: weights ~61.1 GB FP16 + KV reserved ~98 GB (gpu_memory_utilization=0.85).

Identical pattern to A100-80GB Modal bench: ZSE dominates cold-start + memory footprint, vLLM wins steady-state throughput. The throughput delta corresponds to Gap #6 (hand-written INT4 dequant matmul not yet routed through Tier-3 `local_array` + `reinterpret` primitives shipped today).
