"""Calculate ZPF break-even point: ingestion overhead vs per-query token savings."""

import os, sys, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.zpf_cost_bench import DOC_CNN, PlainChunker, PlainVectorSearch

# Measure ZPF ingestion time
from zse.core.zrag.pipeline import RAGPipeline
from zse.core.zrag.embedder import Embedder

N_RUNS = 5

zpf_times = []
plain_times = []

for _ in range(N_RUNS):
    with tempfile.TemporaryDirectory() as tmpdir:
        cnn_path = os.path.join(tmpdir, "cnn.md")
        with open(cnn_path, "w") as f:
            f.write(DOC_CNN)

        # ZPF ingestion
        t0 = time.perf_counter()
        p = RAGPipeline(store_dir=os.path.join(tmpdir, "zpf_store"))
        p.ingest(cnn_path)
        zpf_times.append(time.perf_counter() - t0)

        # Plain ingestion
        t0 = time.perf_counter()
        embedder = Embedder()
        plain = PlainVectorSearch(embedder)
        plain.ingest(DOC_CNN)
        plain_times.append(time.perf_counter() - t0)

zpf_avg = sum(zpf_times) / len(zpf_times)
plain_avg = sum(plain_times) / len(plain_times)
overhead = zpf_avg - plain_avg

print(f"ZPF ingestion:   {zpf_avg * 1000:.0f}ms avg ({N_RUNS} runs)")
print(f"Plain ingestion: {plain_avg * 1000:.0f}ms avg ({N_RUNS} runs)")
print(f"ZPF overhead:    {overhead * 1000:.0f}ms per document")
print()

# Per-query savings (from cost benchmark results)
zpf_tokens_per_query = 856
plain_tokens_per_query = 1257
tokens_saved_per_query = plain_tokens_per_query - zpf_tokens_per_query

print(f"Tokens saved per query: {tokens_saved_per_query}")
print()

# GPT-4o: $2.50/M input tokens
cost_per_token = 2.50 / 1_000_000
savings_per_query = tokens_saved_per_query * cost_per_token

# Ingestion overhead cost (CPU time at ~$0.05/hour for a small instance)
cpu_cost_per_hour = 0.05
ingestion_cost = overhead * cpu_cost_per_hour / 3600

print(f"Savings per query: ${savings_per_query:.6f}")
print(f"Ingestion CPU cost: ${ingestion_cost:.8f}")
print()

if savings_per_query > 0:
    breakeven_queries = max(1, ingestion_cost / savings_per_query)
    print(f"Break-even: {breakeven_queries:.1f} queries per document")
    print(f"  (After {breakeven_queries:.0f} queries on this document, ZPF has paid for itself)")
else:
    print("No break-even: ZPF doesn't save tokens")
