"""
ZPF Retrieval Quality Evaluation — 15-question ground-truth benchmark.

Tests whether .zpf semantic compression preserves what matters for LLM retrieval.

5 documents (3 existing + 2 new), 15 questions across 5 difficulty groups:
  Group 1 — Factual / Definition retrieval (DEFINITION, FACT blocks)
  Group 2 — Procedural retrieval (PROCEDURE, CODE blocks)
  Group 3 — Specific detail (tests noise removal didn't kill real content)
  Group 4 — Cross-section (answer spans multiple paragraphs/sections)
  Group 5 — Noise-adjacent (content near boilerplate boundaries)

Scoring per question:
  HIT  — all answer fragments found in retrieved context
  PART — some but not all fragments found
  MISS — no answer fragments found

Comparison: ZPF vs plain 512-char fixed chunking, same embedding model.
Two token budgets: 200 (tight) and 500 (normal).
"""

import os
import sys
import tempfile
import numpy as np
from typing import List, Tuple

# Reuse existing benchmark documents
from zpf_compression_bench import DOC_CLEAN_ARTICLE, DOC_NOISY_WEBPAGE, DOC_LONG_STRUCTURED


# ============================================================================
# DOCUMENT 4: Q&A / Reference style (FAQ + troubleshooting)
# ============================================================================
DOC_QA_REFERENCE = """# KV Cache — Frequently Asked Questions

## What is a KV cache?

Q: What is a KV cache in the context of LLM inference?
A: The KV (Key-Value) cache stores the computed key and value tensors from
previous tokens during autoregressive generation. Without it, every new token
would require recomputing attention over the entire sequence from scratch. The
KV cache trades memory for speed — it uses O(n * d * layers) memory where n is
the sequence length, d is the hidden dimension, and layers is the number of
transformer layers.

## Why does KV cache matter?

Q: Why is the KV cache important for performance?
A: For a typical 7B parameter model with 32 layers and 4096 hidden dimension,
the KV cache for a single 2048-token sequence consumes approximately 1 GB of
VRAM in FP16. At batch size 32, that's 32 GB just for the cache — often more
than the model weights themselves. This is why KV cache management is the primary
bottleneck for LLM serving throughput, not the model computation itself.

## Quantization

Q: Can you quantize the KV cache?
A: Yes. KV cache quantization reduces memory usage significantly:
- FP16 (default): 2 bytes per element, highest quality
- INT8: 1 byte per element, ~50% memory savings, minimal quality loss
- INT4: 0.5 bytes per element, ~75% savings, noticeable quality loss on long contexts
- FP8 (E4M3): 1 byte per element, better quality than INT8 at same size (H100 only)

The recommended setting for most production workloads is INT8, which ZSE uses by
default (configurable via ZSE_KV_QUANT environment variable).

## Paged Attention

Q: What is paged attention?
A: Paged attention (introduced in vLLM) manages KV cache memory like an operating
system manages virtual memory — using fixed-size pages instead of contiguous
allocation. This eliminates memory fragmentation and enables:
- Non-contiguous memory allocation for sequences
- Memory sharing between sequences with common prefixes (prefix caching)
- Dynamic memory allocation that grows/shrinks as sequences progress

ZSE implements paged attention with 16-token page granularity and automatic
page reclamation when sequences complete.

## Prefix Caching

Q: What is prefix caching and when should I use it?
A: Prefix caching reuses KV cache pages across requests that share the same prompt
prefix (e.g., system prompts). For a chatbot with a 500-token system prompt serving
100 concurrent users, prefix caching saves approximately 50 GB of VRAM (100 *
500 tokens * cache_per_token). Enable it when you have repeated system prompts:

    zse serve model --prefix-cache --system-prompt "You are a helpful assistant..."

Prefix caching adds ~2ms lookup latency but saves significant VRAM and speeds up
time-to-first-token for cached prefixes.

## Speculative Decoding

Q: How does speculative decoding interact with the KV cache?
A: Speculative decoding uses a small draft model to predict N tokens ahead, then
verifies them with the main model in a single forward pass. The draft model
maintains its own separate KV cache (much smaller, since draft models are typically
1-3B parameters). When speculative tokens are rejected, their KV cache entries are
discarded and the main model's cache is rolled back to the last accepted position.

The overhead is approximately 15-20% more VRAM for the draft model's separate cache,
but throughput improves by 2-3x for latency-sensitive workloads.

## Limits and Failure Modes

Q: What happens when the KV cache runs out of memory?
A: When KV cache memory is exhausted, ZSE handles it based on the configured policy:
- **evict_oldest**: Evicts the oldest inactive sequence's cache (default)
- **reject_new**: Returns HTTP 503 to new requests until memory is available
- **swap_to_cpu**: Offloads least-recently-used cache pages to CPU memory (slower but no data loss)

The eviction policy is set via --cache-policy flag. For production, evict_oldest
is recommended because it maintains throughput at the cost of occasional recomputation.

Q: What is the maximum context length the KV cache supports?
A: ZSE supports up to 128K tokens per sequence, limited by available VRAM. For a
7B model in INT8 KV mode with 24 GB VRAM, the practical limit is approximately
32K tokens per sequence at batch size 4, or 8K tokens at batch size 16.
"""


# ============================================================================
# DOCUMENT 5: Messy real-world doc (mixed formatting, inline noise, code examples)
# ============================================================================
DOC_MESSY_REALWORLD = """
Skip to content | Accessibility Help | Site Map

Last updated: March 15, 2026 | Author: ML Team | Reading time: 12 min
Share: Twitter | LinkedIn | Hacker News
Views: 15,234 | Comments (47)

# Fine-Tuning Large Language Models: A Practical Guide

> TL;DR: Fine-tuning adapts a pre-trained model to your task. Use LoRA for
> efficiency, full fine-tuning for maximum quality. Budget 2-4x the model VRAM
> for training.

## Why Fine-Tune?

Pre-trained models are general-purpose. Fine-tuning specializes them for specific
domains or tasks. For example, a general 7B model might score 45% on medical QA,
but after fine-tuning on 10K medical examples, it can reach 78%. The key tradeoff
is cost vs quality — fine-tuning costs GPU hours but produces a permanently better
model for your use case.

Common fine-tuning scenarios:
- Domain adaptation (legal, medical, finance)
- Instruction following (chat format compliance)
- Code generation (language-specific improvements)
- Style transfer (tone, format, length control)

## Methods

### Full Fine-Tuning

Updates ALL model parameters. Produces the highest quality results but requires the
most resources. For a 7B model, full fine-tuning needs approximately 56 GB VRAM
(8x the model size in FP16 for optimizer states and gradients).

Memory breakdown for a 7B FP16 model:
- Model weights: 14 GB
- Gradients: 14 GB
- Optimizer states (AdamW): 28 GB (2x weights for m and v)
- Activations: 2-8 GB depending on batch size and sequence length
- Total: ~58-64 GB

### LoRA (Low-Rank Adaptation)

LoRA freezes the base model and trains small rank-decomposition matrices
(typically rank 8-64) on the attention projection layers. This reduces trainable
parameters by 99%+ while achieving 90-95% of full fine-tuning quality.

LoRA configuration example:
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                # rank
    lora_alpha=32,       # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# Output: trainable params: 13.1M || all params: 6.74B || 0.19%
```

LoRA VRAM requirement for 7B model: ~16-18 GB (model + LoRA adapters + activations).

### QLoRA

QLoRA combines 4-bit quantization with LoRA. The base model is loaded in 4-bit
precision (3.5 GB for 7B) and LoRA adapters are trained in FP16. This enables
fine-tuning a 7B model on a single GPU with just 6 GB VRAM. Quality is typically
within 1-2% of full LoRA.

## Training Hyperparameters

Critical settings to get right:

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| Learning rate | 1e-4 to 2e-5 | Lower for larger models |
| Batch size | 4-8 | Effective batch via gradient accumulation |
| Epochs | 1-3 | More risks overfitting on small datasets |
| Warmup ratio | 0.03-0.1 | Prevents early instability |
| Weight decay | 0.01 | Standard for AdamW |
| Max sequence length | 2048-4096 | Match your use case |
| Gradient accumulation | 4-8 steps | Simulates larger batch sizes |

Warning: Learning rates above 1e-4 frequently cause catastrophic forgetting,
where the model loses its general capabilities after fine-tuning. Always evaluate
on both your target task AND general benchmarks (MMLU, HellaSwag) to detect this.

## Dataset Preparation

Quality matters more than quantity. Key principles:

1. Remove duplicates — even 5% duplication causes measurable overfitting
2. Balance categories — imbalanced datasets bias the model toward majority class
3. Validate formatting — one malformed example can corrupt a training batch
4. Include negative examples — teach the model what NOT to do
5. Set aside 10-15% for validation — never evaluate on training data

For instruction-tuning, use this format:
```json
{
    "instruction": "Summarize the following medical report...",
    "input": "Patient presented with...",
    "output": "Summary: 45-year-old patient with..."
}
```

Minimum dataset sizes (rough guidelines):
- Instruction following: 1K-10K examples
- Domain adaptation: 10K-100K examples
- Style transfer: 500-5K examples

## Common Failures

### Catastrophic Forgetting
The model forgets general knowledge after fine-tuning on a narrow dataset. Symptoms:
degraded performance on unrelated tasks, repetitive outputs, loss of instruction
following ability. Prevention: use a lower learning rate, mix in general-purpose
data (5-10% of training set), and evaluate on general benchmarks.

### Overfitting
Training loss decreases but validation loss increases. The model memorizes training
examples instead of learning patterns. Prevention: early stopping based on
validation loss, reduce epochs, increase dropout, use more diverse training data.

### Mode Collapse
The model generates the same or very similar outputs regardless of input. Usually
caused by too-high learning rate or insufficient data diversity. Fix: reduce learning
rate by 10x, add more diverse examples, increase temperature during evaluation.

Print this page | Download PDF | Save for later

Related reading:
- Understanding Attention Mechanisms
- RLHF Explained: Training with Human Feedback
- Efficient Inference at Scale

Subscribe to our weekly ML newsletter
Enter your email address: [__________] [Subscribe]

Filed under: machine-learning, fine-tuning, LLM
Tags: LoRA, QLoRA, training, PEFT
"""


# ============================================================================
# 15 Ground-Truth QA Pairs
# ============================================================================

QA_PAIRS = [
    # === Group 1: Factual / Definition retrieval ===
    {
        "id": 1,
        "group": "Definition",
        "question": "What is gradient descent?",
        "answer_fragments": [
            "iterative optimization algorithm",
            "minimum",
            "differentiable function",
        ],
        "source": "gradient",
    },
    {
        "id": 2,
        "group": "Definition",
        "question": "What does CNN stand for and what is it used for?",
        "answer_fragments": ["Convolutional Neural Networks", "visual imagery"],
        "source": "cnn_noisy",
    },
    {
        "id": 3,
        "group": "Definition",
        "question": "What is the difference between batch gradient descent and stochastic gradient descent?",
        "answer_fragments": ["entire training dataset", "single randomly selected sample"],
        "source": "gradient",
    },
    # === Group 2: Procedural retrieval ===
    {
        "id": 4,
        "group": "Procedure",
        "question": "How do you install ZSE with CUDA support?",
        "answer_fragments": ["pip install zllm-zse", "cuda"],
        "source": "deployment",
    },
    {
        "id": 5,
        "group": "Procedure",
        "question": "What command do you use to enable tensor parallelism across 4 GPUs?",
        "answer_fragments": ["tensor-parallel 4"],
        "source": "deployment",
    },
    {
        "id": 6,
        "group": "Procedure",
        "question": "What are the steps to configure a LoRA fine-tuning run?",
        "answer_fragments": ["LoraConfig", "r=16", "target_modules"],
        "source": "finetune",
    },
    # === Group 3: Specific detail retrieval ===
    {
        "id": 7,
        "group": "Detail",
        "question": "What are the default beta values for the Adam optimizer?",
        "answer_fragments": ["beta1=0.9", "beta2=0.999", "epsilon=1e-8"],
        "source": "gradient",
    },
    {
        "id": 8,
        "group": "Detail",
        "question": "What happens when the KV cache runs out of memory?",
        "answer_fragments": ["evict_oldest", "reject_new", "swap_to_cpu"],
        "source": "kv_cache",
    },
    {
        "id": 9,
        "group": "Detail",
        "question": "Which environment variable controls the KV cache quantization level in ZSE?",
        "answer_fragments": ["ZSE_KV_QUANT", "int8"],
        "source": "deployment",
    },
    # === Group 4: Cross-section retrieval ===
    {
        "id": 10,
        "group": "Cross-section",
        "question": "What are the hardware requirements for deploying ZSE?",
        "answer_fragments": ["Python 3.10", "NVIDIA GPU", "CUDA 11.8", "16 GB"],
        "source": "deployment",
    },
    {
        "id": 11,
        "group": "Cross-section",
        "question": "Why does catastrophic forgetting happen during fine-tuning and how do you prevent it?",
        "answer_fragments": ["forgets general knowledge", "lower learning rate"],
        "source": "finetune",
    },
    {
        "id": 12,
        "group": "Cross-section",
        "question": "What are the memory limitations of full fine-tuning vs LoRA for a 7B model?",
        "answer_fragments": ["56 GB", "16-18 GB"],
        "source": "finetune",
    },
    # === Group 5: Noise-adjacent retrieval ===
    {
        "id": 13,
        "group": "Noise-adjacent",
        "question": "What does the introduction of the CNN article say about convolution?",
        "answer_fragments": ["convolution", "matrix multiplication"],
        "source": "cnn_noisy",
    },
    {
        "id": 14,
        "group": "Noise-adjacent",
        "question": "What example is given for prefix caching VRAM savings?",
        "answer_fragments": ["500-token system prompt", "100 concurrent users", "50 GB"],
        "source": "kv_cache",
    },
    {
        "id": 15,
        "group": "Noise-adjacent",
        "question": "What learning rate is recommended to avoid catastrophic forgetting?",
        "answer_fragments": ["1e-4"],
        "source": "finetune",
    },
]


# ============================================================================
# Plain fixed-size chunker baseline (no semantic awareness)
# ============================================================================


class PlainChunker:
    """
    Baseline: fixed-size text chunking with overlap.
    No noise removal, no compression, no block typing.
    512-char windows (~128 tokens), 50-char overlap.
    """

    def __init__(self, chunk_size_chars: int = 512, overlap_chars: int = 50):
        self.chunk_size = chunk_size_chars
        self.overlap = overlap_chars

    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                boundary = text.rfind(" ", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.overlap
            if start >= len(text):
                break
        return chunks


class PlainVectorSearch:
    """Baseline vector search with fixed-size chunks, same embedding model."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self._chunker = PlainChunker()

    def ingest(self, text: str):
        new_chunks = self._chunker.chunk(text)
        if not new_chunks:
            return
        new_embeddings = self.embedder.embed(new_chunks)
        self.chunks.extend(new_chunks)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
        if not self.chunks:
            return []
        query_vec = self.embedder.embed_query(query)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self.embeddings / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) or 1)
        scores = normed @ q_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in top_indices]

    def get_context(self, query: str, max_tokens: int = 2000, top_k: int = 10) -> str:
        results = self.search(query, top_k=top_k)
        parts = []
        token_count = 0
        for score, chunk in results:
            chunk_tokens = len(chunk) // 4
            if token_count + chunk_tokens > max_tokens:
                break
            parts.append(chunk)
            token_count += chunk_tokens
        return "\n\n".join(parts)


# ============================================================================
# Scoring
# ============================================================================


def score_answer(context: str, answer_fragments: List[str]) -> str:
    """
    Score retrieval quality:
      HIT  — all fragments found
      PART — some but not all found
      MISS — none found
    """
    ctx_lower = context.lower()
    found = sum(1 for frag in answer_fragments if frag.lower() in ctx_lower)
    total = len(answer_fragments)
    if found == total:
        return "HIT"
    elif found > 0:
        return "PART"
    else:
        return "MISS"


def score_to_points(score: str) -> float:
    """HIT=1.0, PART=0.5, MISS=0.0"""
    return {"HIT": 1.0, "PART": 0.5, "MISS": 0.0}[score]


# ============================================================================
# Main evaluation
# ============================================================================


def run_eval():
    print("=" * 80)
    print("ZPF RETRIEVAL QUALITY EVALUATION — 15-Question Ground-Truth Benchmark")
    print("=" * 80)
    print()
    print("5 documents | 15 questions | 5 groups | 2 token budgets")
    print("ZPF semantic retrieval vs plain 512-char fixed chunking")
    print("Same embedding model: all-MiniLM-L6-v2")
    print()

    from zse.core.zrag.pipeline import RAGPipeline
    from zse.core.zrag.embedder import Embedder

    documents = [
        ("gradient.md", DOC_CLEAN_ARTICLE),
        ("cnn_noisy.md", DOC_NOISY_WEBPAGE),
        ("deployment.md", DOC_LONG_STRUCTURED),
        ("kv_cache.md", DOC_QA_REFERENCE),
        ("finetune.md", DOC_MESSY_REALWORLD),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up ZPF pipeline
        zpf_pipeline = RAGPipeline(store_dir=os.path.join(tmpdir, "zpf_store"))

        # Set up plain baseline
        embedder = Embedder()
        plain_search = PlainVectorSearch(embedder)

        # Ingest into both
        print("Ingesting 5 documents...")
        for filename, content in documents:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            zpf_pipeline.ingest(filepath)
            plain_search.ingest(content)

        zpf_stats = zpf_pipeline.stats
        print(
            f"  ZPF:   {zpf_stats['total_documents']} docs, {zpf_stats['total_blocks']} semantic blocks"
        )
        print(f"  Plain: {len(plain_search.chunks)} fixed-size chunks")
        print()

        # Run at two budgets
        all_results = {}

        for budget_label, max_tok in [("200 tokens", 200), ("500 tokens", 500)]:
            print("=" * 80)
            print(f"  TOKEN BUDGET: {budget_label}")
            print("=" * 80)
            print()
            print(f"{'#':<4} {'Group':<15} {'Question':<42} {'ZPF':>5} {'Plain':>6}")
            print("-" * 80)

            zpf_results = []
            plain_results = []

            current_group = ""
            for qa in QA_PAIRS:
                qid = qa["id"]
                group = qa["group"]
                question = qa["question"]
                fragments = qa["answer_fragments"]

                # Print group separator
                if group != current_group:
                    if current_group:
                        print()
                    current_group = group

                # Retrieve context
                zpf_ctx = zpf_pipeline.get_context(question, max_tokens=max_tok, top_k=3)
                plain_ctx = plain_search.get_context(question, max_tokens=max_tok, top_k=3)

                zpf_score = score_answer(zpf_ctx, fragments)
                plain_score = score_answer(plain_ctx, fragments)

                zpf_results.append(zpf_score)
                plain_results.append(plain_score)

                # Format markers
                zpf_mark = {"HIT": "  \u2705", "PART": "  \u26a0\ufe0f", "MISS": "  \u274c"}[
                    zpf_score
                ]
                plain_mark = {"HIT": "   \u2705", "PART": "   \u26a0\ufe0f", "MISS": "   \u274c"}[
                    plain_score
                ]

                q_short = question[:40] + ".." if len(question) > 42 else question
                print(f"{qid:<4} {group:<15} {q_short:<42} {zpf_mark} {plain_mark}")

            print()
            print("-" * 80)

            # Tally
            zpf_points = sum(score_to_points(s) for s in zpf_results)
            plain_points = sum(score_to_points(s) for s in plain_results)

            zpf_hits = zpf_results.count("HIT")
            zpf_parts = zpf_results.count("PART")
            zpf_misses = zpf_results.count("MISS")

            plain_hits = plain_results.count("HIT")
            plain_parts = plain_results.count("PART")
            plain_misses = plain_results.count("MISS")

            print(
                f"  ZPF:   {zpf_hits} HIT | {zpf_parts} PARTIAL | {zpf_misses} MISS  —  Score: {zpf_points:.1f}/15"
            )
            print(
                f"  Plain: {plain_hits} HIT | {plain_parts} PARTIAL | {plain_misses} MISS  —  Score: {plain_points:.1f}/15"
            )
            print()

            all_results[budget_label] = {
                "zpf": zpf_results,
                "plain": plain_results,
                "zpf_points": zpf_points,
                "plain_points": plain_points,
            }

            # Group breakdown
            groups = ["Definition", "Procedure", "Detail", "Cross-section", "Noise-adjacent"]
            print(f"  {'Group':<16} {'ZPF':>12} {'Plain':>12}")
            print(f"  {'-' * 40}")
            for g in groups:
                g_indices = [i for i, qa in enumerate(QA_PAIRS) if qa["group"] == g]
                zpf_g = sum(score_to_points(zpf_results[i]) for i in g_indices)
                plain_g = sum(score_to_points(plain_results[i]) for i in g_indices)
                g_total = len(g_indices)
                print(f"  {g:<16} {zpf_g:.1f}/{g_total}        {plain_g:.1f}/{g_total}")
            print()

            # Detail on misses/partials
            problems = [
                (i, qa, zpf_results[i], plain_results[i])
                for i, qa in enumerate(QA_PAIRS)
                if zpf_results[i] != "HIT" or plain_results[i] != "HIT"
            ]
            if problems:
                print("  NON-PERFECT RETRIEVALS:")
                for i, qa, zs, ps in problems:
                    who = []
                    if zs != "HIT":
                        who.append(f"ZPF={zs}")
                    if ps != "HIT":
                        who.append(f"Plain={ps}")
                    print(f"    Q{qa['id']}: {qa['question'][:55]}")
                    print(f"         Fragments: {qa['answer_fragments']}")
                    print(f"         {' | '.join(who)}")
                    # Show which fragments were found/missing
                    zpf_ctx = zpf_pipeline.get_context(qa["question"], max_tokens=max_tok, top_k=3)
                    plain_ctx = plain_search.get_context(
                        qa["question"], max_tokens=max_tok, top_k=3
                    )
                    for frag in qa["answer_fragments"]:
                        zf = "\u2705" if frag.lower() in zpf_ctx.lower() else "\u274c"
                        pf = "\u2705" if frag.lower() in plain_ctx.lower() else "\u274c"
                        print(f"           '{frag}': ZPF {zf}  Plain {pf}")
                    print()

        # === Final Summary ===
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print()
        print(f"  {'Budget':<16} {'ZPF Score':>12} {'Plain Score':>14} {'Delta':>8}")
        print(f"  {'-' * 52}")
        for budget_label in ["200 tokens", "500 tokens"]:
            r = all_results[budget_label]
            delta = r["zpf_points"] - r["plain_points"]
            sign = "+" if delta >= 0 else ""
            print(
                f"  {budget_label:<16} {r['zpf_points']:>8.1f}/15   {r['plain_points']:>8.1f}/15   {sign}{delta:.1f}"
            )
        print()

        # Token usage
        print("  TOKEN USAGE (total across 15 queries):")
        for budget_label, max_tok in [("200 tokens", 200), ("500 tokens", 500)]:
            zpf_total = 0
            plain_total = 0
            for qa in QA_PAIRS:
                zpf_total += (
                    len(zpf_pipeline.get_context(qa["question"], max_tokens=max_tok, top_k=3)) // 4
                )
                plain_total += (
                    len(plain_search.get_context(qa["question"], max_tokens=max_tok, top_k=3)) // 4
                )
            print(f"    {budget_label}: ZPF {zpf_total:,} tokens | Plain {plain_total:,} tokens")
        print()


if __name__ == "__main__":
    run_eval()
