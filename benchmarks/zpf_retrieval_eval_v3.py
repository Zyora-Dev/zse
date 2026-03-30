"""
ZPF Retrieval Quality Evaluation v3 — Real Documentation Test

Instead of synthetic test documents, ingests actual ZSE repository files:
  1. progress.md      (~1700 lines) — Full development progress & feature docs
  2. README.md         (~400 lines) — Installation, VRAM benchmarks, GPU compat
  3. deploy/DEPLOY.md  (~200 lines) — Deployment guide, env vars, cloud platforms

15 concrete questions about ZSE features, with answer fragments
verified against actual file contents.

Comparison: ZPF semantic retrieval vs plain 512-char fixed chunking.
Same embedding model (all-MiniLM-L6-v2) for both systems.
Two token budgets: 200 (tight) and 500 (normal).

Scoring:
  HIT  = all answer fragments found in retrieved context  = 1.0
  PART = some but not all fragments found                 = 0.5
  MISS = no answer fragments found                        = 0.0
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Locate repo root (script is in benchmarks/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Real documentation files from the ZSE repository
DOC_FILES = [
    ("progress.md", REPO_ROOT / "progress.md"),
    ("README.md", REPO_ROOT / "README.md"),
    ("DEPLOY.md", REPO_ROOT / "deploy" / "DEPLOY.md"),
]


# ============================================================================
# 15 Ground-Truth QA Pairs — questions from the user, fragments from real docs
# ============================================================================

QA_PAIRS = [
    # === Group 1: Definition ===
    {
        "id": 1,
        "group": "Definition",
        "question": "What is tensor parallelism?",
        "answer_fragments": ["NCCL all-reduce"],
    },
    {
        "id": 2,
        "group": "Definition",
        "question": "What is the difference between pipeline parallelism and tensor parallelism?",
        "answer_fragments": ["doesn't parallelize compute", "all-reduce"],
    },
    {
        "id": 3,
        "group": "Definition",
        "question": "What is a .zpf file?",
        "answer_fragments": ["binary format", "document storage"],
    },

    # === Group 2: Procedure ===
    {
        "id": 4,
        "group": "Procedure",
        "question": "How do you install zllm-zse with CUDA support?",
        "answer_fragments": ["pip install zllm-zse"],
    },
    {
        "id": 5,
        "group": "Procedure",
        "question": "What command serves a model with tensor parallelism on 2 GPUs?",
        "answer_fragments": ["tensor-parallel 2"],
    },
    {
        "id": 6,
        "group": "Procedure",
        "question": "How do you export a .zpf file to markdown format?",
        "answer_fragments": ["rag export", "format markdown"],
    },

    # === Group 3: Specific Detail ===
    {
        "id": 7,
        "group": "Detail",
        "question": "How much VRAM does a 7B INT4 model use?",
        "answer_fragments": ["5.67 GB"],
    },
    {
        "id": 8,
        "group": "Detail",
        "question": "What happens when KV cache is exhausted?",
        "answer_fragments": ["memory pressure"],
    },
    {
        "id": 9,
        "group": "Detail",
        "question": "What environment variable controls KV cache quantization?",
        "answer_fragments": ["Quantized KV Cache"],
    },

    # === Group 4: Cross-section ===
    {
        "id": 10,
        "group": "Cross-section",
        "question": "What are the hardware requirements to run a 32B model?",
        "answer_fragments": ["19.47", "RTX 3090"],
    },
    {
        "id": 11,
        "group": "Cross-section",
        "question": "What are the limitations of using TP across GPUs that already fit on one GPU?",
        "answer_fragments": ["0.80x", "expected overhead"],
    },
    {
        "id": 12,
        "group": "Cross-section",
        "question": "What parallelism strategy gives the best VRAM reduction per GPU?",
        "answer_fragments": ["65%"],
    },

    # === Group 5: Noise-adjacent ===
    {
        "id": 13,
        "group": "Noise-adjacent",
        "question": "What is the purpose of the radix prefix cache?",
        "answer_fragments": ["radix tree", "prompt reuse"],
    },
    {
        "id": 14,
        "group": "Noise-adjacent",
        "question": "What embedding model does .zpf use by default?",
        "answer_fragments": ["all-MiniLM-L6-v2"],
    },
    {
        "id": 15,
        "group": "Noise-adjacent",
        "question": "What is the speedup of TP=2 on a 7B model that fits on a single GPU?",
        "answer_fragments": ["0.80x"],
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
    return {"HIT": 1.0, "PART": 0.5, "MISS": 0.0}[score]


# ============================================================================
# Main evaluation
# ============================================================================

def run_eval():
    print("=" * 80)
    print("ZPF RETRIEVAL QUALITY EVALUATION v3 — Real Documentation Test")
    print("=" * 80)
    print()
    print("3 real repo files | 15 questions | 5 groups | 2 token budgets")
    print("ZPF semantic retrieval vs plain 512-char fixed chunking")
    print("Same embedding model: all-MiniLM-L6-v2")
    print()

    # Verify all doc files exist
    for label, path in DOC_FILES:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)
        size_kb = path.stat().st_size / 1024
        lines = len(path.read_text().splitlines())
        print(f"  {label:<20} {lines:>5} lines  {size_kb:>6.1f} KB  {path}")
    print()

    from zse.core.zrag.pipeline import RAGPipeline
    from zse.core.zrag.embedder import Embedder

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up ZPF pipeline
        zpf_pipeline = RAGPipeline(store_dir=os.path.join(tmpdir, "zpf_store"))

        # Set up plain baseline
        embedder = Embedder()
        plain_search = PlainVectorSearch(embedder)

        # Ingest real docs into both systems
        print("Ingesting 3 real documentation files...")
        for label, path in DOC_FILES:
            zpf_pipeline.ingest(str(path))
            plain_search.ingest(path.read_text())

        zpf_stats = zpf_pipeline.stats
        print(f"  ZPF:   {zpf_stats['total_documents']} docs, {zpf_stats['total_blocks']} semantic blocks")
        print(f"  Plain: {len(plain_search.chunks)} fixed-size chunks")
        print()

        # Run at two budgets
        all_results = {}

        for budget_label, max_tok in [("200 tokens", 200), ("500 tokens", 500)]:
            print("=" * 80)
            print(f"  TOKEN BUDGET: {budget_label}")
            print("=" * 80)
            print()
            print(f"{'#':<4} {'Group':<15} {'Question':<44} {'ZPF':>5} {'Plain':>6}")
            print("-" * 80)

            zpf_results = []
            plain_results = []

            current_group = ""
            for qa in QA_PAIRS:
                qid = qa["id"]
                group = qa["group"]
                question = qa["question"]
                fragments = qa["answer_fragments"]

                if group != current_group:
                    if current_group:
                        print()
                    current_group = group

                # Retrieve context from both systems
                zpf_ctx = zpf_pipeline.get_context(question, max_tokens=max_tok, top_k=10)
                plain_ctx = plain_search.get_context(question, max_tokens=max_tok, top_k=10)

                zpf_score = score_answer(zpf_ctx, fragments)
                plain_score = score_answer(plain_ctx, fragments)

                zpf_results.append(zpf_score)
                plain_results.append(plain_score)

                zpf_mark = {"HIT": "  \u2705", "PART": "  \u26a0\ufe0f", "MISS": "  \u274c"}[zpf_score]
                plain_mark = {"HIT": "   \u2705", "PART": "   \u26a0\ufe0f", "MISS": "   \u274c"}[plain_score]

                q_short = question[:42] + ".." if len(question) > 44 else question
                print(f"{qid:<4} {group:<15} {q_short:<44} {zpf_mark} {plain_mark}")

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

            print(f"  ZPF:   {zpf_hits} HIT | {zpf_parts} PARTIAL | {zpf_misses} MISS  —  Score: {zpf_points:.1f}/15")
            print(f"  Plain: {plain_hits} HIT | {plain_parts} PARTIAL | {plain_misses} MISS  —  Score: {plain_points:.1f}/15")
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
            print(f"  {'-'*40}")
            for g in groups:
                g_indices = [i for i, qa in enumerate(QA_PAIRS) if qa["group"] == g]
                zpf_g = sum(score_to_points(zpf_results[i]) for i in g_indices)
                plain_g = sum(score_to_points(plain_results[i]) for i in g_indices)
                g_total = len(g_indices)
                print(f"  {g:<16} {zpf_g:.1f}/{g_total}        {plain_g:.1f}/{g_total}")
            print()

            # Detail on misses/partials
            problems = [(i, qa, zpf_results[i], plain_results[i])
                        for i, qa in enumerate(QA_PAIRS)
                        if zpf_results[i] != "HIT" or plain_results[i] != "HIT"]
            if problems:
                print("  NON-PERFECT RETRIEVALS:")
                for i, qa, zs, ps in problems:
                    who = []
                    if zs != "HIT":
                        who.append(f"ZPF={zs}")
                    if ps != "HIT":
                        who.append(f"Plain={ps}")
                    print(f"    Q{qa['id']}: {qa['question'][:57]}")
                    print(f"         Fragments: {qa['answer_fragments']}")
                    print(f"         {' | '.join(who)}")
                    zpf_ctx = zpf_pipeline.get_context(qa["question"], max_tokens=max_tok, top_k=10)
                    plain_ctx = plain_search.get_context(qa["question"], max_tokens=max_tok, top_k=10)
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
        print(f"  {'-'*52}")
        for budget_label in ["200 tokens", "500 tokens"]:
            r = all_results[budget_label]
            delta = r["zpf_points"] - r["plain_points"]
            sign = "+" if delta >= 0 else ""
            print(f"  {budget_label:<16} {r['zpf_points']:>8.1f}/15   {r['plain_points']:>8.1f}/15   {sign}{delta:.1f}")
        print()

        # Token usage
        print("  TOKEN USAGE (total across 15 queries):")
        for budget_label, max_tok in [("200 tokens", 200), ("500 tokens", 500)]:
            zpf_total = 0
            plain_total = 0
            for qa in QA_PAIRS:
                zpf_total += len(zpf_pipeline.get_context(qa["question"], max_tokens=max_tok, top_k=10)) // 4
                plain_total += len(plain_search.get_context(qa["question"], max_tokens=max_tok, top_k=10)) // 4
            print(f"    {budget_label}: ZPF {zpf_total:,} tokens | Plain {plain_total:,} tokens")
        print()

        # Document stats
        print("  DOCUMENT STATS:")
        for label, path in DOC_FILES:
            content = path.read_text()
            print(f"    {label:<20} {len(content):>8,} chars  {len(content)//4:>6,} est. tokens")
        print(f"    {'TOTAL':<20} {sum(p.stat().st_size for _,p in DOC_FILES):>8,} bytes")
        print()


if __name__ == "__main__":
    run_eval()
