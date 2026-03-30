"""
ZPF Cost-Efficiency Benchmark — CNN Noisy Article

The core claim: .zpf reduces tokens sent to the LLM without losing answer quality.
This benchmark measures exactly that.

Test document: CNN "Introduction to Convolutional Neural Networks" web page
  - Heavy noise: cookie banners, nav, social buttons, newsletter, ads, comments
  - ~1,473 raw tokens → ~872 ZPF tokens (40.8% compression in prior bench)

10 questions across 4 groups:
  Group A (3): Specific facts — short-phrase answers
  Group B (2): Lists — enumerated content preservation
  Group C (3): Noise-adjacent — answers near boilerplate
  Group D (2): Cross-section — answers spanning sections

For each query we record:
  - Tokens sent to LLM (the context retrieved)
  - Whether the answer is correct (fragment matching)

The headline number: total tokens across 10 queries, ZPF vs plain.
Same embedding model (all-MiniLM-L6-v2), same top_k, same budget.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# The CNN noisy web page (same as compression bench DOC_NOISY_WEBPAGE)
# ============================================================================

DOC_CNN = """
Skip to main content
Home | About | Products | Blog | Contact | Login | Sign Up

Cookie Policy Notice
We use cookies to enhance your browsing experience and analyze site traffic.
By clicking "Accept All", you consent to our use of cookies.
Accept All | Reject All | Cookie Settings

Loading...

Menu
Search
Close

Share this on Facebook | Share on Twitter | Share on LinkedIn
Follow us: Twitter | GitHub | Discord | YouTube
Subscribe to our newsletter for more AI research updates.

# Introduction to Convolutional Neural Networks

In this article, we will explore the fundamentals of Convolutional Neural Networks
(CNNs). It is important to note that CNNs are a class of deep neural networks that
are particularly effective for analyzing visual imagery. Essentially, they utilize a
mathematical operation called convolution in place of general matrix multiplication
in at least one of their layers. As we mentioned earlier, this makes them very
important for computer vision tasks.

## Architecture Overview

Let's take a look at the architecture of a typical CNN. Generally speaking, it
consists of the following layers. In order to understand how CNNs work, we need
to examine each layer in detail.

### Convolutional Layer
The convolutional layer is basically the core building block of a Convolutional Neural
Network. It has the ability to apply a set of learnable filters to the input. Each
filter is small spatially (with respect to width and height) but extends through the
full depth of the input volume. During the course of the forward pass, each filter is
convolved across the width and height of the input volume, computing the dot product
between the filter entries and the input at each position. It should be noted that
this is where the majority of the computation takes place.

### Pooling Layer
Simply put, the pooling layer reduces the spatial dimensions of the representation,
reducing the number of parameters and computation in the network. The most common
form is max pooling with a 2x2 filter and stride 2, which downsamples every depth
slice by taking the maximum value in each 2x2 block. Due to the fact that it reduces
dimensionality, it helps prevent overfitting.

### Fully Connected Layer
After several convolutional and pooling layers, the high-level reasoning in the
neural network is done via fully connected layers. As the name suggests, neurons in a
fully connected layer have connections to all activations in the previous layer, as
seen in regular neural networks. In simple terms, this is where the final
classification decision is made.

## Key Concepts

In this section, we will discuss some key concepts that are very important to
understand in order to work with Convolutional Neural Networks effectively.

### Stride
Stride refers to the number of pixels the filter moves across the input image.
A stride of 1 means the filter moves one pixel at a time. Having said that,
a stride of 2 means it moves two pixels, resulting in a smaller output. It is
important to note that larger strides result in smaller output dimensions.

### Padding
Padding is essentially the process of adding zeros around the border of the input.
Valid padding means no padding (i.e., the output shrinks). Same padding adds enough
zeros so the output has the same spatial dimensions as the input. With that in mind,
it is clear that padding is used to control the output size.

### Receptive Field
The receptive field is basically the region of the input that influences a particular
feature in the output. It is worth noting that deeper layers have larger receptive
fields, allowing them to capture more global patterns.

## Applications

Convolutional Neural Networks are used extensively in a large number of applications:
- Image classification (ResNet, VGG, EfficientNet)
- Object detection (YOLO, Faster R-CNN, SSD)
- Semantic segmentation (U-Net, DeepLab)
- Face recognition (FaceNet, ArcFace)
- Medical imaging (tumor detection, X-ray analysis)
- Autonomous driving (lane detection, pedestrian detection)

## Training Considerations

### Data Augmentation
In order to prevent overfitting, training data is commonly augmented with random
transformations: horizontal flips, random crops, color jittering, rotation,
and scaling. As a matter of fact, this effectively increases the training set size
without requiring more labeled data. It is important to note that augmentation should
be task-appropriate.

### Batch Normalization
Batch normalization essentially normalizes the inputs to each layer, which facilitates
more stable and accelerated training. It is able to reduce internal covariate shift and
allows higher learning rates. As we discussed earlier, it is applied after convolution
and prior to activation.

### Transfer Learning
Pre-trained Convolutional Neural Networks (trained on ImageNet) can be fine-tuned for
new tasks. It should be noted that the lower layers learn general features (that is,
edges, textures) that transfer well across tasks. Needless to say, only the top layers
need retraining for the specific task at hand. This means that transfer learning is
extremely important for practical applications with limited data.

Page 1 of 1

Related Articles:
- Understanding BERT: A Complete Guide
- GPT Architecture Explained
- Vision Transformers: Images as Sequences

See Also
Recommended Articles

Copyright 2026 DeepLearn AI Inc. All Rights Reserved.
Terms of Service | Privacy Policy | Cookie Policy

Sign up for our newsletter
Enter your email: [____________] [Subscribe]

Comments (0)
Be the first to comment! Login to leave a comment.
Leave a reply

Tags: deep-learning, CNN, computer-vision, neural-networks
Categories: machine-learning, tutorials, ai

Share this on Facebook | Share on Twitter | Share on LinkedIn
Tweet this
Like this on Facebook

Back to top
Read more...
Show more
View all

Advertisement
Sponsored content

Loading...
"""


# ============================================================================
# 10 Questions — answer fragments verified against the document above
# ============================================================================

QA_PAIRS = [
    # === Group A: Specific facts ===
    {
        "id": 1,
        "group": "A: Specific fact",
        "question": "What are the three main layers of a CNN architecture?",
        "answer_fragments": ["convolutional", "pooling", "fully connected"],
    },
    {
        "id": 2,
        "group": "A: Specific fact",
        "question": "What does stride control in a convolutional layer?",
        "answer_fragments": ["filter moves"],
    },
    {
        "id": 3,
        "group": "A: Specific fact",
        "question": "What is a receptive field?",
        "answer_fragments": ["region of the input"],
    },

    # === Group B: Lists ===
    {
        "id": 4,
        "group": "B: Lists",
        "question": "Name three applications of CNNs.",
        # Need at least 3 of these 6
        "answer_fragments": ["image classification", "object detection"],
    },
    {
        "id": 5,
        "group": "B: Lists",
        "question": "What techniques are used in CNN training?",
        "answer_fragments": ["augmentation", "batch normalization", "transfer learning"],
    },

    # === Group C: Noise-adjacent ===
    {
        "id": 6,
        "group": "C: Noise-adjacent",
        "question": "What is padding used for in a CNN?",
        "answer_fragments": ["same spatial dimensions"],
    },
    {
        "id": 7,
        "group": "C: Noise-adjacent",
        "question": "What is batch normalization?",
        "answer_fragments": ["normalizes the inputs"],
    },
    {
        "id": 8,
        "group": "C: Noise-adjacent",
        "question": "What is transfer learning in the context of CNNs?",
        "answer_fragments": ["fine-tuned"],
    },

    # === Group D: Cross-section ===
    {
        "id": 9,
        "group": "D: Cross-section",
        "question": "Why is pooling used after a convolutional layer?",
        "answer_fragments": ["reduces the spatial dimensions"],
    },
    {
        "id": 10,
        "group": "D: Cross-section",
        "question": "How does data augmentation help CNN training?",
        "answer_fragments": ["prevent overfitting"],
    },
]


# ============================================================================
# Plain fixed-size chunker baseline
# ============================================================================

class PlainChunker:
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

def score_answer(context: str, answer_fragments: List[str]) -> Tuple[str, int, int]:
    """Returns (verdict, found_count, total_count)."""
    ctx_lower = context.lower()
    found = sum(1 for frag in answer_fragments if frag.lower() in ctx_lower)
    total = len(answer_fragments)
    if found == total:
        return "HIT", found, total
    elif found > 0:
        return "PART", found, total
    else:
        return "MISS", found, total


def count_tokens(text: str) -> int:
    """Estimate token count (len/4 approximation, same as used in get_context)."""
    return len(text) // 4


# ============================================================================
# Main benchmark
# ============================================================================

def run_cost_benchmark():
    print("=" * 80)
    print("ZPF COST-EFFICIENCY BENCHMARK — CNN Noisy Article")
    print("=" * 80)
    print()
    print("Core claim: .zpf reduces tokens sent to LLM without losing answer quality")
    print("Document: CNN 'Introduction to Convolutional Neural Networks' (noisy web page)")
    print("10 questions | 4 groups | Same embedding model: all-MiniLM-L6-v2")
    print()

    # Document stats
    raw_tokens = count_tokens(DOC_CNN)
    raw_chars = len(DOC_CNN)
    print(f"  Raw document: {raw_chars:,} chars, {raw_tokens:,} est. tokens")
    print()

    from zse.core.zrag.pipeline import RAGPipeline
    from zse.core.zrag.embedder import Embedder

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Set up ZPF pipeline ---
        zpf_pipeline = RAGPipeline(store_dir=os.path.join(tmpdir, "zpf_store"))

        # Write CNN doc to temp file for ingestion
        cnn_path = os.path.join(tmpdir, "cnn_noisy.md")
        with open(cnn_path, "w") as f:
            f.write(DOC_CNN)

        zpf_pipeline.ingest(cnn_path)

        # --- Set up plain baseline ---
        embedder = Embedder()
        plain_search = PlainVectorSearch(embedder)
        plain_search.ingest(DOC_CNN)

        zpf_stats = zpf_pipeline.stats
        print(f"  ZPF:   {zpf_stats['total_blocks']} semantic blocks")
        print(f"  Plain: {len(plain_search.chunks)} fixed-size chunks")
        print()

        # Show what ZPF compressed to
        zpf_total_stored = sum(
            len(entry["content"])
            for entry in zpf_pipeline._store._index
        )
        zpf_stored_tokens = zpf_total_stored // 4
        plain_total_stored = sum(len(c) for c in plain_search.chunks)
        plain_stored_tokens = plain_total_stored // 4
        print(f"  ZPF stored content:   {zpf_stored_tokens:,} tokens (compressed)")
        print(f"  Plain stored content: {plain_stored_tokens:,} tokens (raw)")
        compression_pct = (1 - zpf_stored_tokens / plain_stored_tokens) * 100
        print(f"  Storage reduction:    {compression_pct:.1f}%")
        print()

        # === Run at a generous budget (no artificial cap — send ALL relevant context) ===
        # Use max_tokens=2000 so neither system is budget-constrained.
        # The test is: how many tokens does each system ACTUALLY send?
        MAX_TOKENS = 2000
        TOP_K = 10

        print("=" * 80)
        print(f"  RETRIEVAL: top_k={TOP_K}, max_tokens={MAX_TOKENS} (unconstrained)")
        print("=" * 80)
        print()

        header = (
            f"{'Q':<3} {'Group':<18} {'Question':<42} "
            f"{'ZPF tok':>8} {'ZPF':>5} {'Plain tok':>10} {'Plain':>6}"
        )
        print(header)
        print("-" * len(header))

        total_zpf_tokens = 0
        total_plain_tokens = 0
        zpf_correct = 0
        plain_correct = 0
        zpf_partial = 0
        plain_partial = 0

        per_query_results = []

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
            zpf_ctx = zpf_pipeline.get_context(
                question, max_tokens=MAX_TOKENS, top_k=TOP_K
            )
            plain_ctx = plain_search.get_context(
                question, max_tokens=MAX_TOKENS, top_k=TOP_K
            )

            # Count tokens actually sent
            zpf_tok = count_tokens(zpf_ctx)
            plain_tok = count_tokens(plain_ctx)
            total_zpf_tokens += zpf_tok
            total_plain_tokens += plain_tok

            # Score
            zpf_verdict, _, _ = score_answer(zpf_ctx, fragments)
            plain_verdict, _, _ = score_answer(plain_ctx, fragments)

            if zpf_verdict == "HIT":
                zpf_correct += 1
            elif zpf_verdict == "PART":
                zpf_partial += 1
            if plain_verdict == "HIT":
                plain_correct += 1
            elif plain_verdict == "PART":
                plain_partial += 1

            zpf_mark = {"HIT": "  \u2705", "PART": "  \u26a0\ufe0f", "MISS": "  \u274c"}[zpf_verdict]
            plain_mark = {"HIT": "   \u2705", "PART": "   \u26a0\ufe0f", "MISS": "   \u274c"}[plain_verdict]

            q_short = question[:40] + ".." if len(question) > 42 else question
            print(
                f"{qid:<3} {group:<18} {q_short:<42} "
                f"{zpf_tok:>7}  {zpf_mark} {plain_tok:>9}  {plain_mark}"
            )

            per_query_results.append({
                "id": qid,
                "group": group,
                "question": question,
                "zpf_tokens": zpf_tok,
                "zpf_verdict": zpf_verdict,
                "plain_tokens": plain_tok,
                "plain_verdict": plain_verdict,
            })

        # === Summary ===
        print()
        print("=" * 80)
        print("COST-EFFICIENCY SUMMARY")
        print("=" * 80)
        print()

        print(f"  {'Metric':<35} {'ZPF':>12} {'Plain':>12} {'Delta':>10}")
        print(f"  {'-'*69}")
        print(f"  {'Correct answers (HIT)':<35} {zpf_correct:>8}/10   {plain_correct:>8}/10")
        print(f"  {'Partial answers':<35} {zpf_partial:>8}/10   {plain_partial:>8}/10")
        print(f"  {'Total tokens sent (10 queries)':<35} {total_zpf_tokens:>8,}   {total_plain_tokens:>8,}")

        if total_plain_tokens > 0:
            token_reduction = (1 - total_zpf_tokens / total_plain_tokens) * 100
            print(f"  {'Token reduction':<35} {token_reduction:>+7.1f}%")
        print()

        # Per-query token comparison
        print(f"  {'Tokens per query (avg)':<35} {total_zpf_tokens/10:>8.0f}   {total_plain_tokens/10:>8.0f}")
        print()

        # Cost projection
        # GPT-4o pricing: $2.50/M input tokens (as of 2026)
        gpt4o_price_per_m = 2.50
        zpf_cost = total_zpf_tokens / 1_000_000 * gpt4o_price_per_m * 1000  # per 1000 queries
        plain_cost = total_plain_tokens / 1_000_000 * gpt4o_price_per_m * 1000
        print(f"  COST PROJECTION (GPT-4o @ ${gpt4o_price_per_m}/M input tokens):")
        print(f"    Per 1,000 queries: ZPF ${zpf_cost:.4f} | Plain ${plain_cost:.4f}")
        if plain_cost > 0:
            savings = (1 - zpf_cost / plain_cost) * 100
            print(f"    Cost savings: {savings:+.1f}%")
        print()

        # Per 1M queries
        zpf_cost_1m = total_zpf_tokens / 10 / 1_000_000 * gpt4o_price_per_m * 1_000_000
        plain_cost_1m = total_plain_tokens / 10 / 1_000_000 * gpt4o_price_per_m * 1_000_000
        print(f"    Per 1,000,000 queries: ZPF ${zpf_cost_1m:.2f} | Plain ${plain_cost_1m:.2f}")
        if plain_cost_1m > 0:
            savings_1m = plain_cost_1m - zpf_cost_1m
            print(f"    Annual savings at 1M queries: ${savings_1m:.2f}")
        print()

        # The verdict
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print()
        if zpf_correct >= plain_correct and total_zpf_tokens < total_plain_tokens:
            print("  \u2705 ZPF DELIVERS: same or better answers at fewer tokens.")
            print(f"     {zpf_correct}/10 correct at {total_zpf_tokens:,} tokens")
            print(f"     vs {plain_correct}/10 correct at {total_plain_tokens:,} tokens")
            print(f"     Token reduction: {token_reduction:.1f}%")
        elif total_zpf_tokens < total_plain_tokens:
            quality_diff = plain_correct - zpf_correct
            print(f"  \u26a0\ufe0f  ZPF TRADES: {token_reduction:.1f}% fewer tokens but loses {quality_diff} answer(s).")
            print(f"     {zpf_correct}/10 correct at {total_zpf_tokens:,} tokens")
            print(f"     vs {plain_correct}/10 correct at {total_plain_tokens:,} tokens")
        else:
            print(f"  \u274c ZPF DOES NOT SAVE TOKENS on this document.")
            print(f"     {zpf_correct}/10 correct at {total_zpf_tokens:,} tokens")
            print(f"     vs {plain_correct}/10 correct at {total_plain_tokens:,} tokens")
        print()

        # === Detailed breakdown: what each system actually returns ===
        print("=" * 80)
        print("DETAILED RETRIEVAL (showing context for non-perfect queries)")
        print("=" * 80)
        for r in per_query_results:
            if r["zpf_verdict"] != "HIT" or r["plain_verdict"] != "HIT":
                qa = next(q for q in QA_PAIRS if q["id"] == r["id"])
                print(f"\n  Q{r['id']}: {r['question']}")
                print(f"    Fragments: {qa['answer_fragments']}")
                print(f"    ZPF:   {r['zpf_verdict']} ({r['zpf_tokens']} tokens)")
                print(f"    Plain: {r['plain_verdict']} ({r['plain_tokens']} tokens)")

                zpf_ctx = zpf_pipeline.get_context(
                    r["question"], max_tokens=MAX_TOKENS, top_k=TOP_K
                )
                plain_ctx = plain_search.get_context(
                    r["question"], max_tokens=MAX_TOKENS, top_k=TOP_K
                )
                for frag in qa["answer_fragments"]:
                    zf = "\u2705" if frag.lower() in zpf_ctx.lower() else "\u274c"
                    pf = "\u2705" if frag.lower() in plain_ctx.lower() else "\u274c"
                    print(f"      '{frag}': ZPF {zf}  Plain {pf}")
        print()


if __name__ == "__main__":
    run_cost_benchmark()
