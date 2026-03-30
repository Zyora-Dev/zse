"""
ZPF Compression Benchmark — Real measurement with genuinely different documents.

Uses 3 completely different documents (not duplicated content) and measures:
1. Raw token count (original document)
2. ZPF token count (after semantic compression)
3. What was stripped (noise audit)
4. Comparison with naive 512-token fixed chunking
"""

import os
import sys
import tempfile
import time

# ============================================================================
# DOCUMENT 1: Clean technical article (minimal noise)
# ============================================================================
DOC_CLEAN_ARTICLE = """# Gradient Descent Optimization

## Definition
Gradient descent is an iterative optimization algorithm used to find the minimum
of a differentiable function. It works by repeatedly taking steps proportional
to the negative of the gradient of the function at the current point.

## Variants

### Batch Gradient Descent
Computes the gradient using the entire training dataset. This provides a stable
convergence path but is computationally expensive for large datasets. The update
rule is:

    theta = theta - learning_rate * (1/N) * sum(gradient(loss(x_i, y_i)))

### Stochastic Gradient Descent (SGD)
Updates parameters using a single randomly selected sample at each step. This
introduces noise but enables faster updates and can escape local minima.

### Mini-batch Gradient Descent
A compromise between batch and stochastic methods. Uses a small random subset
(typically 32-256 samples) for each update. This balances computation efficiency
with gradient estimate quality.

## Learning Rate Schedules

The learning rate is the most important hyperparameter to tune:

1. Constant learning rate — simplest but often suboptimal
2. Step decay — reduce by a factor every N epochs
3. Exponential decay — lr = lr_0 * exp(-decay_rate * epoch)
4. Cosine annealing — lr follows a cosine curve from max to min
5. Warmup + decay — linearly increase then decrease (used in transformers)

## Momentum

Momentum accelerates SGD by accumulating a velocity vector in directions of
persistent gradient reduction. The update becomes:

    v = gamma * v + learning_rate * gradient
    theta = theta - v

Where gamma is typically 0.9. This helps navigate ravines in the loss surface
where the gradient oscillates.

## Adam Optimizer

Adam combines momentum with adaptive per-parameter learning rates:

    m = beta1 * m + (1 - beta1) * gradient        # first moment
    v = beta2 * v + (1 - beta2) * gradient^2       # second moment
    m_hat = m / (1 - beta1^t)                      # bias correction
    v_hat = v / (1 - beta2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

Default values: beta1=0.9, beta2=0.999, epsilon=1e-8.
"""

# ============================================================================
# DOCUMENT 2: Noisy web page (heavy boilerplate, nav, cookie banners)
# ============================================================================
DOC_NOISY_WEBPAGE = """
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
# DOCUMENT 3: Long structured documentation with tables, code, and procedures
# ============================================================================
DOC_LONG_STRUCTURED = """# ZSE Deployment Guide

## Overview

In this guide, we will discuss the deployment process for ZSE (Z Server Engine),
which is a production-first GPU cluster LLM serving engine built for memory efficiency
and high throughput. It is important to note that this guide covers installation,
configuration, deployment, and monitoring for production use. As a matter of fact,
ZSE is able to serve models of various sizes from a single GPU to multi-node clusters.

## Prerequisites

Prior to deploying ZSE, it is important to ensure the following requirements are met.
In order to run ZSE successfully, you will need:

- Python 3.10 or higher
- NVIDIA GPU with compute capability 7.0+ (V100, A100, H100, A10G)
- CUDA 11.8 or higher
- At least 16 GB system RAM
- Linux (Ubuntu 20.04+ recommended) or macOS for development

## Installation

### Step 1: Install ZSE

In order to install ZSE, you can utilize PyPI:

    pip install zllm-zse
    pip install zllm-zse[cuda]  # with CUDA support

### Step 2: Verify Installation

Subsequently, you should run the hardware check in order to verify the installation:

    zse hardware

This shows detected GPUs, available VRAM, and recommended configurations. It is
worth noting that this step helps identify any driver or hardware issues prior to
deployment.

### Step 3: Login to HuggingFace (Optional)

In the event that you need to leverage gated models like Llama:

    zse login
    # Enter your HuggingFace token

### Step 4: Pull a Model

    zse pull qwen2.5-7b-instruct

## Configuration

### Memory Modes

It is important to note that ZSE provides multiple memory efficiency modes. The
following table shows the available configurations:

| Mode | Quantization | Use Case | VRAM (7B) |
|------|-------------|----------|-----------|
| speed | FP16 | Fastest inference, most VRAM | 14 GB |
| balanced | INT8 | Good balance of speed and memory | 7 GB |
| memory | INT4 | Memory efficient, slight quality trade | 3.5 GB |
| ultra | INT4+streaming | Extreme memory savings | 2.5 GB |

### Server Configuration

In order to launch the ZSE server, you can utilize the following command:

    zse serve qwen-7b \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --efficiency balanced \\
        --max-memory 8GB \\
        --tensor-parallel 2

### Environment Variables

The following environment variables can be used to configure ZSE. It should be
noted that all of these have sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| ZSE_CACHE_DIR | ~/.zse/cache | Model cache directory |
| ZSE_LOG_LEVEL | INFO | Logging verbosity |
| ZSE_MAX_BATCH | 32 | Maximum concurrent batch size |
| ZSE_KV_QUANT | int8 | KV cache quantization level |

## Monitoring

### Health Check Endpoint

In order to check the health of a running ZSE instance:

    GET /health

Returns:

    {
        "status": "healthy",
        "model": "qwen2.5-7b-instruct",
        "uptime_seconds": 3600,
        "requests_served": 1250,
        "gpu_utilization": 0.45,
        "vram_used_gb": 3.8
    }

### Metrics

ZSE exposes Prometheus-compatible metrics at /metrics. It is important to note
that these metrics facilitate comprehensive monitoring:

- zse_requests_total: Total request count
- zse_tokens_generated: Total tokens generated
- zse_latency_seconds: Request latency histogram
- zse_vram_bytes: Current VRAM usage
- zse_batch_size: Current batch size

### Logging

ZSE logs structured JSON to stderr. Essentially, each log line contains:

    {"level": "INFO", "msg": "Request completed", "tokens": 128, "latency_ms": 450}

## Scaling

### Horizontal Scaling

In order to scale horizontally, you should deploy multiple ZSE instances behind
a load balancer. Each instance is able to serve independently. It is recommended
to utilize round-robin or least-connections balancing.

### Tensor Parallelism

Tensor Parallelism (TP) has the ability to split a single model across multiple GPUs
on one node. In order to utilize Tensor Parallelism:

    zse serve llama-70b --tensor-parallel 4

It is important to note that this requires GPUs connected via NVLink or PCIe with
NCCL support.

### Pipeline Parallelism

Pipeline Parallelism (PP) facilitates splitting model layers across GPUs sequentially:

    zse serve llama-70b --pipeline-parallel 4

Due to the fact that Pipeline Parallelism has a lower inter-GPU bandwidth requirement
than Tensor Parallelism, it is a good choice for PCIe setups.

### Combined TP+PP

For the purpose of achieving maximum scale, you can leverage both Tensor Parallelism
and Pipeline Parallelism simultaneously:

    zse serve llama-70b -tp 2 -pp 4  # 8 GPUs total

## Troubleshooting

### Common Issues

As we discussed earlier, there are several common issues you may encounter:

1. **Out of VRAM**: In order to resolve this, reduce batch size, switch to a more
   aggressive memory mode, or utilize streaming (ultra mode).

2. **Slow first request**: It should be noted that cold start includes model loading.
   Make use of the --preload flag or set up a warm-up script that sends a dummy request
   after launch.

3. **NCCL errors with multi-GPU**: Ensure all GPUs are visible
   (CUDA_VISIBLE_DEVICES), and NCCL is able to access the interconnect.
   Set NCCL_DEBUG=INFO for diagnostics. Needless to say, proper NCCL setup is
   critical for multi-GPU deployments.

4. **Quantization quality issues**: INT4 quantization is capable of degrading output
   quality for reasoning-heavy tasks. Make use of INT8 or FP16 for those workloads.

5. **KV cache exhaustion**: For very long contexts, the KV cache may fill up.
   In order to fix this, increase --max-memory or reduce --max-context-length.

## Security

### API Key Authentication

In order to enable API key authentication:

    zse api-key generate --name production

Pass the key in requests:

    curl -H "Authorization: Bearer zse_xxxx" http://localhost:8000/v1/chat/completions

### Network Security

It is important to note that network security is critical for production deployments:

- Bind to 127.0.0.1 for local-only access
- Make use of a reverse proxy (nginx, caddy) for TLS termination
- Enable rate limiting at the proxy level
- Needless to say, never expose the ZSE port directly to the internet without authentication

Read more...
Back to top
Share this on Facebook | Share on Twitter
Related Articles
Recommended reading
View all
"""


def estimate_tokens(text):
    """Same as chunker: ~4 chars per token."""
    return max(1, len(text) // 4)


def naive_chunk(text, chunk_size=512):
    """Naive fixed-size chunking (baseline comparison)."""
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0
    for word in words:
        word_tokens = max(1, len(word) // 4)
        if current_tokens + word_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def run_benchmark():
    print("=" * 75)
    print("ZPF SEMANTIC COMPRESSION BENCHMARK (v2 — real different documents)")
    print("=" * 75)
    print()

    from zse.core.zrag.pipeline import RAGPipeline
    from zse.core.zrag.zpf_reader import ZPFReader
    from zse.core.zrag.semantic_chunker import SemanticChunker, _is_noise
    from zse.core.zrag.parsers import parse_file

    documents = [
        ("Doc 1: Clean article (gradient descent)", DOC_CLEAN_ARTICLE, "gradient.md"),
        ("Doc 2: Noisy web page (CNN guide)", DOC_NOISY_WEBPAGE, "cnn_noisy.md"),
        ("Doc 3: Structured docs (deployment)", DOC_LONG_STRUCTURED, "deployment.md"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = RAGPipeline(store_dir=os.path.join(tmpdir, "store"))
        results = []

        for label, content, filename in documents:
            raw_bytes = len(content.encode("utf-8"))
            raw_tokens = estimate_tokens(content)

            # Write file
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(content)

            # Measure noise
            lines = content.split("\n")
            noise_lines = [l for l in lines if l.strip() and _is_noise(l)]
            noise_chars = sum(len(l) for l in noise_lines)

            # Convert to .zpf
            zpf_path = os.path.join(tmpdir, filename.replace(".md", ".zpf"))
            t0 = time.time()
            pipeline.convert(filepath, output_path=zpf_path)
            convert_time = time.time() - t0

            # Read .zpf
            reader = ZPFReader(zpf_path)
            zpf_tokens = reader.header.total_tokens
            zpf_bytes = os.path.getsize(zpf_path)
            block_count = reader.header.block_count

            # Naive chunking baseline
            naive_chunks = naive_chunk(content, chunk_size=512)
            naive_tokens = sum(estimate_tokens(c) for c in naive_chunks)

            token_reduction = (1 - zpf_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0
            vs_naive = (1 - zpf_tokens / naive_tokens) * 100 if naive_tokens > 0 else 0

            results.append({
                "label": label,
                "raw_bytes": raw_bytes,
                "raw_tokens": raw_tokens,
                "zpf_tokens": zpf_tokens,
                "zpf_bytes": zpf_bytes,
                "blocks": block_count,
                "naive_chunks": len(naive_chunks),
                "naive_tokens": naive_tokens,
                "noise_lines": len(noise_lines),
                "noise_chars": noise_chars,
                "token_reduction": token_reduction,
                "vs_naive": vs_naive,
                "time_ms": convert_time * 1000,
            })

            print(f"{'='*75}")
            print(f"  {label}")
            print(f"{'='*75}")
            print(f"  Original:       {raw_bytes:>8,} bytes | {raw_tokens:>6,} tokens")
            print(f"  .zpf output:    {zpf_bytes:>8,} bytes | {zpf_tokens:>6,} tokens | {block_count} blocks")
            print(f"  Naive 512-chunk: {len(naive_chunks)} chunks | {naive_tokens:>6,} tokens")
            print(f"  Noise stripped:  {len(noise_lines)} lines ({noise_chars:,} chars)")
            print(f"  Token savings:   {token_reduction:.1f}% vs raw")
            print(f"  vs naive chunk:  {vs_naive:.1f}% fewer tokens")
            print(f"  Convert time:    {convert_time*1000:.0f}ms")
            print()

            # Show what noise was stripped
            if noise_lines:
                print(f"  Noise removed (sample):")
                for nl in noise_lines[:5]:
                    print(f"    ✗ {nl.strip()[:70]}")
                if len(noise_lines) > 5:
                    print(f"    ... and {len(noise_lines) - 5} more lines")
                print()

            # Show block types
            blocks = reader.read_all_blocks()
            from zse.core.zrag.zpf_spec import BlockType
            type_counts = {}
            for b in blocks:
                name = BlockType(b.block_type).name
                type_counts[name] = type_counts.get(name, 0) + 1
            print(f"  Block types: {type_counts}")
            print()

        # Summary
        print("=" * 75)
        print("SUMMARY")
        print("=" * 75)
        print(f"{'Document':<42} {'Raw':>7} {'ZPF':>7} {'Naive':>7} {'Blocks':>7} {'Save%':>7}")
        print("-" * 75)
        for r in results:
            print(
                f"{r['label']:<42} "
                f"{r['raw_tokens']:>7,} "
                f"{r['zpf_tokens']:>7,} "
                f"{r['naive_tokens']:>7,} "
                f"{r['blocks']:>7} "
                f"{r['token_reduction']:>6.1f}%"
            )
        print("-" * 75)
        avg = sum(r["token_reduction"] for r in results) / len(results)
        print(f"{'AVERAGE':<42} {'':>7} {'':>7} {'':>7} {'':>7} {avg:>6.1f}%")
        print()
        print(f"Embedding: {pipeline.stats['embedding_model']} ({pipeline.stats['embedding_dim']}d)")
        print()

        # Validation checks
        print("VALIDATION:")
        zpf_counts = [r["zpf_tokens"] for r in results]
        if len(set(zpf_counts)) == 1:
            print("  ⚠ WARNING: All documents produced identical .zpf token counts!")
            print("  This likely indicates a bug in the benchmark or chunker.")
        else:
            print("  ✓ Different documents produce different .zpf sizes (no dedup artifact)")

        block_counts = [r["blocks"] for r in results]
        if len(set(block_counts)) == 1:
            print("  ⚠ WARNING: All documents produced identical block counts!")
        else:
            print("  ✓ Different documents produce different block counts")

        # Check larger docs produce more blocks
        if block_counts[-1] > block_counts[0]:
            print("  ✓ Larger document produces more blocks (expected)")
        else:
            print("  ⚠ Larger document did NOT produce more blocks than smaller one")

        print()


if __name__ == "__main__":
    run_benchmark()
