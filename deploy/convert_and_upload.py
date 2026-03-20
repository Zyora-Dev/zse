"""
ZSE Model Conversion & Upload to HuggingFace

Converts HuggingFace models to .zse format on Modal GPU and uploads
to the zse-zllm HuggingFace organization.

Usage:
    # Convert a single model
    modal run deploy/convert_and_upload.py --model "Qwen/Qwen2.5-7B-Instruct"

    # Convert with custom quantization
    modal run deploy/convert_and_upload.py --model "Qwen/Qwen2.5-7B-Instruct" --quant int8

    # List all models to convert
    modal run deploy/convert_and_upload.py --list

    # Convert all registry models (batch)
    modal run deploy/convert_and_upload.py --all

Requirements:
    - Modal account with GPU access
    - HuggingFace token with write access to zse-zllm org
    - Set HF_TOKEN as Modal secret: modal secret create huggingface HF_TOKEN=hf_xxx
"""

import modal
import os
import sys

# Modal app
app = modal.App("zse-convert")

# Path setup
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

# HuggingFace org for uploads
HF_ORG = "zse-zllm"

# GPU image with dependencies
convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch>=2.1.0",
        "triton>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.20.0",
        "pynvml",
        "rich",
        "typer",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
)

# Add ZSE source code
convert_image_with_code = convert_image.add_local_dir(
    ZSE_ROOT, remote_path="/root/zse"
)

# Persistent volume for converted models
model_volume = modal.Volume.from_name("zse-converted-models", create_if_missing=True)

# Models to convert (ordered by size - smallest first)
MODELS_TO_CONVERT = [
    # Tiny / Small
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "int4"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "int4"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "int4"),
    ("Qwen/Qwen2.5-3B-Instruct", "int4"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "int4"),
    ("google/gemma-2-2b-it", "int4"),
    ("meta-llama/Llama-3.2-3B-Instruct", "int4"),
    ("microsoft/Phi-3-mini-4k-instruct", "int4"),

    # Medium (7B class)
    ("Qwen/Qwen2.5-7B-Instruct", "int4"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "int4"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "int4"),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", "int4"),
    ("meta-llama/Llama-3.1-8B-Instruct", "int4"),
    ("google/gemma-2-9b-it", "int4"),

    # Large
    ("Qwen/Qwen2.5-14B-Instruct", "int4"),

    # XLarge (need A100-80GB)
    ("Qwen/Qwen2.5-32B-Instruct", "int4"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "int4"),

    # XXL (need A100-80GB, may need multiple)
    ("meta-llama/Llama-3.1-70B-Instruct", "int4"),
    ("Qwen/Qwen2.5-72B-Instruct", "int4"),
]


def get_gpu_for_model(model_id: str) -> str:
    """Select appropriate GPU based on model size."""
    model_lower = model_id.lower()

    # 70B+ models need A100-80GB
    if any(s in model_lower for s in ["70b", "72b", "mixtral"]):
        return "a100-80gb"

    # 14B-32B models need A100-40GB
    if any(s in model_lower for s in ["14b", "32b"]):
        return "a100"

    # Everything else can use A10G (cheaper)
    return "a10g"


def get_repo_name(model_id: str, quant: str) -> str:
    """Generate HF repo name for the converted model."""
    name = model_id.split("/")[-1]
    return f"{HF_ORG}/{name}-zse-{quant}"


@app.function(
    image=convert_image_with_code,
    gpu="a10g",
    timeout=1800,  # 30 min max
    volumes={"/converted": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def convert_small(model_id: str, quant: str = "int4") -> dict:
    """Convert a small model (< 10B params) on A10G GPU."""
    return _convert_and_upload(model_id, quant)


@app.function(
    image=convert_image_with_code,
    gpu="a100",
    timeout=3600,  # 60 min max
    volumes={"/converted": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def convert_medium(model_id: str, quant: str = "int4") -> dict:
    """Convert a medium model (10-34B params) on A100-40GB."""
    return _convert_and_upload(model_id, quant)


@app.function(
    image=convert_image_with_code,
    gpu="A100-80GB",
    timeout=7200,  # 2 hr max
    volumes={"/converted": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def convert_large(model_id: str, quant: str = "int4") -> dict:
    """Convert a large model (34B+ params) on A100-80GB."""
    return _convert_and_upload(model_id, quant)


def _convert_and_upload(model_id: str, quant: str = "int4") -> dict:
    """Core conversion and upload logic (runs on GPU)."""
    import subprocess
    import time
    import torch

    # Install ZSE from local source
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd="/root/zse",
    )

    # Ensure ZSE is importable
    if "/root/zse" not in sys.path:
        sys.path.insert(0, "/root/zse")

    from zse.format.writer import ZSEWriter, ConversionConfig

    print(f"\n{'='*60}")
    print(f"Converting: {model_id}")
    print(f"Quantization: {quant}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    # Output paths
    model_name = model_id.split("/")[-1]
    filename = f"{model_name}-zse-{quant}.zse"
    output_path = f"/converted/{filename}"

    start_time = time.time()

    try:
        # Configure conversion
        quant_map = {"int4": "int4", "int8": "int8", "fp16": "none"}
        config = ConversionConfig(
            quantization=quant_map.get(quant, "int4"),
            compute_dtype=torch.float16,
            include_tokenizer=True,
        )

        # Convert
        writer = ZSEWriter(output_path, config)
        result_path = writer.convert_from_hf(model_id, trust_remote_code=True)

        elapsed = time.time() - start_time
        file_size_gb = os.path.getsize(result_path) / (1024**3)

        print(f"\n✅ Conversion complete!")
        print(f"   File: {result_path}")
        print(f"   Size: {file_size_gb:.2f} GB")
        print(f"   Time: {elapsed:.1f}s")

        # Commit to volume
        model_volume.commit()

        # Upload to HuggingFace
        repo_name = get_repo_name(model_id, quant)
        upload_success = _upload_to_hf(result_path, filename, repo_name, model_id, quant, file_size_gb)

        return {
            "status": "success",
            "model_id": model_id,
            "quant": quant,
            "file": str(result_path),
            "size_gb": round(file_size_gb, 2),
            "time_seconds": round(elapsed, 1),
            "hf_repo": repo_name,
            "uploaded": upload_success,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            "status": "error",
            "model_id": model_id,
            "quant": quant,
            "error": str(e),
            "time_seconds": round(elapsed, 1),
        }


def _upload_to_hf(
    file_path: str,
    filename: str,
    repo_id: str,
    source_model: str,
    quant: str,
    size_gb: float,
) -> bool:
    """Upload .zse file to HuggingFace."""
    try:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not set, skipping upload")
            return False

        api = HfApi(token=token)

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id,
                repo_type="model",
                exist_ok=True,
                token=token,
            )
        except Exception as e:
            print(f"   Repo creation note: {e}")

        # Create a README for the model
        readme_content = f"""---
license: apache-2.0
tags:
  - zse
  - int4
  - quantized
  - inference
base_model: {source_model}
---

# {repo_id.split('/')[-1]}

Pre-converted [ZSE](https://github.com/zse-zllm/zse) model for ultra-fast inference.

## Source Model
- **Original:** [{source_model}](https://huggingface.co/{source_model})
- **Quantization:** {quant.upper()}
- **File Size:** {size_gb:.2f} GB
- **Format:** ZSE binary (.zse)

## Usage

```bash
pip install zllm-zse

# Download and serve
zse pull {source_model.split('/')[-1].lower()}
zse serve {source_model.split('/')[-1].lower()}

# Or direct
zse serve {filename}
```

## Benefits
- **5x faster cold start** compared to HuggingFace loading
- **10-14% less VRAM** with ZSE custom INT4 kernels
- **Single file** — tokenizer and config embedded
- **No internet required** after download

## Benchmarks

See [ZSE Documentation](https://zllm.in) for full benchmarks.

---

*Converted with ZSE v1.4.0*
"""

        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )

        # Upload .zse file
        print(f"\n📤 Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
        )

        print(f"✅ Uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.local_entrypoint()
def main(
    model: str = "",
    quant: str = "int4",
    list_models: bool = False,
    all: bool = False,
):
    """
    Convert HuggingFace models to .zse format and upload to zse-zllm org.

    Args:
        model: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
        quant: Quantization type (int4, int8, fp16)
        list_models: List all models available for conversion
        all_models: Convert all registry models
    """
    if list_models:
        print("\n📋 Models available for conversion:\n")
        print(f"{'#':<4} {'Model':<45} {'Quant':<6} {'GPU':<12} {'HF Repo'}")
        print("-" * 100)
        for i, (m, q) in enumerate(MODELS_TO_CONVERT, 1):
            gpu = get_gpu_for_model(m)
            repo = get_repo_name(m, q)
            print(f"{i:<4} {m:<45} {q:<6} {gpu:<12} {repo}")
        print(f"\nTotal: {len(MODELS_TO_CONVERT)} models")
        return

    if not model and not all:
        print("Usage:")
        print("  modal run deploy/convert_and_upload.py --model 'Qwen/Qwen2.5-7B-Instruct'")
        print("  modal run deploy/convert_and_upload.py --list-models")
        print("  modal run deploy/convert_and_upload.py --all")
        return

    if all:
        print(f"\n🚀 Converting ALL {len(MODELS_TO_CONVERT)} models...\n")
        results = []

        for model_id, model_quant in MODELS_TO_CONVERT:
            print(f"\n{'─'*60}")
            print(f"📦 Converting: {model_id} ({model_quant})")
            print(f"{'─'*60}")

            result = _dispatch_conversion(model_id, model_quant)
            results.append(result)

            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} {model_id}: {result.get('size_gb', '?')} GB in {result.get('time_seconds', '?')}s")

        # Summary
        print(f"\n{'='*60}")
        print("📊 Conversion Summary")
        print(f"{'='*60}")
        success = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        print(f"✅ Success: {len(success)}/{len(results)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for r in failed:
                print(f"   - {r['model_id']}: {r.get('error', 'unknown')}")

    else:
        # Single model conversion
        print(f"\n📦 Converting: {model} ({quant})")
        result = _dispatch_conversion(model, quant)

        if result["status"] == "success":
            print(f"\n✅ Done! {result['size_gb']} GB in {result['time_seconds']}s")
            if result.get("uploaded"):
                print(f"📤 Available at: https://huggingface.co/{result['hf_repo']}")
        else:
            print(f"\n❌ Failed: {result.get('error', 'unknown')}")


def _dispatch_conversion(model_id: str, quant: str) -> dict:
    """Dispatch to appropriate GPU function based on model size."""
    gpu = get_gpu_for_model(model_id)

    if gpu == "a100-80gb":
        return convert_large.remote(model_id, quant)
    elif gpu == "a100":
        return convert_medium.remote(model_id, quant)
    else:
        return convert_small.remote(model_id, quant)
