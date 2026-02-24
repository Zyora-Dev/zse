# ZSE CUDA Source Code

This directory contains custom CUDA kernels for ZSE:

## Structure

```
csrc/
├── attention/          # zAttention kernels
│   ├── paged_attention.cu
│   ├── flash_attention.cu
│   └── sparse_attention.cu
├── quantize/           # zQuantize kernels
│   ├── int2_gemm.cu
│   ├── int3_gemm.cu
│   ├── int4_gemm.cu
│   └── int8_gemm.cu
└── kv_cache/           # zKV kernels
    └── quantized_kv.cu
```

## Build

CUDA kernels are automatically compiled during `pip install`:

```bash
pip install -e .
```

If CUDA is not available, ZSE falls back to Triton/PyTorch implementations.

## Requirements

- CUDA Toolkit 11.8+ or 12.x
- PyTorch with CUDA support
- Ninja (for faster compilation)
