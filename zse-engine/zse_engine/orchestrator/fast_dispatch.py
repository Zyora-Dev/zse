"""ZSE Fast Decode — C extension for the full decode forward pass.

Executes the entire decode step (~450 kernel launches + memcpys) in a
tight C loop, eliminating Python interpreter overhead.

The Python side sets up a Ctx struct once with all kernel handles, weight
pointers, and scratch buffers. Per step, Python only uploads token/position/
block_table/seq_lens to pre-allocated GPU buffers, then calls one C function.

Expected: ~10-20x speedup (from 240ms to ~15-25ms per decode step).
"""

import ctypes
import tempfile
import os


_C_SOURCE = r"""
#include <stdint.h>

typedef int CUresult;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

typedef CUresult (*LaunchFn)(
    void* f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned smem, void* stream,
    void** params, void** extra
);

typedef CUresult (*MemcpyFn)(CUdeviceptr dst, CUdeviceptr src, unsigned long long nbytes);

#define CDIV(a,b) (((a)+(b)-1)/(b))
#define MAX_LAYERS 64

typedef struct {
    CUdeviceptr data;
    CUdeviceptr scales;
    CUdeviceptr zeros;
} WeightPtrs;

typedef struct {
    LaunchFn launch;
    MemcpyFn memcpy;

    /* Model dims */
    int H;              /* hidden_size */
    int I;              /* intermediate_size */
    int nh;             /* num_heads */
    int nkv;            /* num_kv_heads */
    int hd;             /* head_dim */
    int NL;             /* num_layers */
    int V;              /* vocab_size */
    int BS;             /* kv block_size */
    int MBps;           /* max_blocks_per_seq */
    int group_size;     /* INT4 quantization group size */
    float scale;        /* 1/sqrt(head_dim) */
    float rope_theta;
    float rms_eps;

    /* Kernel function handles */
    CUfunction k_embed_f32;       /* embedding_lookup_f32out */
    CUfunction k_rmsnorm_f32in;   /* rmsnorm_f32in: float* in -> half* out */
    CUfunction k_fused_res_norm_f32; /* fused_residual_rmsnorm_f32 */
    CUfunction k_residual_add_f32;   /* residual_add_f32 */
    CUfunction k_matmul_int4;
    CUfunction k_batched_rope;
    CUfunction k_kv_write;
    CUfunction k_paged_attn;
    CUfunction k_silu_mul;
    CUfunction k_bias_add;

    /* GEMV kernel handles (M=1 optimized) */
    CUfunction k_gemv_int4;
    CUfunction k_gemv_fp16;
    CUfunction k_fused_gate_up;

    /* Scratch GPU buffers (fp16) */
    CUdeviceptr hidden;
    CUdeviceptr residual;
    CUdeviceptr qkv;
    CUdeviceptr attn_out;
    CUdeviceptr mlp_gate;
    CUdeviceptr mlp_up;
    CUdeviceptr mlp_out;
    CUdeviceptr logits;
    CUdeviceptr norm_out;

    /* Scratch GPU buffers (fp32 residual stream) */
    CUdeviceptr hidden_f32;
    CUdeviceptr residual_f32;

    /* Global weights */
    CUdeviceptr kv_slab;
    CUdeviceptr w_embed;
    CUdeviceptr w_final_norm;
    WeightPtrs w_lm_head;

    /* Per-layer weights */
    CUdeviceptr w_in_norm[MAX_LAYERS];
    CUdeviceptr w_post_norm[MAX_LAYERS];
    WeightPtrs w_q[MAX_LAYERS];
    WeightPtrs w_k[MAX_LAYERS];
    WeightPtrs w_v[MAX_LAYERS];
    WeightPtrs w_o[MAX_LAYERS];
    WeightPtrs w_gate[MAX_LAYERS];
    WeightPtrs w_up[MAX_LAYERS];
    WeightPtrs w_down[MAX_LAYERS];

    /* Per-layer QKV bias pointers (0 if no bias) */
    CUdeviceptr w_q_bias[MAX_LAYERS];
    CUdeviceptr w_k_bias[MAX_LAYERS];
    CUdeviceptr w_v_bias[MAX_LAYERS];

    /* LM head type: 1=fp16, 0=int4 */
    int lm_head_fp16;
    CUfunction k_fp16_matmul;
} Ctx;

/*
 * fast_decode_m1: Full decode forward pass for M=1 (single sequence).
 * Uses fp32 residual stream to prevent fp16 overflow across layers.
 *
 * Flow per layer:
 *   1. Copy hidden_f32 -> residual_f32 (fp32, H*4 bytes)
 *   2. RMSNorm: hidden_f32 (fp32) -> norm_out (fp16)
 *   3. QKV matmuls: norm_out (fp16) -> qkv/attn_out/mlp_out (fp16)
 *   4. Bias add (if present)
 *   5. RoPE, KV write, Paged attention
 *   6. O proj -> hidden (fp16)
 *   7. Residual add: residual_f32 = residual_f32 + hidden (fp32 += fp16)
 *   8. Fused residual+RMSNorm: hidden_f32 = hidden(fp16) + residual_f32; norm_out = RMSNorm(hidden_f32)
 *      Actually: use fused_residual_rmsnorm_f32 which does both
 *   ... but we need a different flow. Let me match model_runner exactly:
 *
 * Per layer (matching _transformer_block_decode):
 *   1. residual_f32 = hidden_f32  (memcpy, H*4 bytes)
 *   2. norm_out = rmsnorm_f32in(hidden_f32)
 *   3. QKV matmuls (fp16)
 *   4. Bias add (if present)
 *   5. RoPE, KV write, Paged attention
 *   6. O proj -> hidden (fp16)
 *   7. residual_add_f32: hidden_f32 = residual_f32 + hidden (fp32 = fp32 + fp16)
 *   8. Post-attn: residual_f32 = hidden_f32 (memcpy H*4)
 *   9. rmsnorm_f32in: norm_out = RMSNorm(hidden_f32)
 *  10. Gate/Up/SiLU/Down MLP (fp16)
 *  11. residual_add_f32: hidden_f32 = residual_f32 + hidden (fp32 = fp32 + fp16)
 */
int fast_decode_m1(
    Ctx* c,
    CUdeviceptr token_buf,
    CUdeviceptr pos_buf,
    CUdeviceptr bt_buf,
    CUdeviceptr sl_buf,
    int position,
    int total_tokens
) {
    LaunchFn F = c->launch;
    MemcpyFn MC = c->memcpy;
    int H = c->H, I = c->I, nh = c->nh, nkv = c->nkv, hd = c->hd;
    int NL = c->NL, V = c->V, BS = c->BS, MBps = c->MBps, gs = c->group_size;
    int q_dim = nh * hd, k_dim = nkv * hd;
    int one = 1;
    CUresult s;

    /* Embedding lookup -> hidden_f32 (float output) */
    {
        void* a[] = {&c->hidden_f32, &c->w_embed, &token_buf, &H, &one};
        s = F(c->k_embed_f32, CDIV(H,256),1,1, 256,1,1, 0, 0, a, 0);
        if (s) return -1;
    }

    for (int l = 0; l < NL; l++) {
        /* 1. Save residual: residual_f32 = hidden_f32 (fp32 copy, H*4 bytes) */
        s = MC(c->residual_f32, c->hidden_f32, (unsigned long long)H * 4);
        if (s) return -(l * 100 + 10);

        /* 2. Pre-attention RMSNorm: hidden_f32 (fp32) -> norm_out (fp16) */
        {
            void* a[] = {&c->norm_out, &c->hidden_f32, &c->w_in_norm[l], &H, &c->rms_eps};
            s = F(c->k_rmsnorm_f32in, 1,1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 20);
        }

        /* 3. Q projection (INT4 GEMV — M=1 optimized) */
        {
            void* a[] = {&c->qkv, &c->w_q[l].data, &c->w_q[l].scales, &c->w_q[l].zeros,
                        &c->norm_out, &q_dim, &H, &gs};
            s = F(c->k_gemv_int4, CDIV(q_dim,8),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 30);
        }

        /* 4. K projection */
        {
            void* a[] = {&c->attn_out, &c->w_k[l].data, &c->w_k[l].scales, &c->w_k[l].zeros,
                        &c->norm_out, &k_dim, &H, &gs};
            s = F(c->k_gemv_int4, CDIV(k_dim,8),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 40);
        }

        /* 5. V projection */
        {
            void* a[] = {&c->mlp_out, &c->w_v[l].data, &c->w_v[l].scales, &c->w_v[l].zeros,
                        &c->norm_out, &k_dim, &H, &gs};
            s = F(c->k_gemv_int4, CDIV(k_dim,8),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 50);
        }

        /* 5b. QKV bias add (Qwen2 has biases) */
        if (c->w_q_bias[l]) {
            int total_q = one * q_dim;
            void* a[] = {&c->qkv, &c->w_q_bias[l], &one, &q_dim};
            s = F(c->k_bias_add, CDIV(total_q,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 51);
        }
        if (c->w_k_bias[l]) {
            int total_k = one * k_dim;
            void* a[] = {&c->attn_out, &c->w_k_bias[l], &one, &k_dim};
            s = F(c->k_bias_add, CDIV(total_k,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 52);
        }
        if (c->w_v_bias[l]) {
            int total_v = one * k_dim;
            void* a[] = {&c->mlp_out, &c->w_v_bias[l], &one, &k_dim};
            s = F(c->k_bias_add, CDIV(total_v,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 53);
        }

        /* 6. Batched RoPE */
        {
            int half = hd / 2;
            int tr = (nh > nkv ? nh : nkv) * half;
            void* a[] = {&c->qkv, &c->attn_out, &pos_buf, &one, &nh, &nkv, &hd, &c->rope_theta};
            s = F(c->k_batched_rope, CDIV(tr,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 60);
        }

        /* 7. KV cache write */
        {
            int tkv = nkv * hd;
            void* a[] = {&c->kv_slab, &c->attn_out, &c->mlp_out, &bt_buf,
                        &one, &nkv, &hd, &BS, &MBps, &NL, &l, &position};
            s = F(c->k_kv_write, CDIV(tkv,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 70);
        }

        /* 8. Paged attention */
        {
            unsigned int smem = (unsigned int)(total_tokens * 4);
            int bd = hd < 256 ? hd : 256;
            void* a[] = {&c->attn_out, &c->qkv, &c->kv_slab, &bt_buf, &sl_buf,
                        &nh, &nkv, &hd, &BS, &MBps, &NL, &l, &c->scale};
            s = F(c->k_paged_attn, 1,nh,1, bd,1,1, smem, 0, a, 0);
            if (s) return -(l * 100 + 80);
        }

        /* 9. O projection -> hidden (fp16, GEMV) */
        {
            void* a[] = {&c->hidden, &c->w_o[l].data, &c->w_o[l].scales, &c->w_o[l].zeros,
                        &c->attn_out, &H, &q_dim, &gs};
            s = F(c->k_gemv_int4, CDIV(H,8),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 90);
        }

        /* 10. Attention residual: hidden_f32 = residual_f32 + hidden (fp32 += fp16) */
        {
            void* a[] = {&c->hidden_f32, &c->residual_f32, &c->hidden, &H};
            s = F(c->k_residual_add_f32, CDIV(H,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 91);
        }

        /* 11. Save post-attention residual: residual_f32 = hidden_f32 */
        s = MC(c->residual_f32, c->hidden_f32, (unsigned long long)H * 4);
        if (s) return -(l * 100 + 92);

        /* 12. Post-attention RMSNorm: hidden_f32 (fp32) -> norm_out (fp16) */
        {
            void* a[] = {&c->norm_out, &c->hidden_f32, &c->w_post_norm[l], &H, &c->rms_eps};
            s = F(c->k_rmsnorm_f32in, 1,1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 93);
        }

        /* 13-14. Fused Gate+Up projection (single kernel, saves input re-read) */
        {
            void* a[] = {&c->mlp_gate, &c->mlp_up,
                        &c->w_gate[l].data, &c->w_gate[l].scales, &c->w_gate[l].zeros,
                        &c->w_up[l].data, &c->w_up[l].scales, &c->w_up[l].zeros,
                        &c->norm_out, &I, &H, &gs};
            s = F(c->k_fused_gate_up, CDIV(I,4),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 94);
        }

        /* 15. SiLU * Up */
        {
            void* a[] = {&c->mlp_gate, &c->mlp_gate, &c->mlp_up, &I};
            s = F(c->k_silu_mul, CDIV(I,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 96);
        }

        /* 16. Down projection -> hidden (fp16, GEMV) */
        {
            void* a[] = {&c->hidden, &c->w_down[l].data, &c->w_down[l].scales, &c->w_down[l].zeros,
                        &c->mlp_gate, &H, &I, &gs};
            s = F(c->k_gemv_int4, CDIV(H,8),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 97);
        }

        /* 17. MLP residual: hidden_f32 = residual_f32 + hidden (fp32 += fp16) */
        {
            void* a[] = {&c->hidden_f32, &c->residual_f32, &c->hidden, &H};
            s = F(c->k_residual_add_f32, CDIV(H,256),1,1, 256,1,1, 0, 0, a, 0);
            if (s) return -(l * 100 + 98);
        }
    }

    /* Final RMSNorm: hidden_f32 (fp32) -> norm_out (fp16) */
    {
        void* a[] = {&c->norm_out, &c->hidden_f32, &c->w_final_norm, &H, &c->rms_eps};
        s = F(c->k_rmsnorm_f32in, 1,1,1, 256,1,1, 0, 0, a, 0);
        if (s) return -9001;
    }

    /* LM head (GEMV) */
    if (c->lm_head_fp16) {
        void* a[] = {&c->logits, &c->w_lm_head.data, &c->norm_out, &V, &H};
        s = F(c->k_gemv_fp16, CDIV(V,8),1,1, 256,1,1, 0, 0, a, 0);
    } else {
        void* a[] = {&c->logits, &c->w_lm_head.data, &c->w_lm_head.scales, &c->w_lm_head.zeros,
                    &c->norm_out, &V, &H, &gs};
        s = F(c->k_gemv_int4, CDIV(V,8),1,1, 256,1,1, 0, 0, a, 0);
    }
    if (s) return -9002;

    return 0;
}
"""


class _WeightPtrs(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_ulonglong),
        ("scales", ctypes.c_ulonglong),
        ("zeros", ctypes.c_ulonglong),
    ]


class _Ctx(ctypes.Structure):
    _fields_ = [
        ("launch", ctypes.c_void_p),
        ("memcpy", ctypes.c_void_p),
        # Model dims
        ("H", ctypes.c_int), ("I", ctypes.c_int),
        ("nh", ctypes.c_int), ("nkv", ctypes.c_int),
        ("hd", ctypes.c_int), ("NL", ctypes.c_int),
        ("V", ctypes.c_int), ("BS", ctypes.c_int),
        ("MBps", ctypes.c_int), ("group_size", ctypes.c_int),
        ("scale", ctypes.c_float), ("rope_theta", ctypes.c_float),
        ("rms_eps", ctypes.c_float),
        # Kernel handles
        ("k_embed_f32", ctypes.c_void_p),
        ("k_rmsnorm_f32in", ctypes.c_void_p),
        ("k_fused_res_norm_f32", ctypes.c_void_p),
        ("k_residual_add_f32", ctypes.c_void_p),
        ("k_matmul_int4", ctypes.c_void_p),
        ("k_batched_rope", ctypes.c_void_p),
        ("k_kv_write", ctypes.c_void_p),
        ("k_paged_attn", ctypes.c_void_p),
        ("k_silu_mul", ctypes.c_void_p),
        ("k_bias_add", ctypes.c_void_p),
        # GEMV kernel handles
        ("k_gemv_int4", ctypes.c_void_p),
        ("k_gemv_fp16", ctypes.c_void_p),
        ("k_fused_gate_up", ctypes.c_void_p),
        # Scratch (fp16)
        ("hidden", ctypes.c_ulonglong), ("residual", ctypes.c_ulonglong),
        ("qkv", ctypes.c_ulonglong), ("attn_out", ctypes.c_ulonglong),
        ("mlp_gate", ctypes.c_ulonglong), ("mlp_up", ctypes.c_ulonglong),
        ("mlp_out", ctypes.c_ulonglong), ("logits", ctypes.c_ulonglong),
        ("norm_out", ctypes.c_ulonglong),
        # Scratch (fp32)
        ("hidden_f32", ctypes.c_ulonglong), ("residual_f32", ctypes.c_ulonglong),
        # Global weights
        ("kv_slab", ctypes.c_ulonglong), ("w_embed", ctypes.c_ulonglong),
        ("w_final_norm", ctypes.c_ulonglong),
        ("w_lm_head", _WeightPtrs),
        # Per-layer weights
        ("w_in_norm", ctypes.c_ulonglong * 64),
        ("w_post_norm", ctypes.c_ulonglong * 64),
        ("w_q", _WeightPtrs * 64), ("w_k", _WeightPtrs * 64),
        ("w_v", _WeightPtrs * 64), ("w_o", _WeightPtrs * 64),
        ("w_gate", _WeightPtrs * 64), ("w_up", _WeightPtrs * 64),
        ("w_down", _WeightPtrs * 64),
        # Per-layer QKV biases
        ("w_q_bias", ctypes.c_ulonglong * 64),
        ("w_k_bias", ctypes.c_ulonglong * 64),
        ("w_v_bias", ctypes.c_ulonglong * 64),
        # LM head type
        ("lm_head_fp16", ctypes.c_int),
        ("k_fp16_matmul", ctypes.c_void_p),
    ]


class FastDecoder:
    """C-accelerated decode forward pass for M=1."""

    def __init__(self):
        self._lib = None
        self._ctx = _Ctx()
        self._available = False

    def compile(self):
        """Compile the C extension. Call once at startup."""
        tmpdir = tempfile.mkdtemp(prefix="zse_fast_")
        c_path = os.path.join(tmpdir, "fast_decode.c")
        so_path = os.path.join(tmpdir, "fast_decode.so")
        with open(c_path, "w") as f:
            f.write(_C_SOURCE)
        ret = os.system(f"gcc -shared -fPIC -O2 -o {so_path} {c_path} 2>/dev/null")
        if ret != 0:
            ret = os.system(f"cc -shared -fPIC -O2 -o {so_path} {c_path} 2>/dev/null")
        if ret != 0:
            return False
        self._lib = ctypes.CDLL(so_path)
        self._lib.fast_decode_m1.restype = ctypes.c_int
        self._lib.fast_decode_m1.argtypes = [
            ctypes.POINTER(_Ctx),
            ctypes.c_ulonglong,  # token_buf
            ctypes.c_ulonglong,  # pos_buf
            ctypes.c_ulonglong,  # bt_buf
            ctypes.c_ulonglong,  # sl_buf
            ctypes.c_int,        # position
            ctypes.c_int,        # total_tokens
        ]
        self._available = True
        return True

    @property
    def available(self):
        return self._available

    def setup(self, config, weights, scratch, kv_cache, kernels, gpu_mem, layer_weights=None):
        """Initialize Ctx with model state. Call once after kernel compilation."""
        c = self._ctx

        # Resolve cuLaunchKernel and cuMemcpy function pointers via dlsym
        driver = gpu_mem._driver
        try:
            libdl = ctypes.CDLL("libdl.so.2")
            libdl.dlsym.restype = ctypes.c_void_p
            libdl.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            c.launch = libdl.dlsym(driver._handle, b"cuLaunchKernel")
            c.memcpy = libdl.dlsym(driver._handle, b"cuMemcpy")
            if not c.launch or not c.memcpy:
                self._available = False
                return
        except Exception:
            self._available = False
            return

        # Model dimensions
        c.H = config.hidden_size
        c.I = config.intermediate_size
        c.nh = config.num_heads
        c.nkv = config.num_kv_heads
        c.hd = config.head_dim
        c.NL = config.num_layers
        c.V = config.vocab_size
        c.BS = kv_cache.block_size
        c.MBps = 256  # max_blocks_per_seq (matches model_runner)
        c.scale = 1.0 / (config.head_dim ** 0.5)
        c.rope_theta = config.rope_theta
        c.rms_eps = config.rms_norm_eps

        # Kernel function handles — fp32 versions
        c.k_embed_f32 = kernels.get("embedding_lookup_f32out").function
        c.k_rmsnorm_f32in = kernels.get("rmsnorm_f32in").function
        c.k_fused_res_norm_f32 = kernels.get("fused_residual_rmsnorm_f32").function
        c.k_residual_add_f32 = kernels.get("residual_add_f32").function
        c.k_matmul_int4 = kernels.get("tiled_dequant_matmul_int4").function
        c.k_batched_rope = kernels.get("batched_rotary_embedding").function
        c.k_kv_write = kernels.get("kv_cache_write").function
        c.k_paged_attn = kernels.get("paged_attention").function
        c.k_silu_mul = kernels.get("silu_mul").function
        c.k_bias_add = kernels.get("bias_add").function

        # GEMV kernel handles (M=1 optimized)
        c.k_gemv_int4 = kernels.get("dequant_gemv_int4").function
        gemv_fp16 = kernels.get("fp16_gemv")
        if gemv_fp16:
            c.k_gemv_fp16 = gemv_fp16.function
        fused_gu = kernels.get("fused_gate_up_gemv_int4")
        if fused_gu:
            c.k_fused_gate_up = fused_gu.function

        # FP16 matmul (for lm_head if needed)
        fp16_k = kernels.get("tiled_fp16_matmul")
        if fp16_k:
            c.k_fp16_matmul = fp16_k.function

        # Scratch buffers (fp16)
        c.hidden = scratch.hidden.data_ptr
        c.residual = scratch.residual.data_ptr
        c.qkv = scratch.qkv.data_ptr
        c.attn_out = scratch.attn_out.data_ptr
        c.mlp_gate = scratch.mlp_gate.data_ptr
        c.mlp_up = scratch.mlp_up.data_ptr
        c.mlp_out = scratch.mlp_out.data_ptr
        c.logits = scratch.logits.data_ptr
        c.norm_out = scratch.norm_out.data_ptr

        # Scratch buffers (fp32)
        c.hidden_f32 = scratch.hidden_f32.data_ptr
        c.residual_f32 = scratch.residual_f32.data_ptr

        # Global weights
        c.kv_slab = kv_cache._pool._gpu_base_ptr
        c.w_embed = weights.get("embed_tokens.weight").data_ptr

        norm_w = weights.find("model.norm.weight") or weights.find("norm.weight")
        c.w_final_norm = norm_w.data_ptr

        # LM head
        lm_head = weights.get("lm_head.weight")
        c.w_lm_head.data = lm_head.data_ptr
        if lm_head.dtype == "float16":
            c.lm_head_fp16 = 1
        else:
            c.lm_head_fp16 = 0
            c.w_lm_head.scales = lm_head.scales_ptr
            c.w_lm_head.zeros = lm_head.zeros_ptr

        # Determine group_size from first weight
        first_q = weights.find("model.layers.0.self_attn.q_proj.weight") or \
                  weights.find("layers.0.self_attn.q_proj.weight")
        c.group_size = first_q.group_size if first_q else 128

        # Per-layer weights
        for l in range(config.num_layers):
            def _find(name):
                w = weights.find(f"model.layers.{l}.{name}")
                if w is None:
                    w = weights.find(f"layers.{l}.{name}")
                return w

            # Norm weights (just data_ptr)
            in_norm = _find("input_layernorm.weight")
            c.w_in_norm[l] = in_norm.data_ptr if in_norm else 0

            post_norm = _find("post_attention_layernorm.weight")
            c.w_post_norm[l] = post_norm.data_ptr if post_norm else 0

            # Projection weights (data + scales + zeros)
            proj_names = [
                ("self_attn.q_proj.weight", "w_q"),
                ("self_attn.k_proj.weight", "w_k"),
                ("self_attn.v_proj.weight", "w_v"),
                ("self_attn.o_proj.weight", "w_o"),
                ("mlp.gate_proj.weight", "w_gate"),
                ("mlp.up_proj.weight", "w_up"),
                ("mlp.down_proj.weight", "w_down"),
            ]
            for weight_name, attr_name in proj_names:
                w = _find(weight_name)
                arr = getattr(c, attr_name)
                if w:
                    arr[l].data = w.data_ptr
                    arr[l].scales = w.scales_ptr if hasattr(w, 'scales_ptr') and w.scales_ptr else 0
                    arr[l].zeros = w.zeros_ptr if hasattr(w, 'zeros_ptr') and w.zeros_ptr else 0

            # QKV biases (Qwen2)
            if layer_weights and l < len(layer_weights):
                lw = layer_weights[l]
                q_bias = lw.get("self_attn.q_proj.bias")
                c.w_q_bias[l] = q_bias.data_ptr if q_bias else 0
                k_bias = lw.get("self_attn.k_proj.bias")
                c.w_k_bias[l] = k_bias.data_ptr if k_bias else 0
                v_bias = lw.get("self_attn.v_proj.bias")
                c.w_v_bias[l] = v_bias.data_ptr if v_bias else 0
            else:
                c.w_q_bias[l] = 0
                c.w_k_bias[l] = 0
                c.w_v_bias[l] = 0

    def decode_step(self, token_buf_ptr, pos_buf_ptr, bt_buf_ptr, sl_buf_ptr,
                    position, total_tokens):
        """Execute one full decode step (M=1) in C.

        Args:
            token_buf_ptr: GPU ptr to int32 token ID buffer
            pos_buf_ptr: GPU ptr to int32 positions buffer
            bt_buf_ptr: GPU ptr to int32 block table buffer
            sl_buf_ptr: GPU ptr to int32 seq lengths buffer
            position: current position (int)
            total_tokens: total tokens in sequence (for paged attn shared mem)

        Returns:
            0 on success, negative on failure
        """
        return self._lib.fast_decode_m1(
            ctypes.byref(self._ctx),
            ctypes.c_ulonglong(token_buf_ptr),
            ctypes.c_ulonglong(pos_buf_ptr),
            ctypes.c_ulonglong(bt_buf_ptr),
            ctypes.c_ulonglong(sl_buf_ptr),
            ctypes.c_int(position),
            ctypes.c_int(total_tokens),
        )
