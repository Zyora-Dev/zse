"""ZSE Dense Embedder — Neural embeddings via the already-loaded LLM.

Uses the inference model's last-layer hidden state (mean-pooled, L2-normalized)
as a dense semantic embedding. Zero extra dependencies — reuses the LLM the
server already has in VRAM.

Trade-off:
- Pro: no second model, no extra VRAM, no extra deps
- Con: ~prefill-cost per document at ingestion time (one-shot, then cached)
- Quality: comparable to small encoder models (e.g. MiniLM) for retrieval

Stores fp16 vectors (hidden_size * 2 bytes per chunk) — for Qwen2.5-14B this
is 5120 * 2 = 10 KB per chunk.
"""

import struct
import math
from typing import List, Optional, Dict


class DenseEmbedder:
    """Wraps a ModelRunner to produce dense semantic embeddings.

    Thread-safety: NOT safe for concurrent embed() calls — caller must serialize
    (the underlying ModelRunner shares scratch buffers / KV slots).
    """

    # Reserved KV slot IDs for embedding (high range to avoid collision with
    # regular inference sequences). Cycled to avoid block pool fragmentation.
    _EMBED_SLOT_BASE = 1_000_000_000
    _EMBED_SLOT_COUNT = 16

    def __init__(self, model_runner, tokenizer):
        self._runner = model_runner
        self._tokenizer = tokenizer
        self._slot_cursor = 0
        self._dim = model_runner.hidden_size
        # Simple LRU cache of recently embedded query texts
        self._query_cache: Dict[str, bytes] = {}
        self._query_cache_max = 128

    @property
    def dim(self) -> int:
        return self._dim

    def _next_slot(self) -> int:
        s = self._EMBED_SLOT_BASE + self._slot_cursor
        self._slot_cursor = (self._slot_cursor + 1) % self._EMBED_SLOT_COUNT
        return s

    def _tokenize(self, text: str, max_tokens: int = 512) -> List[int]:
        """Tokenize text, truncating to max_tokens."""
        if self._tokenizer is None:
            return []
        try:
            if hasattr(self._tokenizer, "encode"):
                ids = self._tokenizer.encode(text)
            elif callable(self._tokenizer):
                ids = self._tokenizer(text)
            else:
                return []
        except Exception:
            return []
        if not ids:
            return []
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return list(ids)

    def embed(self, text: str, max_tokens: int = 512) -> Optional[bytes]:
        """Embed a single text. Returns fp16 bytes (hidden_size elements) or None."""
        if not text or not text.strip():
            return None
        token_ids = self._tokenize(text, max_tokens=max_tokens)
        if len(token_ids) < 1:
            return None

        slot = self._next_slot()
        try:
            emb = self._runner.embed_pooled(token_ids, seq_id=slot)
        except Exception:
            try:
                self._runner._kv_cache.free_sequence(slot)
            except Exception:
                pass
            return None
        finally:
            try:
                self._runner._kv_cache.free_sequence(slot)
            except Exception:
                pass
        return emb if emb else None

    def embed_query(self, text: str) -> Optional[bytes]:
        """Embed a query with LRU caching (queries are often repeated)."""
        if text in self._query_cache:
            # Move to end (LRU)
            v = self._query_cache.pop(text)
            self._query_cache[text] = v
            return v
        emb = self.embed(text)
        if emb is not None:
            self._query_cache[text] = emb
            if len(self._query_cache) > self._query_cache_max:
                # Pop oldest (insertion order)
                oldest = next(iter(self._query_cache))
                self._query_cache.pop(oldest, None)
        return emb

    def embed_batch(self, texts: List[str], max_tokens: int = 512) -> List[Optional[bytes]]:
        """Embed many texts sequentially. Returns list aligned with input."""
        return [self.embed(t, max_tokens=max_tokens) for t in texts]


# ---------------------------------------------------------------------------
# Cosine similarity on stored dense vectors (fp16-encoded, L2-normalized)
# ---------------------------------------------------------------------------

def dense_cosine(query_emb: bytes, doc_emb: bytes) -> float:
    """Cosine similarity between two L2-normalized fp16 vectors.

    Since both vectors are unit-normalized, cosine = dot product.
    """
    if not query_emb or not doc_emb or len(query_emb) != len(doc_emb):
        return 0.0
    n = len(query_emb) // 2
    if n == 0:
        return 0.0
    q = struct.unpack(f'<{n}e', query_emb)
    d = struct.unpack(f'<{n}e', doc_emb)
    return sum(a * b for a, b in zip(q, d))
