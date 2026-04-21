"""ZSE RAG Embeddings — Hybrid BM25 + TF-IDF document embeddings.

Dual scoring strategies combined for robust retrieval:
1. TF-IDF cosine: Sparse token frequency vectors with IDF weighting
2. BM25: Probabilistic term weighting (better for short queries matching long docs)

Both are zero-cost (no neural model needed) and use the loaded model's tokenizer.
The hybrid combination handles both exact-token and term-importance matching.
"""

import math
import struct
from typing import List, Optional, Dict, Tuple, Set


# ---------------------------------------------------------------------------
# BM25 Scorer
# ---------------------------------------------------------------------------

class BM25Scorer:
    """Okapi BM25 scoring for document retrieval.

    BM25 advantages over pure TF-IDF:
    - Sublinear TF saturation (repeated terms have diminishing returns)
    - Document length normalization (short docs aren't penalized)
    - Better for short queries against long documents

    Parameters follow Robertson et al. defaults: k1=1.5, b=0.75
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_count = 0
        self._doc_freq: Dict[int, int] = {}  # token_id → num docs containing it
        self._doc_lengths: Dict[int, int] = {}  # doc_index → token count
        self._avg_dl: float = 0.0
        self._doc_token_freqs: Dict[int, Dict[int, int]] = {}  # doc_index → {token: freq}
        self._next_doc_id = 0

    def add_document(self, tokens: List[int]) -> int:
        """Add a document to the BM25 index. Returns doc_id."""
        doc_id = self._next_doc_id
        self._next_doc_id += 1

        # Count term frequencies
        tf: Dict[int, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        self._doc_token_freqs[doc_id] = tf
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_count += 1

        # Update document frequency
        for t in tf:
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        # Update average document length
        total = sum(self._doc_lengths.values())
        self._avg_dl = total / self._doc_count if self._doc_count > 0 else 0.0

        return doc_id

    def remove_document(self, doc_id: int):
        """Remove a document from the BM25 index."""
        if doc_id not in self._doc_token_freqs:
            return
        tf = self._doc_token_freqs.pop(doc_id)
        self._doc_lengths.pop(doc_id, None)
        self._doc_count -= 1
        for t in tf:
            self._doc_freq[t] = max(0, self._doc_freq.get(t, 1) - 1)
        total = sum(self._doc_lengths.values())
        self._avg_dl = total / self._doc_count if self._doc_count > 0 else 0.0

    def score(self, query_tokens: List[int], doc_id: int) -> float:
        """Compute BM25 score for a query against a document."""
        if doc_id not in self._doc_token_freqs:
            return 0.0

        tf = self._doc_token_freqs[doc_id]
        dl = self._doc_lengths.get(doc_id, 0)
        score = 0.0

        for qt in set(query_tokens):
            if qt not in tf:
                continue
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self._doc_freq.get(qt, 0)
            idf = math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))
            # TF with saturation and length normalization
            term_freq = tf[qt]
            tf_norm = (term_freq * (self.k1 + 1)) / (
                term_freq + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
            )
            score += idf * tf_norm

        return score

    def score_all(self, query_tokens: List[int]) -> Dict[int, float]:
        """Score all documents against a query. Returns {doc_id: score}."""
        scores = {}
        query_set = set(query_tokens)
        for doc_id in self._doc_token_freqs:
            s = self.score(query_tokens, doc_id)
            if s > 0:
                scores[doc_id] = s
        return scores


# ---------------------------------------------------------------------------
# Token Embedder (TF-IDF + BM25 hybrid)
# ---------------------------------------------------------------------------

class TokenEmbedder:
    """Hybrid document embedder: TF-IDF cosine + BM25 scoring.

    Combines two complementary approaches:
    - TF-IDF cosine: Good for matching similar documents (symmetric)
    - BM25: Good for matching queries to documents (asymmetric, handles length)

    The final search score is a weighted combination of both.

    Args:
        tokenizer: ZSE tokenizer with encode/decode methods
        vocab_size: Vocabulary size (for vector dimensionality)
        bm25_weight: Weight for BM25 score in hybrid (0-1, default 0.5)
    """

    def __init__(self, tokenizer=None, vocab_size: int = 32000, bm25_weight: float = 0.5):
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._bm25_weight = bm25_weight
        # IDF weights: log(N / df) — updated as documents are added
        self._doc_count = 0
        self._doc_freq: Dict[int, int] = {}  # token_id → num docs containing it
        # BM25 scorer
        self._bm25 = BM25Scorer()
        # Map chunk_db_id → bm25_doc_id for cross-referencing
        self._chunk_to_bm25: Dict[int, int] = {}
        self._bm25_to_chunk: Dict[int, int] = {}

    def embed(self, text: str) -> bytes:
        """Compute TF-IDF embedding vector for a text string.

        Returns sparse vector as bytes for SQLite BLOB storage.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return b''

        # Token frequency in this document
        tf: Dict[int, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # Normalize TF (divide by max freq to prevent length bias)
        max_freq = max(tf.values()) if tf else 1
        for t in tf:
            tf[t] = tf[t] / max_freq

        # Apply IDF if we have corpus stats
        weighted = {}
        for t, freq in tf.items():
            idf = 1.0
            if self._doc_count > 0 and t in self._doc_freq:
                idf = math.log(1 + self._doc_count / (1 + self._doc_freq[t]))
            weighted[t] = freq * max(idf, 0.1)

        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in weighted.values()))
        if norm > 0:
            for t in weighted:
                weighted[t] /= norm

        return _sparse_to_bytes(weighted)

    def add_to_bm25(self, text: str, chunk_db_id: int) -> int:
        """Add a chunk to the BM25 index. Returns BM25 doc_id."""
        tokens = self._tokenize(text)
        bm25_id = self._bm25.add_document(tokens)
        self._chunk_to_bm25[chunk_db_id] = bm25_id
        self._bm25_to_chunk[bm25_id] = chunk_db_id
        return bm25_id

    def remove_from_bm25(self, chunk_db_id: int):
        """Remove a chunk from the BM25 index."""
        bm25_id = self._chunk_to_bm25.pop(chunk_db_id, None)
        if bm25_id is not None:
            self._bm25.remove_document(bm25_id)
            self._bm25_to_chunk.pop(bm25_id, None)

    def bm25_search(self, query: str) -> Dict[int, float]:
        """BM25 search. Returns {chunk_db_id: score}."""
        tokens = self._tokenize(query)
        raw_scores = self._bm25.score_all(tokens)
        # Map back to chunk DB IDs
        return {
            self._bm25_to_chunk[bm25_id]: score
            for bm25_id, score in raw_scores.items()
            if bm25_id in self._bm25_to_chunk
        }

    def update_idf(self, token_sets: List[set]):
        """Update IDF weights from a batch of document token sets."""
        self._doc_count += len(token_sets)
        for token_set in token_sets:
            for t in token_set:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

    def similarity(self, emb_a: bytes, emb_b: bytes) -> float:
        """Cosine similarity between two embeddings."""
        a = _bytes_to_sparse(emb_a)
        b = _bytes_to_sparse(emb_b)
        return _sparse_cosine(a, b)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        if self._tokenizer and hasattr(self._tokenizer, 'encode'):
            try:
                return self._tokenizer.encode(text, add_bos=False)
            except Exception:
                pass
        # Fallback: simple word-level hashing
        return _simple_hash_tokenize(text, self._vocab_size)


# ---------------------------------------------------------------------------
# Sparse vector storage (compact BLOB format)
# ---------------------------------------------------------------------------

def _sparse_to_bytes(sparse: Dict[int, float]) -> bytes:
    """Encode sparse vector as bytes: [count:u32] [token_id:u32 weight:f32]..."""
    buf = struct.pack('<I', len(sparse))
    for token_id, weight in sparse.items():
        buf += struct.pack('<If', token_id, weight)
    return buf


def _bytes_to_sparse(data: bytes) -> Dict[int, float]:
    """Decode sparse vector from bytes."""
    if not data or len(data) < 4:
        return {}
    count = struct.unpack_from('<I', data, 0)[0]
    sparse = {}
    offset = 4
    for _ in range(count):
        if offset + 8 > len(data):
            break
        token_id, weight = struct.unpack_from('<If', data, offset)
        sparse[token_id] = weight
        offset += 8
    return sparse


def _sparse_cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0

    # Dot product (only over shared keys)
    dot = 0.0
    # Iterate over the smaller dict
    if len(a) > len(b):
        a, b = b, a
    for k, v in a.items():
        if k in b:
            dot += v * b[k]

    # Both vectors are already unit-normalized, so dot = cosine
    return dot


def _simple_hash_tokenize(text: str, vocab_size: int) -> List[int]:
    """Fallback tokenizer: split on whitespace, hash to token IDs."""
    words = text.lower().split()
    tokens = []
    for w in words:
        # Simple hash
        h = 0
        for ch in w:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        tokens.append(h % vocab_size)
    return tokens
