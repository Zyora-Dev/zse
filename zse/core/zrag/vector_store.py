"""
Vector Store — Efficient similarity search over block embeddings.

NumPy-based cosine similarity with optional batch search.
Stores embeddings in a compact .npz file alongside a JSON metadata index.

Enhanced retrieval:
  - Size-normalized scoring (penalise oversized blocks)
  - Block-type boosting (query intent → block type multiplier)
  - BM25 hybrid (0.6 embedding + 0.4 BM25)
"""

import json
import math
import os
import re
import fcntl
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── BM25 helpers ──────────────────────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer for BM25."""
    return re.findall(r"[a-z0-9_./-]+", text.lower())


# ── Query intent → block-type boost ──────────────────────────────────────────

_LIST_PATTERN = re.compile(
    r"\b(what\s+(techniques|types|kinds|methods|ways|are\s+the)|list\s+(all|the)|name\s+(three|all|the|some)|which\s+(techniques|types|methods))\b",
    re.I,
)

_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*[-_][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)*")

_INTENT_PATTERNS = {
    "code": (
        re.compile(
            r"\b(command|code|script|run|execute|syntax|flag|cli|example|snippet|import|function)\b",
            re.I,
        ),
        {1: 1.25},  # CODE blocks get 1.25× boost
    ),
    "procedure": (
        re.compile(
            r"\b(how\s+(?:do|to|can)|steps?|install|configure|deploy|setup|set\s+up|serve|launch|start|build)\b",
            re.I,
        ),
        {5: 1.2, 1: 1.15},  # PROCEDURE 1.2×, CODE 1.15×
    ),
    "definition": (
        re.compile(
            r"\b(what\s+is|define|meaning|purpose|explain|description|concept)\b",
            re.I,
        ),
        {4: 1.2, 6: 1.15},  # DEFINITION 1.2×, TEXT 1.15×
    ),
    "detail": (
        re.compile(
            r"\b(how\s+much|how\s+many|size|amount|number|value|env\w*\s*var|variable|parameter|setting|config)\b",
            re.I,
        ),
        {2: 1.2, 4: 1.15, 3: 1.15},  # TABLE 1.2×, DEFINITION 1.15×, QA 1.15×
    ),
    "table": (
        re.compile(
            r"\b(vram|memory|benchmark|comparison|gpu|hardware|require|specification|spec)\b",
            re.I,
        ),
        {2: 1.25},  # TABLE 1.25×
    ),
}


def _get_block_type_boosts(query: str) -> Dict[int, float]:
    """Return block_type → multiplier based on query intent."""
    boosts: Dict[int, float] = {}
    for _intent, (pattern, type_boosts) in _INTENT_PATTERNS.items():
        if pattern.search(query):
            for bt, mult in type_boosts.items():
                boosts[bt] = max(boosts.get(bt, 1.0), mult)
    return boosts


@dataclass
class SearchResult:
    """A single search hit."""

    doc_id: str
    block_idx: int
    score: float
    content: str
    block_type: int
    summary: str = ""
    metadata: Dict = field(default_factory=dict)


class VectorStore:
    """
    In-memory vector store backed by NumPy.

    Data layout on disk:
        <store_dir>/embeddings.npz   — float32 matrix (N x dim)
        <store_dir>/index.json       — doc/block metadata for each row
    """

    def __init__(self, store_dir: str, dimension: int = 384):
        self.store_dir = Path(store_dir)
        self.dimension = dimension

        self._embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._index: List[Dict] = []  # parallel to rows

        # BM25 state (rebuilt on load/add)
        self._bm25_tf: List[Counter] = []  # per-doc term frequencies
        self._bm25_df: Counter = Counter()  # document frequency per term
        self._bm25_avgdl: float = 0.0

        self._load()

    # ----- Public API -----

    def add(
        self,
        doc_id: str,
        embeddings: np.ndarray,
        blocks: List[Dict],
    ) -> int:
        """
        Add a document's embeddings + block metadata.

        Args:
            doc_id:     Unique document identifier.
            embeddings: (num_blocks, dim) float32 array.
            blocks:     List of dicts with at minimum: content, block_type.

        Returns:
            Number of blocks added.
        """
        if len(embeddings) == 0:
            return 0

        assert embeddings.shape[0] == len(blocks), (
            f"embeddings ({embeddings.shape[0]}) and blocks ({len(blocks)}) must have same length"
        )

        entries = []
        for i, blk in enumerate(blocks):
            entries.append(
                {
                    "doc_id": doc_id,
                    "block_idx": i,
                    "content": blk.get("content", ""),
                    "block_type": blk.get("block_type", 0),
                    "summary": blk.get("summary", ""),
                    "token_count": blk.get("token_count", 0),
                    "metadata": blk.get("metadata", {}),
                }
            )

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack(
                [
                    self._embeddings,
                    embeddings.astype(np.float32),
                ]
            )
        self._index.extend(entries)

        self._rebuild_bm25()
        self._save()
        return len(entries)

    def remove(self, doc_id: str) -> int:
        """Remove all blocks for a document. Returns count removed."""
        keep_mask = np.array(
            [e["doc_id"] != doc_id for e in self._index],
            dtype=bool,
        )
        removed = int((~keep_mask).sum())

        if removed == 0:
            return 0

        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = self._embeddings[keep_mask]
        self._index = [e for e, k in zip(self._index, keep_mask) if k]

        self._rebuild_bm25()
        self._save()
        return removed

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        score_threshold: float = 0.0,
        query_text: str = "",
    ) -> List[SearchResult]:
        """
        Hybrid retrieval: 0.6 × embedding + 0.4 × BM25, with size
        normalization and block-type boosting.

        Args:
            query_vec:       (dim,) normalized query embedding.
            top_k:           Max results to return.
            doc_filter:      Optional doc_id to restrict search.
            score_threshold: Minimum similarity score.
            query_text:      Raw query string (for BM25 + type boost).
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        n = len(self._index)
        qv = query_vec.astype(np.float32).reshape(1, -1)
        qn = np.linalg.norm(qv)
        if qn > 0:
            qv /= qn

        # ── 1. Cosine similarity ──
        emb_scores = (self._embeddings @ qv.T).flatten()

        # ── 2. Size normalization: penalise large blocks ──
        token_counts = np.array(
            [max(e.get("token_count", 1), 1) for e in self._index],
            dtype=np.float32,
        )
        size_penalty = np.power(token_counts, 0.3)
        emb_scores = emb_scores / size_penalty

        # Normalize embedding scores to [0, 1]
        emb_max = emb_scores.max()
        if emb_max > 0:
            emb_scores_norm = emb_scores / emb_max
        else:
            emb_scores_norm = emb_scores

        # ── 3. BM25 scores ──
        bm25_scores = np.zeros(n, dtype=np.float32)
        if query_text and self._bm25_tf:
            bm25_scores = self._bm25_score(query_text)
            bm25_max = bm25_scores.max()
            if bm25_max > 0:
                bm25_scores = bm25_scores / bm25_max

        # ── 4. Hybrid combination ──
        # If query contains identifiers (model names, env vars, CLI flags),
        # shift to BM25-heavy scoring — embeddings can't match version strings
        if query_text and _IDENTIFIER_PATTERN.search(query_text):
            scores = 0.2 * emb_scores_norm + 0.8 * bm25_scores
        else:
            scores = 0.6 * emb_scores_norm + 0.4 * bm25_scores

        # ── 5. Block-type boosting ──
        if query_text:
            boosts = _get_block_type_boosts(query_text)
            if boosts:
                for i, entry in enumerate(self._index):
                    bt = entry.get("block_type", 0)
                    if bt in boosts:
                        scores[i] *= boosts[bt]

        # Apply doc filter
        if doc_filter:
            mask = np.array(
                [e["doc_id"] == doc_filter for e in self._index],
                dtype=bool,
            )
            scores = np.where(mask, scores, -1.0)

        # Apply threshold
        scores = np.where(scores >= score_threshold, scores, -1.0)

        # Top-k
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s < score_threshold:
                continue
            entry = self._index[idx]
            results.append(
                SearchResult(
                    doc_id=entry["doc_id"],
                    block_idx=entry["block_idx"],
                    score=s,
                    content=entry["content"],
                    block_type=entry["block_type"],
                    summary=entry.get("summary", ""),
                    metadata=entry.get("metadata", {}),
                )
            )

        return results

    def list_documents(self) -> List[Dict]:
        """Return summary of each document in the store."""
        docs: Dict[str, Dict] = {}
        for entry in self._index:
            did = entry["doc_id"]
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "block_count": 0,
                    "total_tokens": 0,
                }
            docs[did]["block_count"] += 1
            docs[did]["total_tokens"] += entry.get("token_count", 0)
        return list(docs.values())

    @property
    def total_blocks(self) -> int:
        return len(self._index)

    @property
    def total_documents(self) -> int:
        return len(set(e["doc_id"] for e in self._index))

    # ----- BM25 -----

    def _rebuild_bm25(self):
        """Rebuild BM25 term-frequency index from current _index."""
        self._bm25_tf = []
        self._bm25_df = Counter()
        total_len = 0

        for entry in self._index:
            tokens = _tokenize(entry.get("content", ""))
            tf = Counter(tokens)
            self._bm25_tf.append(tf)
            total_len += len(tokens)
            for term in set(tokens):
                self._bm25_df[term] += 1

        n = len(self._index)
        self._bm25_avgdl = total_len / n if n > 0 else 1.0

    def _bm25_score(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Compute BM25 scores for all blocks given a query string."""
        n = len(self._index)
        query_tokens = _tokenize(query)
        scores = np.zeros(n, dtype=np.float32)

        for term in query_tokens:
            if term not in self._bm25_df:
                continue
            df = self._bm25_df[term]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for i, tf in enumerate(self._bm25_tf):
                if term not in tf:
                    continue
                freq = tf[term]
                dl = sum(tf.values())
                num = freq * (k1 + 1)
                denom = freq + k1 * (1 - b + b * dl / self._bm25_avgdl)
                scores[i] += idf * num / denom

        return scores

    # ----- Persistence -----

    @contextmanager
    def _file_lock(self):
        """File-based lock for concurrent write safety."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.store_dir / ".store.lock"
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _save(self):
        with self._file_lock():
            self.store_dir.mkdir(parents=True, exist_ok=True)
            emb_path = self.store_dir / "embeddings.npz"
            idx_path = self.store_dir / "index.json"

            if self._embeddings is not None and len(self._embeddings) > 0:
                np.savez_compressed(str(emb_path), embeddings=self._embeddings)
            elif emb_path.exists():
                emb_path.unlink()

            with open(idx_path, "w") as f:
                json.dump(self._index, f, separators=(",", ":"))

    def _load(self):
        emb_path = self.store_dir / "embeddings.npz"
        idx_path = self.store_dir / "index.json"

        if idx_path.exists():
            with open(idx_path) as f:
                self._index = json.load(f)
        else:
            self._index = []

        if emb_path.exists():
            data = np.load(str(emb_path))
            self._embeddings = data["embeddings"].astype(np.float32)
        else:
            self._embeddings = np.zeros((0, self.dimension), dtype=np.float32)

        if self._index:
            self._rebuild_bm25()
