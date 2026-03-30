"""
Vector Store — Efficient similarity search over block embeddings.

NumPy-based cosine similarity with optional batch search.
Stores embeddings in a compact .npz file alongside a JSON metadata index.
"""

import json
import os
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


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
            f"embeddings ({embeddings.shape[0]}) and blocks ({len(blocks)}) "
            "must have same length"
        )

        entries = []
        for i, blk in enumerate(blocks):
            entries.append({
                "doc_id": doc_id,
                "block_idx": i,
                "content": blk.get("content", ""),
                "block_type": blk.get("block_type", 0),
                "summary": blk.get("summary", ""),
                "token_count": blk.get("token_count", 0),
                "metadata": blk.get("metadata", {}),
            })

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack([
                self._embeddings,
                embeddings.astype(np.float32),
            ])
        self._index.extend(entries)

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

        self._save()
        return removed

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Cosine similarity search.

        Args:
            query_vec:       (dim,) normalized query embedding.
            top_k:           Max results to return.
            doc_filter:      Optional doc_id to restrict search.
            score_threshold: Minimum similarity score.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        qv = query_vec.astype(np.float32).reshape(1, -1)
        # Normalize
        qn = np.linalg.norm(qv)
        if qn > 0:
            qv /= qn

        scores = (self._embeddings @ qv.T).flatten()

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
            results.append(SearchResult(
                doc_id=entry["doc_id"],
                block_idx=entry["block_idx"],
                score=s,
                content=entry["content"],
                block_type=entry["block_type"],
                summary=entry.get("summary", ""),
                metadata=entry.get("metadata", {}),
            ))

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
