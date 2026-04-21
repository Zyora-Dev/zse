"""ZSE RAG Document Store — SQLite-backed storage for RAG documents and chunks.

Stores documents, chunks, and embeddings. Provides hybrid search:
- TF-IDF cosine similarity (sparse vectors)
- BM25 scoring (probabilistic term weighting)
- Inverted index for fast token-level lookup (avoids brute-force scan)
"""

import json
import struct
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set

from zse_engine.rag.embeddings import (
    TokenEmbedder, _bytes_to_sparse, _sparse_cosine,
)


@dataclass
class StoredDocument:
    """A document in the store."""
    id: int
    name: str
    doc_type: str
    chunk_count: int
    original_tokens: int
    compressed_tokens: int
    created_at: float

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compressed_tokens / self.original_tokens) * 100


@dataclass
class StoredChunk:
    """A chunk in the store."""
    id: int
    doc_id: int
    chunk_index: int
    content: str
    compressed: str
    metadata: Dict[str, Any]
    embedding: bytes
    token_count: int


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk: StoredChunk
    score: float
    doc_name: str = ""
    cosine_score: float = 0.0
    bm25_score: float = 0.0


class InvertedIndex:
    """In-memory inverted index mapping token_id → set of chunk_ids.

    Enables O(|query_tokens|) candidate lookup instead of scanning all chunks.
    Only chunks sharing at least one token with the query are evaluated.
    """

    def __init__(self):
        self._index: Dict[int, Set[int]] = {}  # token_id → {chunk_id, ...}

    def add(self, chunk_id: int, token_ids: Set[int]):
        """Add a chunk's tokens to the index."""
        for t in token_ids:
            if t not in self._index:
                self._index[t] = set()
            self._index[t].add(chunk_id)

    def remove(self, chunk_id: int, token_ids: Set[int]):
        """Remove a chunk from the index."""
        for t in token_ids:
            if t in self._index:
                self._index[t].discard(chunk_id)
                if not self._index[t]:
                    del self._index[t]

    def remove_chunk(self, chunk_id: int):
        """Remove a chunk from all postings (slower, no token set needed)."""
        to_delete = []
        for t, chunks in self._index.items():
            chunks.discard(chunk_id)
            if not chunks:
                to_delete.append(t)
        for t in to_delete:
            del self._index[t]

    def candidates(self, query_tokens: Set[int]) -> Set[int]:
        """Get candidate chunk IDs that share tokens with the query."""
        result = set()
        for t in query_tokens:
            if t in self._index:
                result.update(self._index[t])
        return result

    def clear(self):
        self._index.clear()

    @property
    def token_count(self) -> int:
        return len(self._index)


class RAGStore:
    """SQLite-backed RAG document store with hybrid search.

    Search combines:
    1. Inverted index for fast candidate retrieval
    2. TF-IDF cosine similarity
    3. BM25 scoring
    Final score = (1 - bm25_weight) * cosine + bm25_weight * normalized_bm25

    Args:
        db: ServerDatabase instance (reuses existing DB connection)
    """

    def __init__(self, db):
        self._db = db
        self._conn = db._conn
        self._create_tables()
        # In-memory inverted index (rebuilt on startup)
        self._inverted = InvertedIndex()

    def _create_tables(self):
        """Create RAG tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                doc_type TEXT NOT NULL DEFAULT 'text',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                original_tokens INTEGER NOT NULL DEFAULT 0,
                compressed_tokens INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                compressed TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                token_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc
                ON rag_chunks(doc_id, chunk_index);
        """)

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def add_document(
        self,
        name: str,
        doc_type: str,
        chunks: list,  # List[Chunk] from parser
        embeddings: List[bytes],
        token_sets: Optional[List[Set[int]]] = None,
    ) -> int:
        """Add a document with its chunks and embeddings.

        Args:
            name: Document filename/name
            doc_type: Type (text, json, csv)
            chunks: List of Chunk objects from parser
            embeddings: List of embedding bytes (one per chunk)
            token_sets: Optional token sets per chunk (for inverted index)

        Returns:
            Document ID
        """
        total_orig = sum(c.original_tokens for c in chunks)
        total_comp = sum(c.compressed_tokens for c in chunks)

        cursor = self._conn.execute(
            """INSERT INTO rag_documents
               (name, doc_type, chunk_count, original_tokens, compressed_tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, doc_type, len(chunks), total_orig, total_comp, time.time()),
        )
        doc_id = cursor.lastrowid

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
            cursor2 = self._conn.execute(
                """INSERT INTO rag_chunks
                   (doc_id, chunk_index, content, compressed, metadata, embedding, token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, i, chunk.text, chunk.compressed, meta_json, emb,
                 chunk.compressed_tokens),
            )
            chunk_id = cursor2.lastrowid

            # Add to inverted index
            if token_sets and i < len(token_sets):
                self._inverted.add(chunk_id, token_sets[i])

        return doc_id

    def remove_document(self, doc_id: int) -> bool:
        """Remove a document and all its chunks."""
        # Get chunk IDs for inverted index cleanup
        chunk_ids = self._conn.execute(
            "SELECT id FROM rag_chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        for (cid,) in chunk_ids:
            self._inverted.remove_chunk(cid)

        self._conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
        cursor = self._conn.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))
        return cursor.rowcount > 0

    def list_documents(self) -> List[StoredDocument]:
        """List all documents."""
        rows = self._conn.execute(
            "SELECT * FROM rag_documents ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def get_document(self, doc_id: int) -> Optional[StoredDocument]:
        """Get a document by ID."""
        row = self._conn.execute(
            "SELECT * FROM rag_documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def get_chunks(self, doc_id: int) -> List[StoredChunk]:
        """Get all chunks for a document."""
        rows = self._conn.execute(
            "SELECT * FROM rag_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: Set[int]) -> Dict[int, Tuple[StoredChunk, str]]:
        """Get chunks by IDs with their document names. Returns {chunk_id: (chunk, doc_name)}."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"""SELECT c.*, d.name as doc_name
               FROM rag_chunks c
               JOIN rag_documents d ON c.doc_id = d.id
               WHERE c.id IN ({placeholders})""",
            list(chunk_ids),
        ).fetchall()
        result = {}
        for row in rows:
            chunk = self._row_to_chunk(row[:8])
            doc_name = row[8] if len(row) > 8 else ""
            result[chunk.id] = (chunk, doc_name)
        return result

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: bytes,
        top_k: int = 5,
        candidate_ids: Optional[Set[int]] = None,
        bm25_scores: Optional[Dict[int, float]] = None,
        bm25_weight: float = 0.5,
    ) -> List[SearchResult]:
        """Hybrid search combining cosine similarity and BM25.

        If candidate_ids is provided (from inverted index), only those chunks
        are evaluated — avoiding brute-force scan. If not, falls back to full scan.

        Args:
            query_embedding: Query TF-IDF embedding as bytes
            top_k: Number of results to return
            candidate_ids: Pre-filtered chunk IDs from inverted index
            bm25_scores: Pre-computed BM25 scores {chunk_db_id: score}
            bm25_weight: Weight for BM25 in hybrid score (0-1)
        """
        query_sparse = _bytes_to_sparse(query_embedding)
        if not query_sparse:
            return []

        # Decide which chunks to scan
        if candidate_ids is not None:
            # Fast path: only load candidate chunks
            chunk_data = self.get_chunks_by_ids(candidate_ids)
            scored = []
            for chunk_id, (chunk, doc_name) in chunk_data.items():
                chunk_sparse = _bytes_to_sparse(chunk.embedding)
                cosine = _sparse_cosine(query_sparse, chunk_sparse)

                # Hybrid score
                bm25 = 0.0
                if bm25_scores and chunk_id in bm25_scores:
                    bm25 = bm25_scores[chunk_id]

                score = self._hybrid_score(cosine, bm25, bm25_weight, bm25_scores)
                if score > 0.01:
                    scored.append(SearchResult(
                        chunk=chunk, score=score, doc_name=doc_name,
                        cosine_score=cosine, bm25_score=bm25,
                    ))
        else:
            # Fallback: brute-force scan all chunks
            rows = self._conn.execute(
                """SELECT c.*, d.name as doc_name
                   FROM rag_chunks c
                   JOIN rag_documents d ON c.doc_id = d.id
                   ORDER BY c.doc_id, c.chunk_index"""
            ).fetchall()

            scored = []
            for row in rows:
                chunk = self._row_to_chunk(row[:8])
                doc_name = row[8] if len(row) > 8 else ""
                chunk_sparse = _bytes_to_sparse(chunk.embedding)
                cosine = _sparse_cosine(query_sparse, chunk_sparse)

                bm25 = 0.0
                if bm25_scores and chunk.id in bm25_scores:
                    bm25 = bm25_scores[chunk.id]

                score = self._hybrid_score(cosine, bm25, bm25_weight, bm25_scores)
                if score > 0.01:
                    scored.append(SearchResult(
                        chunk=chunk, score=score, doc_name=doc_name,
                        cosine_score=cosine, bm25_score=bm25,
                    ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _hybrid_score(
        self, cosine: float, bm25: float, bm25_weight: float,
        all_bm25: Optional[Dict[int, float]],
    ) -> float:
        """Compute hybrid score. BM25 is normalized to [0,1] range."""
        if not all_bm25 or bm25_weight <= 0:
            return cosine

        # Normalize BM25 by max score in result set
        max_bm25 = max(all_bm25.values()) if all_bm25 else 1.0
        norm_bm25 = bm25 / max_bm25 if max_bm25 > 0 else 0.0

        return (1 - bm25_weight) * cosine + bm25_weight * norm_bm25

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def rebuild_inverted_index(self, embedder: 'TokenEmbedder'):
        """Rebuild the inverted index and BM25 index from all stored chunks."""
        self._inverted.clear()
        rows = self._conn.execute(
            "SELECT id, content FROM rag_chunks ORDER BY id"
        ).fetchall()
        for chunk_id, content in rows:
            tokens = embedder._tokenize(content)
            token_set = set(tokens)
            self._inverted.add(chunk_id, token_set)
            embedder.add_to_bm25(content, chunk_id)

    def reembed_all(self, embedder: 'TokenEmbedder'):
        """Re-embed all chunks with current IDF weights. Fixes stale embeddings."""
        rows = self._conn.execute(
            "SELECT id, content FROM rag_chunks ORDER BY id"
        ).fetchall()
        for chunk_id, content in rows:
            new_emb = embedder.embed(content)
            self._conn.execute(
                "UPDATE rag_chunks SET embedding = ? WHERE id = ?",
                (new_emb, chunk_id),
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Get store statistics."""
        doc_count = self._conn.execute(
            "SELECT COUNT(*) FROM rag_documents"
        ).fetchone()[0]

        chunk_count = self._conn.execute(
            "SELECT COUNT(*) FROM rag_chunks"
        ).fetchone()[0]

        totals = self._conn.execute(
            "SELECT COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) FROM rag_documents"
        ).fetchone()

        orig = totals[0]
        comp = totals[1]
        savings = (1.0 - comp / orig) * 100 if orig > 0 else 0.0

        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "total_original_tokens": orig,
            "total_compressed_tokens": comp,
            "token_savings_pct": round(savings, 1),
            "inverted_index_tokens": self._inverted.token_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_document(self, row) -> StoredDocument:
        return StoredDocument(
            id=row[0], name=row[1], doc_type=row[2],
            chunk_count=row[3], original_tokens=row[4],
            compressed_tokens=row[5], created_at=row[6],
        )

    def _row_to_chunk(self, row) -> StoredChunk:
        meta = {}
        try:
            meta = json.loads(row[5]) if row[5] else {}
        except Exception:
            pass
        return StoredChunk(
            id=row[0], doc_id=row[1], chunk_index=row[2],
            content=row[3], compressed=row[4], metadata=meta,
            embedding=row[6] if row[6] else b'', token_count=row[7],
        )
