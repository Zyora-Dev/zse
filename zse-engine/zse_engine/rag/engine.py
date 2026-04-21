"""ZSE RAG Engine — Orchestrator for document ingestion, search, and context injection.

Ties together: parsers, ZPF compression, hybrid embeddings (TF-IDF + BM25),
inverted index, and the document store.
Provides a clean API for the HTTP layer and chat integration.
"""

import base64
import json
from typing import List, Optional, Dict, Any, Set

from zse_engine.rag.parser import get_parser, Chunk
from zse_engine.rag.embeddings import TokenEmbedder
from zse_engine.rag.store import RAGStore, SearchResult, StoredDocument


class RAGEngine:
    """RAG orchestrator — ingest, search, and augment.

    Features:
    - ZPF compression for token-optimized storage
    - Hybrid search: TF-IDF cosine + BM25
    - Inverted index for fast candidate retrieval (scales to 100K+ chunks)
    - Automatic re-embedding when IDF weights change significantly
    - PDF, JSONL, JSON, CSV, text/markdown support

    Args:
        store: RAGStore instance (SQLite-backed)
        tokenizer: ZSE tokenizer (for token counting and embedding)
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        bm25_weight: Weight for BM25 in hybrid scoring (0-1)
    """

    def __init__(
        self,
        store: RAGStore,
        tokenizer=None,
        chunk_size: int = 512,
        overlap: int = 64,
        bm25_weight: float = 0.5,
    ):
        self._store = store
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._bm25_weight = bm25_weight

        # Initialize embedder
        vocab_size = 32000
        if tokenizer and hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size
        self._embedder = TokenEmbedder(
            tokenizer=tokenizer, vocab_size=vocab_size,
            bm25_weight=bm25_weight,
        )

        # Track IDF drift for re-embedding decisions
        self._docs_since_reembed = 0
        self._reembed_threshold = 10  # Re-embed after every N new documents

        # Rebuild IDF + inverted index + BM25 from existing documents
        self._rebuild_all()

    def _rebuild_all(self):
        """Rebuild IDF weights, inverted index, and BM25 from all stored documents."""
        docs = self._store.list_documents()
        all_token_sets = []
        for doc in docs:
            chunks = self._store.get_chunks(doc.id)
            for chunk in chunks:
                tokens = self._embedder._tokenize(chunk.content)
                all_token_sets.append(set(tokens))
        if all_token_sets:
            self._embedder.update_idf(all_token_sets)

        # Rebuild inverted index and BM25
        self._store.rebuild_inverted_index(self._embedder)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        filename: str,
        content: bytes,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ingest a document: parse, compress, embed, store.

        Args:
            filename: Original filename (used for parser selection)
            content: Raw file content as bytes
            chunk_size: Override default chunk size
            overlap: Override default overlap

        Returns:
            Dict with doc_id, chunk_count, token savings, etc.
        """
        cs = chunk_size or self._chunk_size
        ov = overlap or self._overlap

        # Decode content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        # Parse into chunks
        parser = get_parser(filename, chunk_size=cs, overlap=ov)
        chunks = parser.parse(text, tokenizer=self._tokenizer, metadata={
            "source": filename,
        })

        if not chunks:
            return {"error": "No content could be extracted from the document"}

        # Compute embeddings and collect token sets
        embeddings = []
        token_sets = []
        for chunk in chunks:
            emb = self._embedder.embed(chunk.text)
            embeddings.append(emb)
            tokens = self._embedder._tokenize(chunk.text)
            token_sets.append(set(tokens))

        # Update IDF with new document
        self._embedder.update_idf(token_sets)

        # Determine doc type
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "text"
        doc_type = {"json": "json", "jsonl": "json", "csv": "csv", "pdf": "pdf"}.get(ext, "text")

        # Store (pass token_sets for inverted index)
        doc_id = self._store.add_document(
            name=filename,
            doc_type=doc_type,
            chunks=chunks,
            embeddings=embeddings,
            token_sets=token_sets,
        )

        # Add chunks to BM25 index
        stored_chunks = self._store.get_chunks(doc_id)
        for sc in stored_chunks:
            self._embedder.add_to_bm25(sc.content, sc.id)

        # Check if we need to re-embed existing docs (IDF drift)
        self._docs_since_reembed += 1
        if self._docs_since_reembed >= self._reembed_threshold:
            self._reembed_existing()
            self._docs_since_reembed = 0

        total_orig = sum(c.original_tokens for c in chunks)
        total_comp = sum(c.compressed_tokens for c in chunks)
        savings = (1.0 - total_comp / total_orig) * 100 if total_orig > 0 else 0.0

        return {
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunk_count": len(chunks),
            "original_tokens": total_orig,
            "compressed_tokens": total_comp,
            "token_savings_pct": round(savings, 1),
        }

    def _reembed_existing(self):
        """Re-embed all existing chunks with updated IDF weights."""
        self._store.reembed_all(self._embedder)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Hybrid search for relevant chunks (TF-IDF cosine + BM25).

        Uses inverted index for fast candidate retrieval, then scores
        candidates with both cosine similarity and BM25.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters {doc_type, doc_name, doc_id}

        Returns:
            List of dicts with chunk content, score, doc info
        """
        # Get candidates from inverted index
        query_tokens = self._embedder._tokenize(query)
        query_token_set = set(query_tokens)
        candidate_ids = self._store._inverted.candidates(query_token_set)

        # If inverted index is empty (no index built), fall back to None (full scan)
        if not candidate_ids and self._store._inverted.token_count == 0:
            candidate_ids = None

        # BM25 scores
        bm25_scores = self._embedder.bm25_search(query)

        # If BM25 found results not in cosine candidates, add them
        if candidate_ids is not None and bm25_scores:
            candidate_ids = candidate_ids | set(bm25_scores.keys())

        # Apply metadata filters to narrow candidates
        if filters:
            candidate_ids = self._apply_filters(candidate_ids, filters)

        # Embed query for cosine similarity
        query_emb = self._embedder.embed(query)

        # Hybrid search
        results = self._store.search(
            query_emb, top_k=top_k,
            candidate_ids=candidate_ids,
            bm25_scores=bm25_scores,
            bm25_weight=self._bm25_weight,
        )

        return [
            {
                "content": r.chunk.compressed,
                "original": r.chunk.content,
                "score": round(r.score, 4),
                "cosine_score": round(r.cosine_score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "doc_name": r.doc_name,
                "doc_id": r.chunk.doc_id,
                "chunk_index": r.chunk.chunk_index,
                "token_count": r.chunk.token_count,
            }
            for r in results
        ]

    def _apply_filters(self, candidate_ids: Optional[set], filters: Dict) -> Optional[set]:
        """Apply metadata filters to narrow candidate chunks."""
        # Build SQL filter for matching chunk IDs
        conditions = []
        params = []

        if "doc_type" in filters:
            conditions.append("d.doc_type = ?")
            params.append(filters["doc_type"])
        if "doc_name" in filters:
            conditions.append("d.name LIKE ?")
            params.append(f"%{filters['doc_name']}%")
        if "doc_id" in filters:
            conditions.append("c.doc_id = ?")
            params.append(int(filters["doc_id"]))

        if not conditions:
            return candidate_ids

        where = " AND ".join(conditions)
        query = f"""SELECT c.id FROM rag_chunks c
                    JOIN rag_documents d ON c.doc_id = d.id
                    WHERE {where}"""

        rows = self._store._conn.execute(query, params).fetchall()
        filtered_ids = {r[0] for r in rows}

        if candidate_ids is not None:
            return candidate_ids & filtered_ids
        return filtered_ids

    def multi_query_search(
        self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query RAG: decompose a complex query into sub-queries.

        Splits the query by conjunctions and punctuation, runs each sub-query
        independently, then merges and re-ranks results by max score per chunk.

        This helps when a user asks a compound question like:
        "What is the revenue for Q3 and what were the top products?"

        Args:
            query: Complex search query
            top_k: Total results to return
            filters: Optional metadata filters
        """
        # Decompose query into sub-queries
        sub_queries = self._decompose_query(query)

        if len(sub_queries) <= 1:
            # Not decomposable, use standard search
            return self.search(query, top_k=top_k, filters=filters)

        # Search each sub-query
        all_results: Dict[int, Dict] = {}  # chunk_id → best result
        for sq in sub_queries:
            results = self.search(sq, top_k=top_k, filters=filters)
            for r in results:
                key = (r["doc_id"], r["chunk_index"])
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Sort by score and return top_k
        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries.

        Splits on: 'and', 'or', '?', ';', commas in question context.
        Only decomposes if sub-queries are meaningful (>3 words each).
        """
        import re

        # Split on common conjunctions and punctuation
        parts = re.split(r'\band\b|\bor\b|[?;]|,\s*(?=\w+\s+\w+)', query, flags=re.IGNORECASE)
        parts = [p.strip().rstrip('?').strip() for p in parts if p.strip()]

        # Filter out very short fragments (likely not meaningful queries)
        meaningful = [p for p in parts if len(p.split()) >= 3]

        if len(meaningful) >= 2:
            return meaningful

        # Not decomposable
        return [query]

    # ------------------------------------------------------------------
    # Chat augmentation
    # ------------------------------------------------------------------

    def augment_messages(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 5,
        max_context_tokens: int = 2048,
    ) -> List[Dict[str, str]]:
        """Augment chat messages with RAG context.

        Finds the last user message, searches the RAG store for relevant
        context, and injects it as a system message before the user's query.
        """
        last_user_msg = None
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_msg = messages[i]["content"]
                last_user_idx = i
                break

        if not last_user_msg:
            return messages

        results = self.search(last_user_msg, top_k=top_k)
        if not results:
            return messages

        context_parts = []
        total_tokens = 0
        for r in results:
            if total_tokens + r["token_count"] > max_context_tokens:
                break
            context_parts.append(r["content"])
            total_tokens += r["token_count"]

        if not context_parts:
            return messages

        context_text = (
            "Relevant context from uploaded documents:\n"
            + "\n---\n".join(context_parts)
            + "\n\nUse the above context to help answer the user's question. "
            "If the context is not relevant, ignore it."
        )

        augmented = list(messages)
        rag_msg = {"role": "system", "content": context_text}
        augmented.insert(last_user_idx, rag_msg)

        return augmented

    # ------------------------------------------------------------------
    # Document management (delegates to store)
    # ------------------------------------------------------------------

    def list_documents(self) -> List[Dict]:
        """List all documents."""
        docs = self._store.list_documents()
        return [
            {
                "id": d.id,
                "name": d.name,
                "doc_type": d.doc_type,
                "chunk_count": d.chunk_count,
                "original_tokens": d.original_tokens,
                "compressed_tokens": d.compressed_tokens,
                "token_savings_pct": round(d.savings_pct, 1),
                "created_at": d.created_at,
            }
            for d in docs
        ]

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get document details with chunk previews."""
        doc = self._store.get_document(doc_id)
        if not doc:
            return None

        chunks = self._store.get_chunks(doc_id)
        return {
            "id": doc.id,
            "name": doc.name,
            "doc_type": doc.doc_type,
            "chunk_count": doc.chunk_count,
            "original_tokens": doc.original_tokens,
            "compressed_tokens": doc.compressed_tokens,
            "token_savings_pct": round(doc.savings_pct, 1),
            "created_at": doc.created_at,
            "chunks": [
                {
                    "index": c.chunk_index,
                    "preview": c.content[:200],
                    "compressed_preview": c.compressed[:200],
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        }

    def remove_document(self, doc_id: int) -> bool:
        """Remove a document and clean up BM25 index."""
        # Remove chunks from BM25
        chunks = self._store.get_chunks(doc_id)
        for c in chunks:
            self._embedder.remove_from_bm25(c.id)
        return self._store.remove_document(doc_id)

    def stats(self) -> Dict:
        """Get RAG store statistics."""
        return self._store.stats()
