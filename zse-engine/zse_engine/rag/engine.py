"""ZSE RAG Engine — Orchestrator for document ingestion, search, and context injection.

Ties together: parsers, ZPF compression, hybrid embeddings (TF-IDF + BM25),
inverted index, and the document store.
Provides a clean API for the HTTP layer and chat integration.
"""

import base64
import json
from typing import List, Optional, Dict, Any, Set, Tuple

from zse_engine.rag.parser import get_parser, Chunk
from zse_engine.rag.embeddings import TokenEmbedder
from zse_engine.rag.store import RAGStore, SearchResult, StoredDocument
from zse_engine.rag.dense_embedder import DenseEmbedder, dense_cosine
from zse_engine.rag.reranker import LLMReranker


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

        # Dense + rerank — opt-in, wired post-init via set_model_runner()
        self._dense_embedder: Optional[DenseEmbedder] = None
        self._reranker: Optional[LLMReranker] = None
        self._dense_enabled_by_default = False

        # Rebuild IDF + inverted index + BM25 from existing documents
        self._rebuild_all()

    # ------------------------------------------------------------------
    # Modern RAG: dense embeddings + LLM reranker (zero extra deps)
    # ------------------------------------------------------------------

    def set_model_runner(self, model_runner, enable_dense_by_default: bool = True):
        """Wire the inference LLM into RAG for dense embeddings + reranking.

        After this call, search(use_dense=True, use_rerank=True) becomes valid.
        Re-embedding of existing chunks happens lazily on first search.
        """
        if model_runner is None or self._tokenizer is None:
            return
        try:
            self._dense_embedder = DenseEmbedder(model_runner, self._tokenizer)
            self._reranker = LLMReranker(model_runner, self._tokenizer)
            self._dense_enabled_by_default = enable_dense_by_default
        except Exception:
            self._dense_embedder = None
            self._reranker = None

    def has_dense(self) -> bool:
        return self._dense_embedder is not None

    def backfill_dense_embeddings(self, batch_size: int = 32) -> int:
        """Compute dense vectors for any chunks that don't have one. Returns count."""
        if self._dense_embedder is None:
            return 0
        missing = self._store.all_chunk_ids_missing_dense()
        if not missing:
            return 0
        done = 0
        for cid in missing:
            text = self._store.get_chunk_text(cid)
            if not text:
                continue
            vec = self._dense_embedder.embed(text)
            if vec is not None:
                self._store.set_dense_vector(cid, vec)
                done += 1
        return done

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
        pdf_password: Optional[str] = None,
        ocr_fn=None,
    ) -> Dict[str, Any]:
        """Ingest a document: parse, compress, embed, store.

        Args:
            filename: Original filename (used for parser selection)
            content: Raw file content as bytes
            chunk_size: Override default chunk size
            overlap: Override default overlap
            pdf_password: Optional non-empty user password for encrypted PDFs
            ocr_fn: Optional ``ocr_fn(image_bytes, format_hint) -> str``
                callable invoked when a PDF yields no extractable text.
                The user supplies the OCR backend (pytesseract, EasyOCR,
                cloud API); ZSE itself remains zero-dep.

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
        parse_meta: Dict[str, Any] = {"source": filename}
        if pdf_password:
            parse_meta["pdf_password"] = pdf_password
        if ocr_fn is not None:
            parse_meta["ocr_fn"] = ocr_fn
        chunks = parser.parse(text, tokenizer=self._tokenizer, metadata=parse_meta)

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

        # Dense embeddings (if model_runner wired)
        if self._dense_embedder is not None:
            for sc in stored_chunks:
                vec = self._dense_embedder.embed(sc.content)
                if vec is not None:
                    self._store.set_dense_vector(sc.id, vec)

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

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        use_dense: Optional[bool] = None,
        use_rerank: bool = False,
        candidate_pool: int = 30,
        fusion: str = "rrf",
    ) -> List[Dict[str, Any]]:
        """Hybrid search with optional dense retrieval and LLM reranking.

        Pipeline:
          1. BM25 + TF-IDF cosine (sparse) over inverted-index candidates
          2. If dense enabled: dense cosine over the union of candidates +
             top BM25 / sparse results
          3. Fuse rankings via Reciprocal Rank Fusion (or weighted-sum if
             fusion="weighted") to produce top `candidate_pool` results
          4. If use_rerank: LLM cross-encoder rerank top `candidate_pool` \u2192 top_k

        Args:
            query: Search query text
            top_k: Final number of results
            filters: Optional metadata filters {doc_type, doc_name, doc_id}
            use_dense: Force dense on/off (None = auto if model_runner wired)
            use_rerank: Run LLM cross-encoder rerank on the candidate pool
            candidate_pool: Initial candidate count before rerank (>=top_k)
            fusion: 'rrf' (Reciprocal Rank Fusion, default) or 'weighted'
        """
        if use_dense is None:
            use_dense = self._dense_enabled_by_default and self._dense_embedder is not None
        elif use_dense and self._dense_embedder is None:
            use_dense = False  # Not wired \u2014 silently fall back

        # ---- 1. Candidate pruning via inverted index ----
        query_tokens = self._embedder._tokenize(query)
        query_token_set = set(query_tokens)
        candidate_ids = self._store._inverted.candidates(query_token_set)
        if not candidate_ids and self._store._inverted.token_count == 0:
            candidate_ids = None

        # ---- 2. Sparse signals: BM25 + TF-IDF cosine ----
        bm25_scores = self._embedder.bm25_search(query)
        if candidate_ids is not None and bm25_scores:
            candidate_ids = candidate_ids | set(bm25_scores.keys())
        if filters:
            candidate_ids = self._apply_filters(candidate_ids, filters)

        query_emb = self._embedder.embed(query)
        # Pull a larger pool than top_k so rerank / fusion has signal
        sparse_pool = max(candidate_pool, top_k * 4)
        sparse_results = self._store.search(
            query_emb, top_k=sparse_pool,
            candidate_ids=candidate_ids,
            bm25_scores=bm25_scores,
            bm25_weight=self._bm25_weight,
        )

        # ---- 3. Dense signal (optional) ----
        dense_scores: Dict[int, float] = {}
        if use_dense:
            qvec = self._dense_embedder.embed_query(query)
            if qvec:
                # Score the union of (sparse pool, BM25 hits)
                cand_ids = {r.chunk.id for r in sparse_results}
                cand_ids |= set(bm25_scores.keys())
                # Filter to chunks that actually have a stored dense vector
                doc_vecs = self._store.get_dense_vectors(cand_ids)
                for cid, dvec in doc_vecs.items():
                    dense_scores[cid] = dense_cosine(qvec, dvec)
                # Backfill any high-BM25 candidates missing dense vectors
                if self._dense_embedder is not None:
                    missing = [c for c in cand_ids if c not in doc_vecs]
                    for cid in missing[:8]:  # cap to bound latency
                        text = self._store.get_chunk_text(cid)
                        if not text:
                            continue
                        vec = self._dense_embedder.embed(text)
                        if vec is not None:
                            self._store.set_dense_vector(cid, vec)
                            dense_scores[cid] = dense_cosine(qvec, vec)

        # ---- 4. Fuse rankings ----
        fused = self._fuse(
            sparse_results=sparse_results,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            mode=fusion,
            top_n=max(candidate_pool, top_k),
        )

        # ---- 5. LLM cross-encoder rerank (optional, expensive) ----
        rerank_scores: Dict[int, float] = {}
        if use_rerank and self._reranker is not None and fused:
            pairs = [(r.chunk.id, r.chunk.content) for r in fused[:candidate_pool]]
            rerank_out = self._reranker.rerank(query, pairs, top_n=len(pairs))
            rerank_scores = {cid: s for cid, s in rerank_out}
            # Sort fused by rerank score (descending)
            fused.sort(key=lambda r: rerank_scores.get(r.chunk.id, -1e9), reverse=True)
            for r in fused:
                if r.chunk.id in rerank_scores:
                    r.rerank_score = rerank_scores[r.chunk.id]

        return [
            {
                "content": r.chunk.compressed,
                "original": r.chunk.content,
                "score": round(r.score, 4),
                "cosine_score": round(r.cosine_score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "dense_score": round(dense_scores.get(r.chunk.id, 0.0), 4),
                "rerank_score": (round(r.rerank_score, 4)
                                  if r.rerank_score is not None else None),
                "doc_name": r.doc_name,
                "doc_id": r.chunk.doc_id,
                "chunk_index": r.chunk.chunk_index,
                "token_count": r.chunk.token_count,
            }
            for r in fused[:top_k]
        ]

    def _fuse(
        self,
        sparse_results: List[SearchResult],
        bm25_scores: Dict[int, float],
        dense_scores: Dict[int, float],
        mode: str = "rrf",
        top_n: int = 20,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Combine sparse + dense rankings.

        - 'rrf' (default): Reciprocal Rank Fusion \u2014 industry standard, robust
          to score scale differences. score = sum(1 / (k + rank_i)).
        - 'weighted': sparse score blended with normalized dense score.
        """
        # Build per-list rankings (rank starts at 0)
        sparse_rank: Dict[int, int] = {
            r.chunk.id: i for i, r in enumerate(sparse_results)
        }
        # BM25-only ranking (descending score)
        bm25_sorted = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_rank: Dict[int, int] = {cid: i for i, (cid, _) in enumerate(bm25_sorted)}
        # Dense ranking
        dense_sorted = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_rank: Dict[int, int] = {cid: i for i, (cid, _) in enumerate(dense_sorted)}

        # Universe of candidates = union of all sources
        candidates: Dict[int, SearchResult] = {r.chunk.id: r for r in sparse_results}
        # (Sparse pool already includes BM25 contributors via store.search)

        if mode == "rrf":
            fused_scores: Dict[int, float] = {}
            for cid in candidates:
                s = 0.0
                if cid in sparse_rank:
                    s += 1.0 / (rrf_k + sparse_rank[cid] + 1)
                if cid in bm25_rank:
                    s += 1.0 / (rrf_k + bm25_rank[cid] + 1)
                if cid in dense_rank:
                    s += 1.0 / (rrf_k + dense_rank[cid] + 1)
                fused_scores[cid] = s
            for cid, sr in candidates.items():
                sr.score = fused_scores.get(cid, 0.0)
                sr.dense_score = dense_scores.get(cid, 0.0)
        else:
            # weighted: blend existing hybrid score (already in sr.score) with
            # normalized dense. Dense is in [-1,1] (cosine on unit vectors).
            max_dense = max(dense_scores.values()) if dense_scores else 1.0
            if max_dense <= 0:
                max_dense = 1.0
            dense_w = 0.4 if dense_scores else 0.0
            for cid, sr in candidates.items():
                d = dense_scores.get(cid, 0.0) / max_dense
                sr.dense_score = dense_scores.get(cid, 0.0)
                sr.score = (1 - dense_w) * sr.score + dense_w * max(d, 0.0)

        fused = list(candidates.values())
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_n]


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
