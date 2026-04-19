"""
RAG Pipeline — End-to-end document ingestion, conversion, and retrieval.

This is the main orchestrator:
    1. Ingest any file → parse → chunk → embed → write .zpf + store vectors
    2. Query → embed → vector search → extract relevant blocks → format context

Works with both CLI and Web UI through a shared RAGPipeline instance.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .embedder import Embedder
from .parsers import ParsedDocument, parse_bytes, parse_file
from .semantic_chunker import SemanticChunker
from .vector_store import SearchResult, VectorStore
from .zpf_reader import ZPFReader
from .zpf_spec import BlockType, SemanticBlock, ZPFHeader
from .zpf_writer import write_zpf

# Default store directory
_DEFAULT_STORE = os.path.join(os.path.expanduser("~"), ".zse", "rag_store")


class RAGPipeline:
    """
    End-to-end RAG pipeline with .zpf semantic compression.

    Usage:
        pipeline = RAGPipeline()
        doc_id = pipeline.ingest("paper.pdf")
        results = pipeline.search("what is attention?")
        context = pipeline.get_context("what is attention?", max_tokens=2000)
    """

    def __init__(
        self,
        store_dir: Optional[str] = None,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        zpf_dir: Optional[str] = None,
    ):
        self.store_dir = store_dir or _DEFAULT_STORE
        self.zpf_dir = zpf_dir or os.path.join(self.store_dir, "zpf_files")

        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.zpf_dir, exist_ok=True)

        self._embedder = Embedder(model_name=embedding_model)
        self._chunker = SemanticChunker()
        self._store = VectorStore(
            store_dir=self.store_dir,
            dimension=self._embedder.dimension,
        )

    # ----- Ingestion -----

    def ingest(
        self,
        file_path: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Ingest a file: parse → chunk → embed → store + write .zpf.

        Args:
            file_path:  Path to the document.
            title:      Optional title override.
            metadata:   Extra metadata.

        Returns:
            doc_id assigned to this document.
        """
        path = Path(file_path)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()[:16]

        # Parse
        doc = parse_file(str(path))

        if not title:
            title = doc.title or path.stem

        # Chunk
        blocks = self._chunker.chunk(doc)
        if not blocks:
            raise ValueError(f"No semantic blocks extracted from {path.name}")

        # Embed — contextual enrichment: prepend section hierarchy for
        # better cross-section retrieval while storing original content
        texts = []
        for b in blocks:
            section_path = (b.metadata or {}).get("section_path", "")
            if section_path:
                texts.append(f"{section_path} > {b.content}")
            else:
                texts.append(b.content)
        embeddings = self._embedder.embed(texts)

        # Attach embedding bytes to blocks
        for i, block in enumerate(blocks):
            block.embedding = embeddings[i].tobytes()

        # Write .zpf
        zpf_path = Path(self.zpf_dir) / f"{path.stem}.zpf"
        written = write_zpf(
            blocks=blocks,
            output_path=zpf_path,
            title=title,
            source_type=doc.source_type,
            source_hash=source_hash,
            original_size=doc.original_size,
            embedding_model=self._embedder.name,
            embedding_dim=self._embedder.dimension,
            metadata=metadata or {},
        )

        # Read back the doc_id from written file
        reader = ZPFReader(written)
        doc_id = reader.header.doc_id

        # Add to vector store
        block_dicts = [
            {
                "content": b.content,
                "block_type": int(b.block_type),
                "summary": b.summary,
                "token_count": b.token_count,
                "metadata": {
                    "source": path.name,
                    "zpf_file": str(zpf_path),
                    **(b.metadata or {}),
                },
            }
            for b in blocks
        ]
        self._store.add(doc_id, embeddings, block_dicts)

        return doc_id

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Ingest from raw bytes (for uploads via API)."""
        source_hash = hashlib.sha256(data).hexdigest()[:16]

        doc = parse_bytes(data, filename)
        if not title:
            title = doc.title or Path(filename).stem

        blocks = self._chunker.chunk(doc)
        if not blocks:
            raise ValueError(f"No semantic blocks extracted from {filename}")

        texts = [b.content for b in blocks]
        embeddings = self._embedder.embed(texts)

        for i, block in enumerate(blocks):
            block.embedding = embeddings[i].tobytes()

        stem = Path(filename).stem
        zpf_path = Path(self.zpf_dir) / f"{stem}.zpf"
        written = write_zpf(
            blocks=blocks,
            output_path=zpf_path,
            title=title,
            source_type=doc.source_type,
            source_hash=source_hash,
            original_size=doc.original_size,
            embedding_model=self._embedder.name,
            embedding_dim=self._embedder.dimension,
            metadata=metadata or {},
        )

        reader = ZPFReader(written)
        doc_id = reader.header.doc_id

        block_dicts = [
            {
                "content": b.content,
                "block_type": int(b.block_type),
                "summary": b.summary,
                "token_count": b.token_count,
                "metadata": {"source": filename, "zpf_file": str(zpf_path)},
            }
            for b in blocks
        ]
        self._store.add(doc_id, embeddings, block_dicts)
        return doc_id

    def ingest_zpf(self, zpf_path: str) -> str:
        """Ingest a pre-built .zpf file directly."""
        reader = ZPFReader(zpf_path)
        doc_id = reader.header.doc_id
        embs = reader.embeddings()
        blocks = reader.read_all_blocks()

        block_dicts = [
            {
                "content": b.content,
                "block_type": int(b.block_type),
                "summary": b.summary,
                "token_count": b.token_count,
                "metadata": {"zpf_file": zpf_path},
            }
            for b in blocks
        ]
        self._store.add(doc_id, embs, block_dicts)
        return doc_id

    # ----- Convert Only (no store) -----

    def convert(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """
        Convert a file to .zpf without adding to the RAG store.

        Returns:
            Path to the output .zpf file.
        """
        path = Path(file_path)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()[:16]

        doc = parse_file(str(path))
        blocks = self._chunker.chunk(doc)
        if not blocks:
            raise ValueError(f"No semantic blocks extracted from {path.name}")

        texts = [b.content for b in blocks]
        embeddings = self._embedder.embed(texts)
        for i, block in enumerate(blocks):
            block.embedding = embeddings[i].tobytes()

        if not output_path:
            output_path = str(path.with_suffix(".zpf"))

        write_zpf(
            blocks=blocks,
            output_path=output_path,
            title=title or doc.title or path.stem,
            source_type=doc.source_type,
            source_hash=source_hash,
            original_size=doc.original_size,
            embedding_model=self._embedder.name,
            embedding_dim=self._embedder.dimension,
        )
        return output_path

    # ----- Search & Retrieval -----

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Semantic search across all ingested documents."""
        query_vec = self._embedder.embed_query(query)
        return self._store.search(
            query_vec=query_vec,
            top_k=top_k,
            doc_filter=doc_filter,
            score_threshold=score_threshold,
            query_text=query,
        )

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        top_k: int = 10,
        doc_filter: Optional[str] = None,
    ) -> str:
        """
        Retrieve relevant context formatted for LLM injection.

        Returns a token-budget-aware context string built from the
        highest-scoring semantic blocks.
        """
        from zse.core.zrag.vector_store import _LIST_PATTERN

        effective_top_k = 15 if _LIST_PATTERN.search(query) else top_k
        results = self.search(query, top_k=effective_top_k, doc_filter=doc_filter)

        context_parts = []
        token_count = 0

        for r in results:
            block_tokens = len(r.content) // 4  # rough estimate
            if token_count + block_tokens > max_tokens:
                break

            block_type_name = BlockType(r.block_type).name if r.block_type <= 10 else "TEXT"
            header = f"[{block_type_name}]"
            if r.summary:
                header += f" {r.summary}"

            context_parts.append(f"{header}\n{r.content}")
            token_count += block_tokens

        if not context_parts:
            return ""

        return "\n\n---\n\n".join(context_parts)

    # ----- Management -----

    def remove(self, doc_id: str) -> int:
        """Remove a document from the store."""
        return self._store.remove(doc_id)

    def list_documents(self) -> List[Dict]:
        """List all documents in the store."""
        return self._store.list_documents()

    def list_zpf_files(self) -> List[Dict]:
        """List all .zpf files in the zpf directory."""
        zpf_dir = Path(self.zpf_dir)
        files = []
        for zpf in zpf_dir.glob("*.zpf"):
            try:
                reader = ZPFReader(zpf)
                files.append(
                    {
                        "path": str(zpf),
                        "doc_id": reader.header.doc_id,
                        "title": reader.header.title,
                        "source_type": reader.header.source_type,
                        "block_count": reader.header.block_count,
                        "total_tokens": reader.header.total_tokens,
                        "original_size": reader.header.original_size,
                        "compressed_size": reader.header.compressed_size,
                    }
                )
            except Exception:
                files.append({"path": str(zpf), "error": "unreadable"})
        return files

    def inspect_zpf(self, zpf_path: str) -> Dict[str, Any]:
        """Inspect a .zpf file and return its metadata."""
        reader = ZPFReader(zpf_path)
        header = reader.header
        summaries = reader.get_block_summaries()

        compression_ratio = (
            header.original_size / header.compressed_size if header.compressed_size > 0 else 0
        )
        token_ratio = (
            header.total_tokens / (header.original_size // 4) if header.original_size > 0 else 0
        )

        return {
            "doc_id": header.doc_id,
            "title": header.title,
            "source_type": header.source_type,
            "created_at": header.created_at,
            "block_count": header.block_count,
            "total_tokens": header.total_tokens,
            "embedding_model": header.embedding_model,
            "embedding_dim": header.embedding_dim,
            "original_size": header.original_size,
            "compressed_size": header.compressed_size,
            "compression_ratio": round(compression_ratio, 2),
            "token_efficiency": round(token_ratio, 2),
            "blocks": summaries,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        return {
            "total_documents": self._store.total_documents,
            "total_blocks": self._store.total_blocks,
            "embedding_model": self._embedder.name,
            "embedding_dim": self._embedder.dimension,
            "store_dir": self.store_dir,
            "zpf_dir": self.zpf_dir,
        }

    # ----- Export (anti lock-in) -----

    def export_zpf(
        self,
        zpf_path: str,
        output_path: str,
        format: str = "jsonl",
    ) -> str:
        """
        Export a .zpf file to an open format.

        Args:
            zpf_path:    Path to the .zpf file.
            output_path: Where to write the export.
            format:      'jsonl', 'markdown', or 'json'.

        Returns:
            Path to the exported file.
        """
        reader = ZPFReader(zpf_path)
        blocks = reader.read_all_blocks()
        header = reader.header

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(out, "w", encoding="utf-8") as f:
                # Header line
                f.write(
                    json.dumps(
                        {
                            "type": "header",
                            "doc_id": header.doc_id,
                            "title": header.title,
                            "source_type": header.source_type,
                            "block_count": header.block_count,
                            "total_tokens": header.total_tokens,
                            "embedding_model": header.embedding_model,
                            "original_size": header.original_size,
                        }
                    )
                    + "\n"
                )
                for i, b in enumerate(blocks):
                    f.write(
                        json.dumps(
                            {
                                "type": "block",
                                "index": i,
                                "block_type": BlockType(b.block_type).name,
                                "content": b.content,
                                "token_count": b.token_count,
                                "summary": b.summary,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        elif format == "markdown":
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"# {header.title}\n\n")
                f.write(f"> Exported from .zpf | doc_id: {header.doc_id} | ")
                f.write(f"source: {header.source_type} | blocks: {header.block_count}\n\n")
                for i, b in enumerate(blocks):
                    btype = BlockType(b.block_type).name
                    f.write(f"## Block {i} [{btype}]\n\n")
                    if b.summary:
                        f.write(f"*{b.summary}*\n\n")
                    f.write(f"{b.content}\n\n---\n\n")

        elif format == "json":
            data = {
                "doc_id": header.doc_id,
                "title": header.title,
                "source_type": header.source_type,
                "block_count": header.block_count,
                "total_tokens": header.total_tokens,
                "embedding_model": header.embedding_model,
                "blocks": [
                    {
                        "index": i,
                        "block_type": BlockType(b.block_type).name,
                        "content": b.content,
                        "token_count": b.token_count,
                        "summary": b.summary,
                    }
                    for i, b in enumerate(blocks)
                ],
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"Unsupported export format: {format}. Use: jsonl, markdown, json")

        return str(out)

    # ----- Re-indexing -----

    def reindex(
        self,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Re-embed and re-index all .zpf files with a new (or same) model.

        This solves embedding model migration: switch from MiniLM to a
        better model without re-ingesting source documents.

        Args:
            embedding_model: New model name. Uses current if None.

        Returns:
            Summary dict with counts.
        """
        if embedding_model and embedding_model != self._embedder.name:
            self._embedder = Embedder(model_name=embedding_model)
            self._store = VectorStore(
                store_dir=self.store_dir,
                dimension=self._embedder.dimension,
            )

        zpf_dir = Path(self.zpf_dir)
        zpf_files = list(zpf_dir.glob("*.zpf"))

        if not zpf_files:
            return {"reindexed": 0, "blocks": 0, "model": self._embedder.name}

        # Clear existing store
        for doc in self._store.list_documents():
            self._store.remove(doc["doc_id"])

        total_blocks = 0
        reindexed = 0

        for zpf in zpf_files:
            try:
                reader = ZPFReader(zpf)
                blocks = reader.read_all_blocks()
                doc_id = reader.header.doc_id

                texts = [b.content for b in blocks]
                embeddings = self._embedder.embed(texts)

                block_dicts = [
                    {
                        "content": b.content,
                        "block_type": int(b.block_type),
                        "summary": b.summary,
                        "token_count": b.token_count,
                        "metadata": {"zpf_file": str(zpf)},
                    }
                    for b in blocks
                ]
                self._store.add(doc_id, embeddings, block_dicts)

                # Update .zpf file with new embeddings
                for i, block in enumerate(blocks):
                    block.embedding = embeddings[i].tobytes()

                write_zpf(
                    blocks=blocks,
                    output_path=zpf,
                    title=reader.header.title,
                    source_type=reader.header.source_type,
                    source_hash=reader.header.source_hash,
                    original_size=reader.header.original_size,
                    embedding_model=self._embedder.name,
                    embedding_dim=self._embedder.dimension,
                    doc_id=doc_id,
                )

                total_blocks += len(blocks)
                reindexed += 1
            except Exception as e:
                import sys

                print(f"Warning: failed to reindex {zpf.name}: {e}", file=sys.stderr)

        return {
            "reindexed": reindexed,
            "total_files": len(zpf_files),
            "blocks": total_blocks,
            "model": self._embedder.name,
        }


# ----- Module-level singleton -----

_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline(
    store_dir: Optional[str] = None,
    embedding_model: Optional[str] = "all-MiniLM-L6-v2",
) -> RAGPipeline:
    """Get or create the shared RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            store_dir=store_dir,
            embedding_model=embedding_model,
        )
    return _pipeline


def init_rag_pipeline(
    store_dir: Optional[str] = None,
    embedding_model: Optional[str] = "all-MiniLM-L6-v2",
) -> RAGPipeline:
    """Initialize the RAG pipeline (call at startup)."""
    global _pipeline
    _pipeline = RAGPipeline(
        store_dir=store_dir,
        embedding_model=embedding_model,
    )
    return _pipeline
