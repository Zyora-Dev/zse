"""
ZSE RAG Module — Built-in Retrieval-Augmented Generation with .zpf format.

.zpf (Z Packed Format) is a semantic compression format designed from the
LLM's perspective. Instead of storing raw text chunks, it encodes only what
the LLM needs, dramatically reducing token usage while preserving comprehension.

Core modules:
- zpf_spec: .zpf binary format specification
- zpf_writer: Convert any document → .zpf
- zpf_reader: Read .zpf for retrieval
- parsers: PDF, DOCX, HTML, TXT, CSV, JSON, MD extraction
- semantic_chunker: Intent-first semantic chunking
- embedder: Embedding generation engine
- vector_store: Fast vector similarity search
- pipeline: End-to-end RAG pipeline
"""

from zse.core.zrag.zpf_spec import (
    ZPF_MAGIC,
    ZPF_VERSION,
    ZPFHeader,
    SemanticBlock,
    BlockType,
)
from zse.core.zrag.zpf_reader import ZPFReader, read_zpf_bytes
from zse.core.zrag.zpf_writer import write_zpf, blocks_to_zpf_bytes
from zse.core.zrag.embedder import Embedder
from zse.core.zrag.vector_store import VectorStore, SearchResult
from zse.core.zrag.semantic_chunker import SemanticChunker
from zse.core.zrag.parsers import parse_file, parse_bytes, ParsedDocument
from zse.core.zrag.pipeline import RAGPipeline, get_rag_pipeline, init_rag_pipeline

__all__ = [
    "ZPF_MAGIC",
    "ZPF_VERSION",
    "ZPFHeader",
    "SemanticBlock",
    "BlockType",
    "ZPFReader",
    "read_zpf_bytes",
    "write_zpf",
    "blocks_to_zpf_bytes",
    "Embedder",
    "VectorStore",
    "SearchResult",
    "SemanticChunker",
    "parse_file",
    "parse_bytes",
    "ParsedDocument",
    "RAGPipeline",
    "get_rag_pipeline",
    "init_rag_pipeline",
]
