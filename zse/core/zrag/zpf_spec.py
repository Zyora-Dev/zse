"""
.zpf (Z Packed Format) Specification

A semantic compression format for LLM-aware document storage.

Philosophy:
    Traditional RAG stores raw text chunks → wastes tokens on formatting,
    boilerplate, repeated headers, navigation text, etc.

    .zpf stores SEMANTIC BLOCKS — content units that encode only what the
    LLM needs to understand the material. Noise is stripped at write time,
    not at query time.

Binary Layout:
    ┌──────────────────────────────────┐
    │ Magic: ZPF\x00  (4 bytes)        │
    │ Version: u8.u8.u8.u8 (4 bytes)  │
    ├──────────────────────────────────┤
    │ Header Length (4 bytes, LE u32)  │
    │ Header (JSON, UTF-8):           │
    │   - doc_id, title, source_type  │
    │   - created_at, source_hash     │
    │   - block_count, total_tokens   │
    │   - embedding_model, embed_dim  │
    │   - metadata {}                 │
    ├──────────────────────────────────┤
    │ Block Index Length (4 bytes)     │
    │ Block Index (JSON array):       │
    │   [{offset, size, block_type,   │
    │     token_count, semantic_hash, │
    │     summary}]                   │
    ├──────────────────────────────────┤
    │ Embedding Section Length (4 B)  │
    │ Embeddings (raw float32):       │
    │   block_count × embed_dim × 4  │
    ├──────────────────────────────────┤
    │ Content Section:                │
    │   Block 0 data (UTF-8 bytes)   │
    │   Block 1 data (UTF-8 bytes)   │
    │   ...                          │
    └──────────────────────────────────┘

Each block stores semantically compressed text — not the original raw
content, but a distilled version that preserves meaning for the LLM.
"""

import struct
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import json

# Magic bytes: "ZPF" + null
ZPF_MAGIC = b"ZPF\x00"
ZPF_VERSION = (1, 0, 0, 0)

# Limits
MAX_HEADER_SIZE = 1 * 1024 * 1024    # 1 MB
MAX_INDEX_SIZE = 10 * 1024 * 1024     # 10 MB
MAX_EMBED_SIZE = 500 * 1024 * 1024    # 500 MB


class BlockType(IntEnum):
    """Semantic block types — what kind of content is this."""
    TEXT = 0          # General text content
    DEFINITION = 1   # A definition, concept, or term explanation
    FACT = 2         # A factual statement or data point
    PROCEDURE = 3    # Instructions, steps, how-to
    TABLE = 4        # Structured tabular data (compressed)
    CODE = 5         # Code snippet
    QA = 6           # Question-answer pair
    SUMMARY = 7      # Summary or abstract
    LIST = 8         # List of items
    REFERENCE = 9    # Citation, reference, or link
    METADATA = 10    # Document metadata block


@dataclass
class SemanticBlock:
    """
    A single semantic unit in a .zpf file.

    This is NOT a raw text chunk. It's a distilled representation
    of content that the LLM can efficiently consume.

    Fields:
        block_type: What kind of content (definition, fact, procedure, etc.)
        content: The semantically compressed text
        token_count: Estimated token count (for budget management)
        semantic_hash: Hash of the semantic content (for dedup)
        summary: One-line summary for index-level filtering
        source_range: (start_char, end_char) in original document
        embedding: Float32 vector (set during write, stripped on serialize)
        metadata: Extra block-level metadata
    """
    block_type: BlockType
    content: str
    token_count: int
    semantic_hash: str
    summary: str
    source_range: Tuple[int, int] = (0, 0)
    embedding: Optional[bytes] = None  # raw float32 bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_index_entry(self, offset: int, size: int) -> Dict[str, Any]:
        """Serialize to block index entry (no content, no embedding)."""
        return {
            "offset": offset,
            "size": size,
            "block_type": int(self.block_type),
            "token_count": self.token_count,
            "semantic_hash": self.semantic_hash,
            "summary": self.summary,
            "source_range": list(self.source_range),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization (for debugging)."""
        d = {
            "block_type": int(self.block_type),
            "content": self.content,
            "token_count": self.token_count,
            "semantic_hash": self.semantic_hash,
            "summary": self.summary,
            "source_range": list(self.source_range),
            "metadata": self.metadata,
        }
        return d


@dataclass
class ZPFHeader:
    """
    .zpf file header — document-level metadata.
    """
    doc_id: str
    title: str
    source_type: str        # "pdf", "docx", "html", "txt", "csv", "json", "md"
    created_at: str         # ISO 8601
    source_hash: str        # Hash of original document
    block_count: int
    total_tokens: int       # Sum of all block token counts
    embedding_model: str    # e.g. "all-MiniLM-L6-v2"
    embedding_dim: int      # e.g. 384
    original_size: int      # Original document size in bytes
    compressed_size: int    # Total .zpf file size
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "ZPFHeader":
        d = json.loads(data.decode("utf-8"))
        return cls(**d)
