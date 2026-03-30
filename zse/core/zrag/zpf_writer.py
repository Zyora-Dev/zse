"""
ZPF Writer — Serialize semantic blocks to .zpf binary format.

Takes parsed+chunked+embedded blocks and writes the compact binary
file per the spec in zpf_spec.py.
"""

import hashlib
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .zpf_spec import (
    ZPF_MAGIC,
    ZPF_VERSION,
    SemanticBlock,
    ZPFHeader,
)


def write_zpf(
    blocks: List[SemanticBlock],
    output_path: Union[str, Path],
    *,
    title: str = "",
    source_type: str = "unknown",
    source_hash: str = "",
    original_size: int = 0,
    embedding_model: str = "unknown",
    embedding_dim: int = 384,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Write semantic blocks to a .zpf file.

    Args:
        blocks:          List of SemanticBlock (with embeddings set).
        output_path:     Where to write the .zpf file.
        title:           Document title.
        source_type:     Original format (pdf, docx, etc.).
        source_hash:     Hash of original file.
        original_size:   Original file size in bytes.
        embedding_model: Name of the embedding model used.
        embedding_dim:   Dimension of embeddings.
        doc_id:          Document ID (auto-generated if None).
        metadata:        Extra header metadata.

    Returns:
        Path to the written .zpf file.
    """
    output_path = Path(output_path)
    if not doc_id:
        doc_id = uuid.uuid4().hex[:16]

    # ---- Encode content blocks ----
    content_chunks: List[bytes] = []
    for b in blocks:
        content_chunks.append(b.content.encode("utf-8"))

    # ---- Build block index ----
    index_entries = []
    offset = 0
    for i, b in enumerate(blocks):
        size = len(content_chunks[i])
        index_entries.append(b.to_index_entry(offset, size))
        offset += size

    import json
    index_json = json.dumps(index_entries, separators=(",", ":")).encode("utf-8")

    # ---- Collect embeddings ----
    embeddings_data = b""
    for b in blocks:
        if b.embedding:
            embeddings_data += b.embedding
        else:
            # Zero embedding if not set
            embeddings_data += b"\x00" * (embedding_dim * 4)

    # ---- Build header ----
    total_tokens = sum(b.token_count for b in blocks)
    header = ZPFHeader(
        doc_id=doc_id,
        title=title,
        source_type=source_type,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_hash=source_hash,
        block_count=len(blocks),
        total_tokens=total_tokens,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        original_size=original_size,
        compressed_size=0,  # placeholder, updated after write
        metadata=metadata or {},
    )
    header_json = header.to_json()

    # ---- Write binary ----
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Magic + version
        f.write(ZPF_MAGIC)
        f.write(struct.pack("4B", *ZPF_VERSION))

        # Header
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)

        # Block index
        f.write(struct.pack("<I", len(index_json)))
        f.write(index_json)

        # Embeddings
        f.write(struct.pack("<I", len(embeddings_data)))
        f.write(embeddings_data)

        # Content blocks (concatenated)
        for chunk in content_chunks:
            f.write(chunk)

        file_size = f.tell()

    # Update compressed_size in the file
    header.compressed_size = file_size
    header_json_final = header.to_json()
    if len(header_json_final) == len(header_json):
        with open(output_path, "r+b") as f:
            f.seek(8)  # skip magic + version
            f.write(struct.pack("<I", len(header_json_final)))
            f.write(header_json_final)

    return output_path


def blocks_to_zpf_bytes(
    blocks: List[SemanticBlock],
    *,
    title: str = "",
    source_type: str = "unknown",
    source_hash: str = "",
    original_size: int = 0,
    embedding_model: str = "unknown",
    embedding_dim: int = 384,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bytes:
    """
    Same as write_zpf but returns bytes instead of writing to disk.
    Useful for streaming or in-memory operations.
    """
    import io
    import json

    if not doc_id:
        doc_id = uuid.uuid4().hex[:16]

    content_chunks = [b.content.encode("utf-8") for b in blocks]

    index_entries = []
    offset = 0
    for i, b in enumerate(blocks):
        size = len(content_chunks[i])
        index_entries.append(b.to_index_entry(offset, size))
        offset += size

    index_json = json.dumps(index_entries, separators=(",", ":")).encode("utf-8")

    embeddings_data = b""
    for b in blocks:
        if b.embedding:
            embeddings_data += b.embedding
        else:
            embeddings_data += b"\x00" * (embedding_dim * 4)

    total_tokens = sum(b.token_count for b in blocks)
    header = ZPFHeader(
        doc_id=doc_id,
        title=title,
        source_type=source_type,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_hash=source_hash,
        block_count=len(blocks),
        total_tokens=total_tokens,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        original_size=original_size,
        compressed_size=0,
        metadata=metadata or {},
    )

    buf = io.BytesIO()
    header_json = header.to_json()
    buf.write(ZPF_MAGIC)
    buf.write(struct.pack("4B", *ZPF_VERSION))
    buf.write(struct.pack("<I", len(header_json)))
    buf.write(header_json)
    buf.write(struct.pack("<I", len(index_json)))
    buf.write(index_json)
    buf.write(struct.pack("<I", len(embeddings_data)))
    buf.write(embeddings_data)
    for chunk in content_chunks:
        buf.write(chunk)

    data = buf.getvalue()
    return data
