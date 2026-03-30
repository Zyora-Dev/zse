"""
ZPF Reader — Deserialize .zpf binary files back to semantic blocks.

Provides:
    - Full file read (header + all blocks + embeddings)
    - Lazy block access (read only what's needed)
    - Embedding extraction for vector search
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .zpf_spec import (
    ZPF_MAGIC,
    ZPF_VERSION,
    MAX_EMBED_SIZE,
    MAX_HEADER_SIZE,
    MAX_INDEX_SIZE,
    BlockType,
    SemanticBlock,
    ZPFHeader,
)


@dataclass
class ZPFFile:
    """Fully loaded .zpf file."""
    header: ZPFHeader
    blocks: List[SemanticBlock]
    embeddings: np.ndarray   # (block_count, embed_dim) float32


class ZPFReader:
    """
    Reader for .zpf binary files.

    Usage:
        reader = ZPFReader("doc.zpf")
        print(reader.header.title)
        print(reader.block_count)
        block = reader.read_block(0)
        embs = reader.embeddings()
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._header: Optional[ZPFHeader] = None
        self._index: Optional[List[Dict]] = None
        self._emb_offset: int = 0
        self._content_offset: int = 0
        self._parsed = False

        self._parse_structure()

    def _parse_structure(self):
        """Parse header, block index, and locate sections."""
        with open(self.path, "rb") as f:
            # Magic
            magic = f.read(4)
            if magic != ZPF_MAGIC:
                raise ValueError(
                    f"Not a .zpf file: bad magic {magic!r} "
                    f"(expected {ZPF_MAGIC!r})"
                )

            # Version
            version = struct.unpack("4B", f.read(4))
            # Allow reading any v1.x
            if version[0] != ZPF_VERSION[0]:
                raise ValueError(
                    f"Unsupported .zpf version: {version}, "
                    f"expected major={ZPF_VERSION[0]}"
                )

            # Header
            header_len = struct.unpack("<I", f.read(4))[0]
            if header_len > MAX_HEADER_SIZE:
                raise ValueError(f"Header too large: {header_len}")
            header_data = f.read(header_len)
            self._header = ZPFHeader.from_json(header_data)

            # Block index
            index_len = struct.unpack("<I", f.read(4))[0]
            if index_len > MAX_INDEX_SIZE:
                raise ValueError(f"Block index too large: {index_len}")
            index_data = f.read(index_len)
            self._index = json.loads(index_data.decode("utf-8"))

            # Embeddings section
            emb_len = struct.unpack("<I", f.read(4))[0]
            if emb_len > MAX_EMBED_SIZE:
                raise ValueError(f"Embedding section too large: {emb_len}")
            self._emb_offset = f.tell()
            f.seek(emb_len, 1)  # skip embeddings for now

            # Content section starts here
            self._content_offset = f.tell()

        self._parsed = True

    @property
    def header(self) -> ZPFHeader:
        return self._header

    @property
    def block_count(self) -> int:
        return self._header.block_count if self._header else 0

    @property
    def index(self) -> List[Dict]:
        return self._index or []

    def read_block(self, idx: int) -> SemanticBlock:
        """Read a single block by index."""
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Block index {idx} out of range [0, {len(self._index)})")

        entry = self._index[idx]
        offset = entry["offset"]
        size = entry["size"]

        with open(self.path, "rb") as f:
            f.seek(self._content_offset + offset)
            content = f.read(size).decode("utf-8")

        return SemanticBlock(
            block_type=BlockType(entry["block_type"]),
            content=content,
            token_count=entry.get("token_count", 0),
            semantic_hash=entry.get("semantic_hash", ""),
            summary=entry.get("summary", ""),
            source_range=tuple(entry.get("source_range", [0, 0])),
        )

    def read_all_blocks(self) -> List[SemanticBlock]:
        """Read all blocks from the file."""
        blocks = []
        with open(self.path, "rb") as f:
            for i, entry in enumerate(self._index):
                f.seek(self._content_offset + entry["offset"])
                content = f.read(entry["size"]).decode("utf-8")
                blocks.append(SemanticBlock(
                    block_type=BlockType(entry["block_type"]),
                    content=content,
                    token_count=entry.get("token_count", 0),
                    semantic_hash=entry.get("semantic_hash", ""),
                    summary=entry.get("summary", ""),
                    source_range=tuple(entry.get("source_range", [0, 0])),
                ))
        return blocks

    def embeddings(self) -> np.ndarray:
        """
        Load all embeddings as a (block_count, embed_dim) float32 array.
        """
        if not self._header or self._header.block_count == 0:
            return np.zeros((0, self._header.embedding_dim if self._header else 384), dtype=np.float32)

        dim = self._header.embedding_dim
        count = self._header.block_count
        expected_bytes = count * dim * 4

        with open(self.path, "rb") as f:
            f.seek(self._emb_offset)
            raw = f.read(expected_bytes)

        if len(raw) < expected_bytes:
            # Pad if file is truncated
            raw = raw + b"\x00" * (expected_bytes - len(raw))

        arr = np.frombuffer(raw, dtype=np.float32).reshape(count, dim)
        return arr.copy()  # Return writable copy

    def read_full(self) -> ZPFFile:
        """Load the entire .zpf file into memory."""
        return ZPFFile(
            header=self._header,
            blocks=self.read_all_blocks(),
            embeddings=self.embeddings(),
        )

    def get_block_summaries(self) -> List[Dict]:
        """Get lightweight summaries without reading content."""
        return [
            {
                "idx": i,
                "block_type": BlockType(e["block_type"]).name,
                "token_count": e.get("token_count", 0),
                "summary": e.get("summary", ""),
            }
            for i, e in enumerate(self._index)
        ]


def read_zpf_bytes(data: bytes) -> ZPFFile:
    """
    Parse a .zpf file from raw bytes (in-memory).
    """
    import io

    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != ZPF_MAGIC:
        raise ValueError(f"Not a .zpf file: bad magic {magic!r}")

    version = struct.unpack("4B", buf.read(4))
    if version[0] != ZPF_VERSION[0]:
        raise ValueError(f"Unsupported version: {version}")

    # Header
    header_len = struct.unpack("<I", buf.read(4))[0]
    header = ZPFHeader.from_json(buf.read(header_len))

    # Index
    index_len = struct.unpack("<I", buf.read(4))[0]
    index = json.loads(buf.read(index_len).decode("utf-8"))

    # Embeddings
    emb_len = struct.unpack("<I", buf.read(4))[0]
    emb_data = buf.read(emb_len)
    dim = header.embedding_dim
    count = header.block_count
    if emb_len >= count * dim * 4:
        embeddings = np.frombuffer(emb_data[:count * dim * 4], dtype=np.float32).reshape(count, dim).copy()
    else:
        embeddings = np.zeros((count, dim), dtype=np.float32)

    # Content
    content_start = buf.tell()
    blocks = []
    for entry in index:
        buf.seek(content_start + entry["offset"])
        content = buf.read(entry["size"]).decode("utf-8")
        blocks.append(SemanticBlock(
            block_type=BlockType(entry["block_type"]),
            content=content,
            token_count=entry.get("token_count", 0),
            semantic_hash=entry.get("semantic_hash", ""),
            summary=entry.get("summary", ""),
            source_range=tuple(entry.get("source_range", [0, 0])),
        ))

    return ZPFFile(header=header, blocks=blocks, embeddings=embeddings)
