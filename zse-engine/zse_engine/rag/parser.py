"""ZSE RAG Document Parsers — Zero-dependency parsers for common formats.

Parsers:
    TextParser  — .txt, .md files (paragraph-based chunking)
    JSONParser  — .json files (schema extraction + chunking)
    JSONLParser — .jsonl files (one JSON object per line)
    CSVParser   — .csv files (row-based chunking with header)
    PDFParser   — .pdf files (pure Python text extraction)

Each parser produces List[Chunk] ready for embedding and storage.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from zse_engine.rag.zpf_format import (
    compress_text, compress_json, compress_csv,
    ZPFDocument, ZPFSchema, _split_csv_line, _count_tokens,
)


@dataclass
class Chunk:
    """A document chunk ready for embedding."""
    text: str                          # Original text
    compressed: str                    # ZPF-compressed text
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_tokens: int = 0
    compressed_tokens: int = 0

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compressed_tokens / self.original_tokens) * 100


# ---------------------------------------------------------------------------
# Text Parser
# ---------------------------------------------------------------------------

class TextParser:
    """Parse plain text and markdown files into chunks.

    Splits on paragraph boundaries with configurable chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Args:
            chunk_size: Target chunk size in tokens.
            overlap: Overlap between chunks in tokens.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse text into chunks."""
        if not text.strip():
            return []

        base_meta = metadata or {}

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Group paragraphs into chunks that fit within chunk_size
        chunks = []
        current_paras = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para, tokenizer)

            # If a single paragraph exceeds chunk_size, split it by sentences
            if para_tokens > self.chunk_size:
                # Flush current buffer
                if current_paras:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_paras), tokenizer, base_meta, len(chunks)
                    ))
                    current_paras = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentence_chunks = self._split_by_sentences(para, tokenizer)
                for sc in sentence_chunks:
                    chunks.append(self._make_chunk(sc, tokenizer, base_meta, len(chunks)))
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > self.chunk_size and current_paras:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_paras), tokenizer, base_meta, len(chunks)
                ))
                # Overlap: keep last paragraph
                if self.overlap > 0 and current_paras:
                    last = current_paras[-1]
                    last_tokens = _count_tokens(last, tokenizer)
                    if last_tokens <= self.overlap:
                        current_paras = [last]
                        current_tokens = last_tokens
                    else:
                        current_paras = []
                        current_tokens = 0
                else:
                    current_paras = []
                    current_tokens = 0

            current_paras.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_paras:
            chunks.append(self._make_chunk(
                "\n\n".join(current_paras), tokenizer, base_meta, len(chunks)
            ))

        return chunks

    def _split_by_sentences(self, text: str, tokenizer=None) -> List[str]:
        """Split a large paragraph into sentence-based chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _count_tokens(sent, tokenizer)
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _make_chunk(
        self, text: str, tokenizer, base_meta: Dict, index: int,
    ) -> Chunk:
        zpf = compress_text(text, tokenizer)
        meta = {**base_meta, "chunk_index": index, "parser": "text"}
        return Chunk(
            text=text,
            compressed=zpf.compressed_text,
            metadata=meta,
            original_tokens=zpf.original_tokens,
            compressed_tokens=zpf.compressed_tokens,
        )


# ---------------------------------------------------------------------------
# JSON Parser
# ---------------------------------------------------------------------------

class JSONParser:
    """Parse JSON files into chunks with schema extraction.

    Handles:
    - Array of objects: schema-compressed, chunked by row groups
    - Single object: key:value compressed
    - Nested: flattened + compressed
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse JSON text into chunks."""
        base_meta = metadata or {}

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fall back to text parsing
            return TextParser(self.chunk_size).parse(text, tokenizer, metadata)

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return self._parse_array(data, text, tokenizer, base_meta)
        elif isinstance(data, dict):
            return self._parse_object(data, text, tokenizer, base_meta)
        else:
            return TextParser(self.chunk_size).parse(text, tokenizer, metadata)

    def _parse_array(
        self, data: List[dict], original: str, tokenizer, base_meta: Dict,
    ) -> List[Chunk]:
        """Parse array of objects — batch into chunks with overlap."""
        chunks = []

        # Determine rows per chunk
        sample_size = min(5, len(data))
        sample_zpf = compress_json(data[:sample_size], tokenizer)
        tokens_per_row = max(1, sample_zpf.compressed_tokens // sample_size)
        rows_per_chunk = max(1, self.chunk_size // tokens_per_row)

        # Overlap in rows
        overlap_rows = 0
        if self.overlap > 0:
            overlap_rows = max(1, self.overlap // tokens_per_row)

        # Extract schema
        all_keys = []
        seen = set()
        for obj in data:
            for k in obj.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # Chunk with overlap
        step = max(1, rows_per_chunk - overlap_rows)
        i = 0
        while i < len(data):
            batch = data[i:i + rows_per_chunk]
            batch_original = json.dumps(batch, ensure_ascii=False)
            zpf = compress_json(batch, tokenizer)

            meta = {
                **base_meta,
                "chunk_index": len(chunks),
                "parser": "json",
                "row_start": i,
                "row_end": i + len(batch),
                "schema": all_keys,
            }

            chunks.append(Chunk(
                text=batch_original,
                compressed=zpf.compressed_text,
                metadata=meta,
                original_tokens=zpf.original_tokens,
                compressed_tokens=zpf.compressed_tokens,
            ))

            i += step
            if i + overlap_rows >= len(data) and i < len(data):
                # Last chunk: take remaining
                if i >= len(data) - overlap_rows:
                    break

        return chunks

    def _parse_object(
        self, data: dict, original: str, tokenizer, base_meta: Dict,
    ) -> List[Chunk]:
        """Parse a single object."""
        zpf = compress_json(data, tokenizer)
        meta = {**base_meta, "chunk_index": 0, "parser": "json"}
        return [Chunk(
            text=original,
            compressed=zpf.compressed_text,
            metadata=meta,
            original_tokens=zpf.original_tokens,
            compressed_tokens=zpf.compressed_tokens,
        )]


# ---------------------------------------------------------------------------
# CSV Parser
# ---------------------------------------------------------------------------

class CSVParser:
    """Parse CSV files into chunks with schema header.

    Preserves the header row in each chunk for context.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse CSV text into chunks with overlap support."""
        base_meta = metadata or {}
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return TextParser(self.chunk_size).parse(text, tokenizer, metadata)

        header = lines[0].strip()
        data_lines = [l.strip() for l in lines[1:] if l.strip()]

        if not data_lines:
            return []

        # Estimate rows per chunk
        sample_text = header + "\n" + "\n".join(data_lines[:5])
        sample_zpf = compress_csv(sample_text, tokenizer)
        data_tokens = max(1, sample_zpf.compressed_tokens - _count_tokens(
            compress_csv(header + "\n" + data_lines[0], tokenizer).compressed_text[:50], tokenizer
        ))
        tokens_per_row = max(1, data_tokens // min(5, len(data_lines)))
        rows_per_chunk = max(1, self.chunk_size // tokens_per_row)

        # Overlap in rows
        overlap_rows = 0
        if self.overlap > 0:
            overlap_rows = max(1, self.overlap // tokens_per_row)

        # Build chunks with overlap
        chunks = []
        step = max(1, rows_per_chunk - overlap_rows)
        i = 0
        while i < len(data_lines):
            batch = data_lines[i:i + rows_per_chunk]
            batch_text = header + "\n" + "\n".join(batch)

            zpf = compress_csv(batch_text, tokenizer)
            meta = {
                **base_meta,
                "chunk_index": len(chunks),
                "parser": "csv",
                "row_start": i,
                "row_end": i + len(batch),
            }

            chunks.append(Chunk(
                text=batch_text,
                compressed=zpf.compressed_text,
                metadata=meta,
                original_tokens=zpf.original_tokens,
                compressed_tokens=zpf.compressed_tokens,
            ))

            i += step
            if i + overlap_rows >= len(data_lines) and i < len(data_lines):
                if i >= len(data_lines) - overlap_rows:
                    break

        return chunks


# ---------------------------------------------------------------------------
# JSONL Parser
# ---------------------------------------------------------------------------

class JSONLParser:
    """Parse JSONL files (one JSON object per line) into chunks.

    JSONL is common for datasets (e.g., HuggingFace, OpenAI fine-tuning).
    Each line is parsed independently, then batched into chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse JSONL text into chunks."""
        base_meta = metadata or {}

        # Parse each line as JSON
        objects = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        if not objects:
            return TextParser(self.chunk_size).parse(text, tokenizer, metadata)

        # Delegate to JSONParser's array handling
        json_parser = JSONParser(chunk_size=self.chunk_size, overlap=self.overlap)
        return json_parser._parse_array(objects, text, tokenizer, base_meta)


# ---------------------------------------------------------------------------
# PDF Parser
# ---------------------------------------------------------------------------

class PDFParser:
    """Pure Python PDF text extraction — zero dependencies.

    Handles:
    - Uncompressed text streams (BT...ET blocks)
    - FlateDecode (zlib) compressed streams
    - Tj/TJ text operators
    - Basic text positioning (Td/Tm for newlines)

    Limitations (intentional — zero deps):
    - No encrypted PDFs
    - No image-based PDFs (needs OCR)
    - No complex font encoding (CIDFont, ToUnicode beyond basic)
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse PDF content. Note: 'text' here is actually raw bytes decoded as latin-1."""
        base_meta = metadata or {}

        # Try to extract text from PDF
        try:
            extracted = self._extract_text(text.encode("latin-1"))
        except Exception:
            try:
                extracted = self._extract_text(text.encode("utf-8"))
            except Exception:
                extracted = ""

        if not extracted or len(extracted.strip()) < 10:
            # PDF might be image-based or encrypted
            return [Chunk(
                text="[PDF text extraction failed — document may be image-based or encrypted]",
                compressed="[PDF extraction failed]",
                metadata={**base_meta, "parser": "pdf", "error": "extraction_failed"},
                original_tokens=0,
                compressed_tokens=0,
            )]

        # Delegate extracted text to TextParser
        text_parser = TextParser(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks = text_parser.parse(extracted, tokenizer, {**base_meta, "parser": "pdf"})

        # Override parser metadata
        for c in chunks:
            c.metadata["parser"] = "pdf"

        return chunks

    def _extract_text(self, raw: bytes) -> str:
        """Extract text from PDF bytes."""
        import zlib

        # Find all stream objects
        text_parts = []

        # Find streams
        i = 0
        while True:
            stream_start = raw.find(b"stream\r\n", i)
            if stream_start == -1:
                stream_start = raw.find(b"stream\n", i)
            if stream_start == -1:
                break

            # Skip past "stream\r\n" or "stream\n"
            if raw[stream_start + 6:stream_start + 8] == b"\r\n":
                data_start = stream_start + 8
            else:
                data_start = stream_start + 7

            stream_end = raw.find(b"endstream", data_start)
            if stream_end == -1:
                break

            stream_data = raw[data_start:stream_end]

            # Check if FlateDecode (compressed)
            # Look back for the object dict
            obj_start = raw.rfind(b"<<", max(0, stream_start - 2000), stream_start)
            is_flate = False
            if obj_start != -1:
                header = raw[obj_start:stream_start]
                is_flate = b"FlateDecode" in header

            # Decompress if needed
            try:
                if is_flate:
                    stream_data = zlib.decompress(stream_data)
            except Exception:
                i = stream_end + 9
                continue

            # Extract text from stream content
            extracted = self._extract_text_from_stream(stream_data)
            if extracted:
                text_parts.append(extracted)

            i = stream_end + 9

        return "\n\n".join(text_parts)

    def _extract_text_from_stream(self, data: bytes) -> str:
        """Extract text from a PDF content stream (BT...ET blocks)."""
        try:
            text_content = data.decode("latin-1")
        except Exception:
            return ""

        parts = []
        in_bt = False
        lines = text_content.split("\n")

        for line in lines:
            line = line.strip()

            if line == "BT":
                in_bt = True
                continue
            elif line == "ET":
                in_bt = False
                continue

            if not in_bt:
                continue

            # Handle Tj operator (show string)
            if line.endswith("Tj"):
                text = self._extract_string(line[:-2].strip())
                if text:
                    parts.append(text)

            # Handle TJ operator (show array of strings)
            elif line.endswith("TJ"):
                text = self._extract_tj_array(line[:-2].strip())
                if text:
                    parts.append(text)

            # Handle ' operator (move to next line and show string)
            elif line.endswith("'"):
                text = self._extract_string(line[:-1].strip())
                if text:
                    parts.append("\n" + text)

            # Td/TD (text position — detect line breaks)
            elif "Td" in line or "TD" in line:
                try:
                    nums = line.replace("Td", "").replace("TD", "").strip().split()
                    if len(nums) >= 2:
                        y_offset = float(nums[1])
                        if abs(y_offset) > 1:
                            parts.append("\n")
                except (ValueError, IndexError):
                    pass

        return "".join(parts).strip()

    def _extract_string(self, s: str) -> str:
        """Extract text from a PDF string literal like (Hello World)."""
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1]
            # Unescape PDF string escapes
            inner = inner.replace("\\(", "(").replace("\\)", ")")
            inner = inner.replace("\\\\", "\\")
            inner = inner.replace("\\n", "\n").replace("\\r", "\r")
            inner = inner.replace("\\t", "\t")
            return inner
        elif s.startswith("<") and s.endswith(">"):
            # Hex string
            hex_str = s[1:-1].replace(" ", "")
            try:
                return bytes.fromhex(hex_str).decode("latin-1")
            except Exception:
                return ""
        return ""

    def _extract_tj_array(self, s: str) -> str:
        """Extract text from TJ array like [(Hello) -10 (World)]."""
        s = s.strip()
        if not s.startswith("[") or not s.endswith("]"):
            return ""
        inner = s[1:-1]
        parts = []
        i = 0
        while i < len(inner):
            if inner[i] == "(":
                # Find matching close paren
                depth = 1
                j = i + 1
                while j < len(inner) and depth > 0:
                    if inner[j] == "(" and inner[j-1] != "\\":
                        depth += 1
                    elif inner[j] == ")" and inner[j-1] != "\\":
                        depth -= 1
                    j += 1
                parts.append(self._extract_string(inner[i:j]))
                i = j
            elif inner[i] == "<":
                j = inner.find(">", i)
                if j != -1:
                    parts.append(self._extract_string(inner[i:j+1]))
                    i = j + 1
                else:
                    i += 1
            else:
                # Skip numbers (kerning values)
                i += 1
        return "".join(parts)


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

def get_parser(filename: str, chunk_size: int = 512, overlap: int = 64):
    """Get the appropriate parser for a file based on extension.

    Args:
        filename: Original filename (used for extension detection)
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        Parser instance with .parse(text, tokenizer, metadata) method
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "jsonl":
        return JSONLParser(chunk_size=chunk_size, overlap=overlap)
    elif ext == "json":
        return JSONParser(chunk_size=chunk_size, overlap=overlap)
    elif ext == "csv":
        return CSVParser(chunk_size=chunk_size, overlap=overlap)
    elif ext == "pdf":
        return PDFParser(chunk_size=chunk_size, overlap=overlap)
    else:
        # txt, md, and anything else → text parser
        return TextParser(chunk_size=chunk_size, overlap=overlap)
