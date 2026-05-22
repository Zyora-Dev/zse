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
from typing import List, Dict, Optional, Any, Tuple

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
    - No built-in OCR. Scanned / image-only PDFs return empty unless an
      ``ocr_fn`` callable is supplied (user-provided, brings its own backend
      such as pytesseract / EasyOCR / a cloud API). Signature:
      ``ocr_fn(image_bytes: bytes, image_format: str) -> str`` where
      ``image_format`` is one of ``"jpeg"``, ``"jp2"``, ``"ccitt"``,
      ``"jbig2"``, ``"raw"``.
    - Per-font CMap scoping not tracked (uses unified document-wide CMap)
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, ocr_fn=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.ocr_fn = ocr_fn

    def parse(self, text: str, tokenizer=None, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Parse PDF content. Note: 'text' here is actually raw bytes decoded as latin-1.

        If the PDF is encrypted with a non-empty user password, supply it via
        ``metadata['pdf_password']`` (str or bytes). Empty password is always
        tried first for permissions-locked PDFs.

        Image-only / scanned PDFs are handled if an OCR callable is supplied
        either on the parser (``PDFParser(ocr_fn=...)``) or per-call via
        ``metadata['ocr_fn']``. The hook is only invoked when normal text
        extraction yields nothing — text-based PDFs never pay the OCR cost.
        """
        from . import pdf_crypto

        base_meta = metadata or {}
        raw_bytes = text.encode("latin-1", errors="ignore")

        # Resolve user password from metadata.
        pw_raw = base_meta.get("pdf_password")
        if isinstance(pw_raw, str):
            user_password = pw_raw.encode("utf-8")
        elif isinstance(pw_raw, (bytes, bytearray)):
            user_password = bytes(pw_raw)
        else:
            user_password = None

        # Detect encryption up front so we can distinguish "image-only" from
        # "encrypted-and-we-couldn't-decrypt" in the failure message.
        enc_info = pdf_crypto.detect_encryption(raw_bytes)
        enc_supported = enc_info is None or enc_info.supported
        enc_decrypt_ok = True
        if enc_info is not None:
            # Try empty password first (covers permissions-locked PDFs);
            # fall back to user-supplied password if empty fails.
            enc_decrypt_ok = enc_info.derive_file_key(b"")
            if not enc_decrypt_ok and user_password is not None:
                enc_decrypt_ok = enc_info.derive_file_key(user_password)

        # Try to extract text from PDF
        try:
            extracted = self._extract_text(raw_bytes, password=user_password)
        except Exception:
            try:
                extracted = self._extract_text(
                    text.encode("utf-8", errors="ignore"),
                    password=user_password,
                )
            except Exception:
                extracted = ""

        if not extracted or len(extracted.strip()) < 10:
            # OCR fallback before we give up. The user supplies the backend
            # (pytesseract / EasyOCR / cloud); ZSE just hands them raw image
            # bytes from the PDF's image XObjects.
            ocr_fn = base_meta.get("ocr_fn") or self.ocr_fn
            if ocr_fn is not None and (enc_info is None or enc_decrypt_ok):
                try:
                    ocr_text = self._run_ocr(raw_bytes, ocr_fn)
                except Exception:
                    ocr_text = ""
                if ocr_text and len(ocr_text.strip()) >= 10:
                    text_parser = TextParser(
                        chunk_size=self.chunk_size, overlap=self.overlap
                    )
                    chunks = text_parser.parse(
                        ocr_text, tokenizer, {**base_meta, "parser": "pdf"}
                    )
                    for c in chunks:
                        c.metadata["parser"] = "pdf"
                        c.metadata["pdf_ocr"] = True
                        if enc_info is not None:
                            c.metadata["pdf_encrypted"] = True
                    return chunks

            if enc_info is not None and (not enc_supported or not enc_decrypt_ok):
                reason = (
                    "encrypted_unsupported"
                    if not enc_supported
                    else "encrypted_password_required"
                )
                msg = (
                    f"[PDF text extraction failed — encryption V={enc_info.version} "
                    f"R={enc_info.revision} ({reason})]"
                )
            else:
                msg = "[PDF text extraction failed — document may be image-based or encrypted]"
            return [Chunk(
                text=msg,
                compressed="[PDF extraction failed]",
                metadata={
                    **base_meta,
                    "parser": "pdf",
                    "error": (
                        "encrypted_unsupported" if (enc_info and not enc_supported)
                        else "encrypted_password_required" if (enc_info and not enc_decrypt_ok)
                        else "extraction_failed"
                    ),
                },
                original_tokens=0,
                compressed_tokens=0,
            )]

        # Delegate extracted text to TextParser
        text_parser = TextParser(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks = text_parser.parse(extracted, tokenizer, {**base_meta, "parser": "pdf"})

        # Override parser metadata
        for c in chunks:
            c.metadata["parser"] = "pdf"
            if enc_info is not None:
                c.metadata["pdf_encrypted"] = True

        return chunks

    def _extract_text(self, raw: bytes, password: Optional[bytes] = None) -> str:
        """Extract text from PDF bytes."""
        from . import pdf_crypto

        # Detect Standard Security Handler encryption. Empty user password is
        # tried automatically (covers permissions-locked PDFs). If that fails
        # and a non-empty password was supplied, try it as a fallback.
        encryption = pdf_crypto.detect_encryption(raw)
        if encryption is not None:
            ok = encryption.derive_file_key(b"")
            if not ok and password:
                ok = encryption.derive_file_key(password)
            if not ok:
                return ""
            obj_map = pdf_crypto.find_stream_object_numbers(raw)
        else:
            obj_map = {}

        # Build document-wide ToUnicode CMap (glyph code -> unicode string).
        cmap, code_len = self._build_cmap(raw, encryption=encryption, obj_map=obj_map)

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
            # PDF spec: an EOL marker (CRLF or LF) appears immediately before
            # 'endstream'. Stream-cipher (RC4) tolerated extra bytes, but AES
            # requires exact 16-byte alignment — strip up to two trailing EOL bytes.
            if stream_data.endswith(b"\r\n"):
                stream_data = stream_data[:-2]
            elif stream_data.endswith(b"\n") or stream_data.endswith(b"\r"):
                stream_data = stream_data[:-1]

            # Decrypt before applying the filter pipeline (PDF spec order).
            if encryption is not None:
                obj_info = obj_map.get(data_start)
                if obj_info is None:
                    i = stream_end + 9
                    continue
                try:
                    stream_data = encryption.decrypt_stream(
                        obj_info[0], obj_info[1], stream_data
                    )
                except Exception:
                    i = stream_end + 9
                    continue
                if not stream_data:
                    i = stream_end + 9
                    continue

            # Parse the preceding object dict to discover /Filter and stream type.
            obj_start = raw.rfind(b"<<", max(0, stream_start - 4000), stream_start)
            header = raw[obj_start:stream_start] if obj_start != -1 else b""
            is_cmap = (
                b"/ToUnicode" in header
                or b"CMapType" in header
                or b"/Subtype /Image" in header  # always skip images
                or b"/Subtype/Image" in header
            )

            # Apply the filter pipeline. Unsupported filters fail closed.
            decoded = self._apply_filters(stream_data, header)
            if decoded is None:
                i = stream_end + 9
                continue
            stream_data = decoded

            # Skip CMap streams — already consumed by _build_cmap.
            if is_cmap or b"beginbfchar" in stream_data or b"beginbfrange" in stream_data:
                i = stream_end + 9
                continue

            # Extract text from stream content
            extracted = self._extract_text_from_stream(stream_data, cmap, code_len)
            if extracted:
                text_parts.append(extracted)

            i = stream_end + 9

        return "\n\n".join(text_parts)

    # ------------------------------------------------------------------
    # OCR fallback
    # ------------------------------------------------------------------

    # Image filter -> external format hint passed to the user's OCR backend.
    _IMAGE_FORMAT_HINTS = {
        "DCTDecode": "jpeg", "DCT": "jpeg",
        "JPXDecode": "jp2",
        "CCITTFaxDecode": "ccitt", "CCF": "ccitt",
        "JBIG2Decode": "jbig2",
    }

    def _extract_images(self, raw: bytes) -> List[Tuple[bytes, str]]:
        """Collect image XObject streams as (bytes, format_hint) tuples.

        Returns the encoded image bytes exactly as stored in the PDF
        (e.g. raw JPEG for /DCTDecode). The caller's OCR backend is
        expected to decode them. Decryption is NOT performed here — OCR
        fallback only activates when text extraction succeeded enough to
        confirm the file is readable but produced no usable text.
        """
        from . import pdf_crypto

        encryption = pdf_crypto.detect_encryption(raw)
        if encryption is not None and not encryption.derive_file_key(b""):
            # Encrypted with non-empty password: we already reported it
            # upstream; don't leak ciphertext to the OCR backend.
            return []
        obj_map = (
            pdf_crypto.find_stream_object_numbers(raw) if encryption else {}
        )

        images: List[Tuple[bytes, str]] = []
        i = 0
        while True:
            stream_start = raw.find(b"stream\r\n", i)
            if stream_start == -1:
                stream_start = raw.find(b"stream\n", i)
            if stream_start == -1:
                break

            if raw[stream_start + 6:stream_start + 8] == b"\r\n":
                data_start = stream_start + 8
            else:
                data_start = stream_start + 7

            stream_end = raw.find(b"endstream", data_start)
            if stream_end == -1:
                break

            stream_data = raw[data_start:stream_end]
            if stream_data.endswith(b"\r\n"):
                stream_data = stream_data[:-2]
            elif stream_data.endswith(b"\n") or stream_data.endswith(b"\r"):
                stream_data = stream_data[:-1]

            obj_start = raw.rfind(b"<<", max(0, stream_start - 4000), stream_start)
            header = raw[obj_start:stream_start] if obj_start != -1 else b""

            is_image = (
                b"/Subtype /Image" in header
                or b"/Subtype/Image" in header
            )
            if not is_image:
                i = stream_end + 9
                continue

            if encryption is not None:
                obj_info = obj_map.get(data_start)
                if obj_info is None:
                    i = stream_end + 9
                    continue
                try:
                    stream_data = encryption.decrypt_stream(
                        obj_info[0], obj_info[1], stream_data
                    )
                except Exception:
                    i = stream_end + 9
                    continue
                if not stream_data:
                    i = stream_end + 9
                    continue

            # Pick a format hint from the filter chain.
            hint = "raw"
            for name, h in self._IMAGE_FORMAT_HINTS.items():
                if b"/" + name.encode("ascii") in header:
                    hint = h
                    break
            images.append((stream_data, hint))

            i = stream_end + 9

        return images

    def _run_ocr(self, raw: bytes, ocr_fn) -> str:
        """Run user-supplied OCR callable against every image in the PDF.

        Errors from the callback are swallowed per-image so a single bad
        image doesn't kill the whole document.
        """
        out: List[str] = []
        for image_bytes, hint in self._extract_images(raw):
            try:
                text = ocr_fn(image_bytes, hint)
            except TypeError:
                # Backward-compat: allow single-arg callables.
                try:
                    text = ocr_fn(image_bytes)
                except Exception:
                    text = ""
            except Exception:
                text = ""
            if text:
                out.append(text)
        return "\n\n".join(out).strip()

    # ------------------------------------------------------------------
    # Stream filter pipeline
    # ------------------------------------------------------------------

    # Map of filter name -> decoder method name.
    _FILTERS = {
        "FlateDecode": "_decode_flate",
        "Fl": "_decode_flate",
        "ASCIIHexDecode": "_decode_ascii_hex",
        "AHx": "_decode_ascii_hex",
        "ASCII85Decode": "_decode_ascii85",
        "A85": "_decode_ascii85",
        "RunLengthDecode": "_decode_run_length",
        "RL": "_decode_run_length",
        "LZWDecode": "_decode_lzw",
        "LZW": "_decode_lzw",
    }

    # Image / opaque filters we recognize but cannot decode to text. Streams
    # using these are skipped (return None) so we don't feed garbage to the
    # text extractor.
    _IMAGE_FILTERS = {
        "DCTDecode", "DCT",
        "JPXDecode",
        "CCITTFaxDecode", "CCF",
        "JBIG2Decode",
        "Crypt",
    }

    @classmethod
    def _parse_filter_chain(cls, header: bytes) -> Optional[list]:
        """Parse the /Filter entry from a stream's object dict.

        Returns the ordered list of filter names, or [] if no /Filter present.
        Returns None if the chain includes an image/opaque filter (caller
        should skip the stream).
        """
        import re

        # /Filter /Name  or  /Filter [/Name1 /Name2]
        m = re.search(rb"/Filter\s*(\[[^\]]*\]|/\w+)", header)
        if not m:
            return []
        spec = m.group(1)
        names = [n.decode("ascii", errors="ignore") for n in re.findall(rb"/(\w+)", spec)]
        for n in names:
            if n in cls._IMAGE_FILTERS:
                return None
        return names

    def _apply_filters(self, data: bytes, header: bytes) -> Optional[bytes]:
        """Run `data` through the /Filter chain declared in `header`.

        Returns None if a filter is unsupported or decoding fails — caller
        should skip the stream rather than process garbage.
        """
        chain = self._parse_filter_chain(header)
        if chain is None:
            return None
        for name in chain:
            decoder_name = self._FILTERS.get(name)
            if decoder_name is None:
                # Unknown filter — fail closed.
                return None
            try:
                data = getattr(self, decoder_name)(data)
            except Exception:
                return None
            if data is None:
                return None
        return data

    @staticmethod
    def _decode_flate(data: bytes) -> bytes:
        import zlib
        return zlib.decompress(data)

    @staticmethod
    def _decode_ascii_hex(data: bytes) -> bytes:
        """PDF ASCIIHexDecode: hex digits terminated by '>'. Whitespace ignored.

        An odd final digit is treated as if followed by '0'.
        """
        cleaned = bytearray()
        for b in data:
            c = chr(b)
            if c in "0123456789abcdefABCDEF":
                cleaned.append(b)
            elif c == ">":
                break
            # else: whitespace / other -> ignored
        if len(cleaned) % 2:
            cleaned.append(ord("0"))
        return bytes(bytearray.fromhex(cleaned.decode("ascii")))

    @staticmethod
    def _decode_ascii85(data: bytes) -> bytes:
        """PDF ASCII85Decode (Adobe variant): base-85 in '!'..'u' + 'z' shortcut.

        Stream ends at '~>'. Whitespace is ignored.
        """
        # Strip optional <~ prefix and ~> suffix; ignore whitespace.
        s = bytes(b for b in data if not chr(b).isspace())
        if s.startswith(b"<~"):
            s = s[2:]
        end = s.find(b"~>")
        if end != -1:
            s = s[:end]

        out = bytearray()
        group = []
        for b in s:
            c = chr(b)
            if c == "z":
                if group:
                    raise ValueError("'z' inside ASCII85 group")
                out.extend(b"\x00\x00\x00\x00")
                continue
            if not ("!" <= c <= "u"):
                raise ValueError(f"invalid ASCII85 char: {c!r}")
            group.append(b - 33)
            if len(group) == 5:
                acc = 0
                for v in group:
                    acc = acc * 85 + v
                out.extend(acc.to_bytes(4, "big"))
                group = []
        if group:
            n = len(group)
            # Pad group to 5 with 'u' (84), then drop (5 - n) trailing bytes.
            padded = group + [84] * (5 - n)
            acc = 0
            for v in padded:
                acc = acc * 85 + v
            out.extend(acc.to_bytes(4, "big")[: n - 1])
        return bytes(out)

    @staticmethod
    def _decode_run_length(data: bytes) -> bytes:
        """PDF RunLengthDecode: each length byte controls the next run.

        0..127  -> copy next length+1 bytes literally
        129..255 -> repeat next byte 257-length times
        128      -> EOD
        """
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            length = data[i]
            i += 1
            if length == 128:
                break
            if length < 128:
                count = length + 1
                out.extend(data[i:i + count])
                i += count
            else:
                count = 257 - length
                if i >= n:
                    break
                out.extend(bytes([data[i]]) * count)
                i += 1
        return bytes(out)

    @staticmethod
    def _decode_lzw(data: bytes) -> bytes:
        """PDF LZWDecode (variable-width codes, 9..12 bits).

        Code 256 = clear table, 257 = EOD. Code width starts at 9 bits and
        grows when the next code to be added would exceed 2^width - 1, with
        the standard PDF off-by-one (grow at 510, 1022, 2046).
        """
        clear_code = 256
        eod_code = 257
        # Initial table: 256 single-byte strings.
        table = {i: bytes([i]) for i in range(256)}
        next_code = 258
        code_width = 9

        out = bytearray()
        prev = None
        bit_buf = 0
        bit_count = 0
        idx = 0
        n = len(data)
        while True:
            # Read code_width bits MSB-first.
            while bit_count < code_width and idx < n:
                bit_buf = (bit_buf << 8) | data[idx]
                bit_count += 8
                idx += 1
            if bit_count < code_width:
                break
            shift = bit_count - code_width
            code = (bit_buf >> shift) & ((1 << code_width) - 1)
            bit_buf &= (1 << shift) - 1
            bit_count = shift

            if code == eod_code:
                break
            if code == clear_code:
                table = {i: bytes([i]) for i in range(256)}
                next_code = 258
                code_width = 9
                prev = None
                continue

            if code in table:
                entry = table[code]
            elif code == next_code and prev is not None:
                entry = prev + prev[:1]
            else:
                raise ValueError(f"invalid LZW code: {code}")

            out.extend(entry)

            if prev is not None:
                table[next_code] = prev + entry[:1]
                next_code += 1
                # Grow code width with PDF's early-change behavior.
                if next_code == (1 << code_width) - 1 and code_width < 12:
                    code_width += 1
            prev = entry
        return bytes(out)


    # ------------------------------------------------------------------
    # ToUnicode CMap parsing
    # ------------------------------------------------------------------

    def _build_cmap(self, raw: bytes, encryption=None, obj_map=None):
        """Scan every stream in the PDF; merge all /ToUnicode CMaps into a
        single glyph-code -> unicode string map.

        Returns (cmap, code_len). code_len is the byte width of source codes
        (1 or 2), inferred from the widest hex literal seen in bfchar/bfrange
        entries. Per-font scoping is not tracked — in practice most PDFs
        assign distinct glyph IDs across fonts, so a unified map covers the
        vast majority of real-world documents.

        If `encryption` is set, each stream is decrypted with its per-object
        key before filter decoding (obj_map: data_start -> (obj_num, gen_num)).
        """
        cmap: Dict[int, str] = {}
        max_hex_width = 0
        i = 0
        while True:
            s = raw.find(b"stream\r\n", i)
            if s == -1:
                s = raw.find(b"stream\n", i)
            if s == -1:
                break
            ds = s + 8 if raw[s + 6:s + 8] == b"\r\n" else s + 7
            e = raw.find(b"endstream", ds)
            if e == -1:
                break
            body = raw[ds:e]
            # Strip trailing EOL (required for AES alignment).
            if body.endswith(b"\r\n"):
                body = body[:-2]
            elif body.endswith(b"\n") or body.endswith(b"\r"):
                body = body[:-1]
            if encryption is not None and obj_map is not None:
                info = obj_map.get(ds)
                if info is None:
                    i = e + 9
                    continue
                try:
                    body = encryption.decrypt_stream(info[0], info[1], body)
                except Exception:
                    i = e + 9
                    continue
                if not body:
                    i = e + 9
                    continue
            hdr = raw[max(0, s - 4000):s]
            decoded = self._apply_filters(body, hdr)
            if decoded is None:
                i = e + 9
                continue
            body = decoded
            if b"beginbfchar" in body or b"beginbfrange" in body:
                w = self._merge_cmap(body, cmap)
                if w > max_hex_width:
                    max_hex_width = w
            i = e + 9

        # Source hex width → byte code length. 4 hex chars = 2 bytes, etc.
        # Cap at 2 — PDF spec allows up to 4 but >2 is exceedingly rare and
        # would require codespace-range tracking we don't currently model.
        if max_hex_width >= 4:
            code_len = 2
        else:
            code_len = 1
        return cmap, code_len

    def _merge_cmap(self, body: bytes, cmap: Dict[int, str]) -> int:
        """Parse one CMap stream and merge bfchar/bfrange entries into `cmap`.

        Returns the maximum source-hex-literal width seen (in hex chars).
        """
        import re

        try:
            txt = body.decode("latin-1", errors="ignore")
        except Exception:
            return 0

        max_width = 0

        # bfchar: <srcCode> <utf16BE>
        for m in re.finditer(r"beginbfchar(.*?)endbfchar", txt, re.DOTALL):
            block = m.group(1)
            for hm in re.finditer(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                src_hex = hm.group(1)
                if len(src_hex) > max_width:
                    max_width = len(src_hex)
                try:
                    code = int(src_hex, 16)
                    val = bytes.fromhex(hm.group(2)).decode("utf-16-be", errors="ignore")
                except Exception:
                    continue
                if val:
                    cmap[code] = val

        # bfrange has two forms; handle array form first, then start form.
        for m in re.finditer(r"beginbfrange(.*?)endbfrange", txt, re.DOTALL):
            block = m.group(1)
            consumed_spans = []
            # Array form: <lo> <hi> [<v1> <v2> ...]
            for hm in re.finditer(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[([^\]]*)\]", block
            ):
                consumed_spans.append((hm.start(), hm.end()))
                src_hex = hm.group(1)
                if len(src_hex) > max_width:
                    max_width = len(src_hex)
                try:
                    lo = int(src_hex, 16)
                    hi = int(hm.group(2), 16)
                except Exception:
                    continue
                arr = re.findall(r"<([0-9A-Fa-f]+)>", hm.group(3))
                for offset, hexval in enumerate(arr):
                    if lo + offset > hi:
                        break
                    try:
                        cmap[lo + offset] = bytes.fromhex(hexval).decode("utf-16-be", errors="ignore")
                    except Exception:
                        pass
            # Start form: <lo> <hi> <start> — skip ranges already consumed by array form.
            for hm in re.finditer(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block
            ):
                if any(a <= hm.start() < b for a, b in consumed_spans):
                    continue
                src_hex = hm.group(1)
                if len(src_hex) > max_width:
                    max_width = len(src_hex)
                try:
                    lo = int(src_hex, 16)
                    hi = int(hm.group(2), 16)
                    start = bytes.fromhex(hm.group(3)).decode("utf-16-be", errors="ignore")
                except Exception:
                    continue
                if not start:
                    continue
                base_cp = ord(start[-1])
                prefix = start[:-1]
                for offset in range(hi - lo + 1):
                    if lo + offset in cmap:
                        continue
                    try:
                        cmap[lo + offset] = prefix + chr(base_cp + offset)
                    except (ValueError, OverflowError):
                        pass

        return max_width

    @staticmethod
    def _apply_cmap_to_bytes(data: bytes, cmap: Dict[int, str], code_len: int) -> str:
        """Map a raw byte sequence through `cmap` using fixed-width codes.

        Bytes that don't resolve fall back to latin-1 decoding so we never lose
        plain ASCII passages that aren't in the CMap.
        """
        out = []
        n = len(data)
        i = 0
        while i + code_len <= n:
            code = data[i] if code_len == 1 else (data[i] << 8) | data[i + 1]
            mapped = cmap.get(code)
            if mapped is not None:
                out.append(mapped)
            else:
                out.append(data[i:i + code_len].decode("latin-1", errors="ignore"))
            i += code_len
        if i < n:
            out.append(data[i:].decode("latin-1", errors="ignore"))
        return "".join(out)

    def _extract_text_from_stream(
        self,
        data: bytes,
        cmap: Optional[Dict[int, str]] = None,
        code_len: int = 1,
    ) -> str:
        """Extract text from a PDF content stream with positional reflow.

        Tracks the text matrix translation (x, y) across BT/ET blocks via the
        Tm/Td/TD/T*/'/'' operators so that fragments can be grouped into lines
        and columns instead of being emitted in raw draw order. This fixes the
        common multi-column / interleaved-table failure mode where the line
        based extractor produced shuffled chunks.
        """
        try:
            text_content = data.decode("latin-1")
        except Exception:
            return ""

        try:
            tokens = self._tokenize_content_stream(text_content)
        except Exception:
            return ""

        fragments: List[tuple] = []  # (x, y, text)

        in_text = False
        tm_x = tm_y = 0.0           # current text matrix translation
        line_x = line_y = 0.0       # line matrix (origin for T*, ', ")
        leading = 0.0
        operand_stack: List[str] = []

        for tok in tokens:
            if not tok:
                continue
            c0 = tok[0]
            is_operand = c0 in "(<[/+-." or c0.isdigit()
            if is_operand:
                operand_stack.append(tok)
                continue

            op = tok
            if op == "BT":
                in_text = True
                tm_x = tm_y = line_x = line_y = 0.0
            elif op == "ET":
                in_text = False
            elif not in_text:
                pass
            elif op == "Tm" and len(operand_stack) >= 6:
                try:
                    tm_x = float(operand_stack[-2])
                    tm_y = float(operand_stack[-1])
                    line_x, line_y = tm_x, tm_y
                except ValueError:
                    pass
            elif op == "Td" and len(operand_stack) >= 2:
                try:
                    line_x += float(operand_stack[-2])
                    line_y += float(operand_stack[-1])
                    tm_x, tm_y = line_x, line_y
                except ValueError:
                    pass
            elif op == "TD" and len(operand_stack) >= 2:
                try:
                    tx = float(operand_stack[-2])
                    ty = float(operand_stack[-1])
                    leading = -ty
                    line_x += tx
                    line_y += ty
                    tm_x, tm_y = line_x, line_y
                except ValueError:
                    pass
            elif op == "TL" and operand_stack:
                try:
                    leading = float(operand_stack[-1])
                except ValueError:
                    pass
            elif op == "T*":
                line_y -= leading
                tm_x, tm_y = line_x, line_y
            elif op == "Tj" and operand_stack:
                text = self._extract_string(operand_stack[-1], cmap, code_len)
                if text:
                    fragments.append((tm_x, tm_y, text))
            elif op == "TJ" and operand_stack:
                text = self._extract_tj_array(operand_stack[-1], cmap, code_len)
                if text:
                    fragments.append((tm_x, tm_y, text))
            elif op == "'" and operand_stack:
                line_y -= leading
                tm_x, tm_y = line_x, line_y
                text = self._extract_string(operand_stack[-1], cmap, code_len)
                if text:
                    fragments.append((tm_x, tm_y, text))
            elif op == '"' and len(operand_stack) >= 3:
                line_y -= leading
                tm_x, tm_y = line_x, line_y
                text = self._extract_string(operand_stack[-1], cmap, code_len)
                if text:
                    fragments.append((tm_x, tm_y, text))

            operand_stack = []

        if not fragments:
            return ""

        return self._reflow_fragments(fragments)

    # ------------------------------------------------------------------
    # Content-stream tokenizer + positional reflow
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_content_stream(text: str) -> List[str]:
        """Whitespace-tokenize a PDF content stream, keeping atomic structures.

        Literal strings `(...)` (with balanced parens + `\\` escapes), hex
        strings `<...>`, dicts `<<...>>`, arrays `[...]`, names `/Name`,
        numbers and operators are each returned as a single token. Comments
        (`%...EOL`) are skipped.
        """
        tokens: List[str] = []
        n = len(text)
        i = 0
        ws = " \t\n\r\f\x00"
        delims = "()<>[]{}/%"
        while i < n:
            c = text[i]
            if c in ws:
                i += 1
            elif c == "%":
                while i < n and text[i] not in "\n\r":
                    i += 1
            elif c == "(":
                start = i
                depth = 1
                i += 1
                while i < n and depth > 0:
                    ch = text[i]
                    if ch == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                tokens.append(text[start:i])
            elif c == "<":
                if i + 1 < n and text[i + 1] == "<":
                    start = i
                    depth = 0
                    while i < n:
                        if text[i:i + 2] == "<<":
                            depth += 1
                            i += 2
                        elif text[i:i + 2] == ">>":
                            depth -= 1
                            i += 2
                            if depth == 0:
                                break
                        else:
                            i += 1
                    tokens.append(text[start:i])
                else:
                    start = i
                    i += 1
                    while i < n and text[i] != ">":
                        i += 1
                    if i < n:
                        i += 1
                    tokens.append(text[start:i])
            elif c == "[":
                start = i
                depth = 1
                i += 1
                while i < n and depth > 0:
                    ch = text[i]
                    if ch == "(":
                        sub_depth = 1
                        i += 1
                        while i < n and sub_depth > 0:
                            if text[i] == "\\" and i + 1 < n:
                                i += 2
                                continue
                            if text[i] == "(":
                                sub_depth += 1
                            elif text[i] == ")":
                                sub_depth -= 1
                            i += 1
                    elif ch == "<" and i + 1 < n and text[i + 1] != "<":
                        while i < n and text[i] != ">":
                            i += 1
                        if i < n:
                            i += 1
                    elif ch == "[":
                        depth += 1
                        i += 1
                    elif ch == "]":
                        depth -= 1
                        i += 1
                    else:
                        i += 1
                tokens.append(text[start:i])
            elif c == "/":
                start = i
                i += 1
                while i < n and text[i] not in ws and text[i] not in delims:
                    i += 1
                tokens.append(text[start:i])
            else:
                start = i
                while i < n and text[i] not in ws and text[i] not in delims:
                    i += 1
                if i == start:
                    # Lone delimiter we didn't otherwise consume; skip it.
                    i += 1
                else:
                    tokens.append(text[start:i])
        return tokens

    @staticmethod
    def _reflow_fragments(fragments: List[tuple]) -> str:
        """Group (x, y, text) fragments into reading-order lines and columns.

        1. Detect column split from the distribution of fragment x-positions
           (must happen before line-grouping — multi-column lines share y).
        2. Partition fragments into columns.
        3. Within each column: sort top-to-bottom, cluster by y-tolerance into
           lines, sort each line left-to-right.
        4. Emit column-by-column, lines joined with '\\n', columns with '\\n\\n'.
        """
        if not fragments:
            return ""

        # --- Column detection on fragment x-positions ---
        xs = sorted({round(f[0], 1) for f in fragments})
        split_at: Optional[float] = None
        if len(fragments) >= 6 and len(xs) >= 2:
            max_gap = 0.0
            best_pos: Optional[float] = None
            for a, b in zip(xs, xs[1:]):
                gap = b - a
                if gap > max_gap:
                    max_gap = gap
                    best_pos = (a + b) / 2.0
            page_span = xs[-1] - xs[0]
            if (
                best_pos is not None
                and max_gap >= 80.0
                and page_span > 0
                and max_gap / page_span >= 0.25
            ):
                left_count = sum(1 for f in fragments if f[0] < best_pos)
                right_count = len(fragments) - left_count
                # Require both columns to be non-trivial.
                if left_count >= 3 and right_count >= 3:
                    split_at = best_pos

        if split_at is None:
            columns = [fragments]
        else:
            columns = [
                [f for f in fragments if f[0] < split_at],
                [f for f in fragments if f[0] >= split_at],
            ]

        # --- Per-column: sort, cluster into lines, stitch ---
        Y_TOL = 3.0
        column_outputs: List[str] = []
        for col_frags in columns:
            col_frags = sorted(col_frags, key=lambda f: (-f[1], f[0]))
            lines: List[tuple] = []
            cur: List[tuple] = []
            cur_y: Optional[float] = None
            for x, y, text in col_frags:
                if cur_y is None or abs(y - cur_y) <= Y_TOL:
                    cur.append((x, text))
                    if cur_y is None:
                        cur_y = y
                else:
                    cur.sort(key=lambda t: t[0])
                    lines.append((cur_y, cur))
                    cur = [(x, text)]
                    cur_y = y
            if cur and cur_y is not None:
                cur.sort(key=lambda t: t[0])
                lines.append((cur_y, cur))

            out_lines: List[str] = []
            for _y, line in lines:
                buf: List[str] = []
                prev_end_x: Optional[float] = None
                for x, text in line:
                    if not text:
                        continue
                    if prev_end_x is not None and x - prev_end_x > 2.0:
                        if buf and not buf[-1].endswith((" ", "\n")) and not text.startswith(" "):
                            buf.append(" ")
                    buf.append(text)
                    prev_end_x = x + max(1.0, len(text) * 4.5)
                line_text = "".join(buf).rstrip()
                if line_text:
                    out_lines.append(line_text)

            if out_lines:
                column_outputs.append("\n".join(out_lines))

        return "\n\n".join(column_outputs).strip()

    def _extract_string(
        self,
        s: str,
        cmap: Optional[Dict[int, str]] = None,
        code_len: int = 1,
    ) -> str:
        """Extract text from a PDF string literal like (Hello World).

        If a CMap is provided, raw bytes are mapped through it as fixed-width
        glyph codes; otherwise the bytes are decoded as latin-1.
        """
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1]
            # Unescape PDF string escapes (operate on the latin-1 byte view).
            inner = inner.replace("\\(", "(").replace("\\)", ")")
            inner = inner.replace("\\\\", "\\")
            inner = inner.replace("\\n", "\n").replace("\\r", "\r")
            inner = inner.replace("\\t", "\t")
            if cmap:
                return self._apply_cmap_to_bytes(
                    inner.encode("latin-1", errors="ignore"), cmap, code_len
                )
            return inner
        elif s.startswith("<") and s.endswith(">"):
            # Hex string
            hex_str = s[1:-1].replace(" ", "")
            # Pad odd-length hex per PDF spec (trailing 0).
            if len(hex_str) % 2:
                hex_str += "0"
            try:
                raw_bytes = bytes.fromhex(hex_str)
            except Exception:
                return ""
            if cmap:
                return self._apply_cmap_to_bytes(raw_bytes, cmap, code_len)
            return raw_bytes.decode("latin-1", errors="ignore")
        return ""

    def _extract_tj_array(
        self,
        s: str,
        cmap: Optional[Dict[int, str]] = None,
        code_len: int = 1,
    ) -> str:
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
                parts.append(self._extract_string(inner[i:j], cmap, code_len))
                i = j
            elif inner[i] == "<":
                j = inner.find(">", i)
                if j != -1:
                    parts.append(self._extract_string(inner[i:j+1], cmap, code_len))
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
