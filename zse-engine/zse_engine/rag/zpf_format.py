"""ZPF (Zyora Parser Format) — Token-optimized document compression.

Three-layer compression for reducing LLM token usage:
1. Schema extraction — JSON/CSV keys defined once, values as delimited rows
2. Dictionary encoding — repeated strings mapped to short codes
3. Whitespace/format stripping — remove noise, normalize spacing

Typical token savings:
    JSON datasets:  60-70%
    CSV/tabular:    50-60%
    Text documents: 30-40%
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ZPFSchema:
    """Column/key schema for structured data."""
    columns: List[str]

    def header(self) -> str:
        """Schema header line."""
        return "§" + "|".join(self.columns) + "§"

    @staticmethod
    def parse_header(line: str) -> Optional["ZPFSchema"]:
        if line.startswith("§") and line.endswith("§"):
            cols = line[1:-1].split("|")
            return ZPFSchema(columns=cols)
        return None


@dataclass
class ZPFDictionary:
    """Dictionary for encoding repeated values to short codes."""
    # value -> code (e.g. "United States" -> "£US")
    encode_map: Dict[str, str] = field(default_factory=dict)
    decode_map: Dict[str, str] = field(default_factory=dict)

    def add(self, value: str, code: str):
        self.encode_map[value] = code
        self.decode_map[code] = value

    def encode(self, text: str) -> str:
        for value, code in self.encode_map.items():
            text = text.replace(value, code)
        return text

    def decode(self, text: str) -> str:
        for code, value in self.decode_map.items():
            text = text.replace(code, value)
        return text

    def header(self) -> str:
        """Dictionary definition lines."""
        lines = []
        for value, code in self.encode_map.items():
            lines.append(f"{code}={value}")
        return "\n".join(lines)


@dataclass
class ZPFDocument:
    """A ZPF-compressed document."""
    original_text: str
    compressed_text: str
    schema: Optional[ZPFSchema] = None
    dictionary: Optional[ZPFDictionary] = None
    original_tokens: int = 0
    compressed_tokens: int = 0
    doc_type: str = "text"

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1.0 - self.compressed_tokens / self.original_tokens) * 100


# ---------------------------------------------------------------------------
# Compression functions
# ---------------------------------------------------------------------------

def compress_text(text: str, tokenizer=None) -> ZPFDocument:
    """Compress plain text — strip noise, normalize whitespace."""
    original = text
    # Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace — collapse multiple spaces/newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()

    orig_tokens = _count_tokens(original, tokenizer)
    comp_tokens = _count_tokens(cleaned, tokenizer)

    return ZPFDocument(
        original_text=original,
        compressed_text=cleaned,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        doc_type="text",
    )


def compress_json(data: Any, tokenizer=None) -> ZPFDocument:
    """Compress JSON data using schema extraction + dictionary encoding."""
    original_text = json.dumps(data, ensure_ascii=False)
    orig_tokens = _count_tokens(original_text, tokenizer)

    # Handle list of objects (most common RAG case)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return _compress_json_array(data, original_text, orig_tokens, tokenizer)

    # Handle single object
    if isinstance(data, dict):
        return _compress_json_object(data, original_text, orig_tokens, tokenizer)

    # Fallback: just clean whitespace
    return compress_text(original_text, tokenizer)


def compress_csv(text: str, tokenizer=None) -> ZPFDocument:
    """Compress CSV data using schema extraction + dictionary encoding."""
    original = text
    orig_tokens = _count_tokens(original, tokenizer)

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return compress_text(text, tokenizer)

    # Parse header
    header = lines[0].strip()
    columns = _split_csv_line(header)
    schema = ZPFSchema(columns=columns)

    # Parse data rows
    rows = []
    all_values = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        values = _split_csv_line(line)
        rows.append(values)
        all_values.extend(values)

    # Build dictionary for repeated values
    dictionary = _build_dictionary(all_values)

    # Build compressed output
    parts = [schema.header()]
    if dictionary.encode_map:
        parts.append(dictionary.header())
        parts.append("---")

    for row in rows:
        encoded = []
        for v in row:
            encoded.append(dictionary.encode(v))
        parts.append("|".join(encoded))

    compressed = "\n".join(parts)
    comp_tokens = _count_tokens(compressed, tokenizer)

    return ZPFDocument(
        original_text=original,
        compressed_text=compressed,
        schema=schema,
        dictionary=dictionary if dictionary.encode_map else None,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        doc_type="csv",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_json_array(
    data: List[dict], original_text: str, orig_tokens: int, tokenizer=None,
) -> ZPFDocument:
    """Compress an array of JSON objects using schema extraction."""
    # Extract union of all keys (preserve order from first object)
    all_keys = []
    seen = set()
    for obj in data:
        for k in obj.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    schema = ZPFSchema(columns=all_keys)

    # Collect all values for dictionary building
    all_values = []
    for obj in data:
        for k in all_keys:
            v = obj.get(k, "")
            all_values.append(str(v))

    dictionary = _build_dictionary(all_values)

    # Build compressed output
    parts = [schema.header()]
    if dictionary.encode_map:
        parts.append(dictionary.header())
        parts.append("---")

    for obj in data:
        row = []
        for k in all_keys:
            v = str(obj.get(k, ""))
            row.append(dictionary.encode(v))
        parts.append("|".join(row))

    compressed = "\n".join(parts)
    comp_tokens = _count_tokens(compressed, tokenizer)

    return ZPFDocument(
        original_text=original_text,
        compressed_text=compressed,
        schema=schema,
        dictionary=dictionary if dictionary.encode_map else None,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        doc_type="json",
    )


def _compress_json_object(
    data: dict, original_text: str, orig_tokens: int, tokenizer=None,
) -> ZPFDocument:
    """Compress a single JSON object — key:value pairs, no quotes."""
    lines = []
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            # Nested: recurse to JSON but compact
            v_str = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
        else:
            v_str = str(v)
        lines.append(f"{k}:{v_str}")

    compressed = "\n".join(lines)
    comp_tokens = _count_tokens(compressed, tokenizer)

    return ZPFDocument(
        original_text=original_text,
        compressed_text=compressed,
        original_tokens=orig_tokens,
        compressed_tokens=comp_tokens,
        doc_type="json",
    )


def _build_dictionary(values: List[str], min_freq: int = 3, min_len: int = 4) -> ZPFDictionary:
    """Build a dictionary of frequently repeated values.

    Only encodes values that appear >= min_freq times and are >= min_len chars.
    Short code = £ + first 2 uppercase letters (dedup with counter).
    """
    freq = Counter(values)
    dictionary = ZPFDictionary()
    used_codes = set()
    code_counter = 0

    # Sort by frequency (most frequent first)
    for value, count in freq.most_common():
        if count < min_freq or len(value) < min_len:
            continue

        # Generate short code
        # Use first 2 letters uppercase, fallback to counter
        base = re.sub(r'[^a-zA-Z0-9]', '', value)[:2].upper()
        if not base:
            base = f"V{code_counter}"
        code = f"£{base}"
        while code in used_codes:
            code_counter += 1
            code = f"£{base}{code_counter}"

        used_codes.add(code)
        dictionary.add(value, code)

        # Don't create too many codes (diminishing returns)
        if len(dictionary.encode_map) >= 50:
            break

    return dictionary


def _split_csv_line(line: str) -> List[str]:
    """Split a CSV line handling quoted fields."""
    fields = []
    current = []
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == ',' and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    fields.append("".join(current).strip())
    return fields


def _count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens using the model's tokenizer, or estimate."""
    if tokenizer and hasattr(tokenizer, 'encode'):
        try:
            return len(tokenizer.encode(text, add_bos=False))
        except Exception:
            pass
    # Rough estimate: ~4 chars per token for English
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Decompression (for verification/debugging)
# ---------------------------------------------------------------------------

def decompress_zpf(compressed: str) -> str:
    """Decompress ZPF text back to readable format.

    Primarily for debugging — the compressed form is what goes to the LLM.
    """
    lines = compressed.split("\n")
    if not lines:
        return compressed

    # Check for schema header
    schema = ZPFSchema.parse_header(lines[0])
    if schema is None:
        return compressed  # Plain text, no decompression needed

    # Parse dictionary if present
    dictionary = ZPFDictionary()
    data_start = 1
    for i, line in enumerate(lines[1:], 1):
        if line == "---":
            data_start = i + 1
            break
        if "=" in line and line.startswith("£"):
            code, value = line.split("=", 1)
            dictionary.add(value, code)
        else:
            data_start = i
            break

    # Reconstruct JSON-like output
    objects = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        values = line.split("|")
        obj = {}
        for j, col in enumerate(schema.columns):
            v = values[j] if j < len(values) else ""
            obj[col] = dictionary.decode(v)
        objects.append(obj)

    if len(objects) == 1:
        return json.dumps(objects[0], indent=2, ensure_ascii=False)
    return json.dumps(objects, indent=2, ensure_ascii=False)
