"""Tests for PDF stream-filter pipeline.

Verifies ASCIIHexDecode, ASCII85Decode, RunLengthDecode, LZWDecode, and
chained filter application (e.g. ASCII85+Flate). Also verifies that streams
with unknown or image filters are skipped rather than feeding garbage to the
text extractor.
"""

import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag.parser import PDFParser  # noqa: E402


def _wrap_pdf(stream_body: bytes, dict_entries: bytes) -> bytes:
    """Build a minimal PDF blob containing one stream with the given /Filter."""
    stream = (
        b"<< /Length " + str(len(stream_body)).encode("ascii")
        + dict_entries + b" >>\n"
        b"stream\n" + stream_body + b"\nendstream\n"
    )
    return b"%PDF-1.4\n" + stream + b"%%EOF\n"


# ---------------------------------------------------------------------------
# Individual decoders
# ---------------------------------------------------------------------------

class TestAsciiHexDecoder:
    def test_basic_round_trip(self):
        # "Hello" -> 48656C6C6F
        data = b"48656C6C6F>"
        assert PDFParser._decode_ascii_hex(data) == b"Hello"

    def test_whitespace_ignored(self):
        data = b"48 65\n6C\t6C 6F >"
        assert PDFParser._decode_ascii_hex(data) == b"Hello"

    def test_odd_final_digit_padded_with_zero(self):
        # 'F' -> 'F0' -> 0xF0
        assert PDFParser._decode_ascii_hex(b"F>") == b"\xF0"


class TestAscii85Decoder:
    def test_basic(self):
        # 'Man ' -> '9jqo^' in standard ASCII85 (per Adobe example)
        assert PDFParser._decode_ascii85(b"9jqo^~>") == b"Man "

    def test_z_shortcut_for_zero_word(self):
        # 'z' = four zero bytes
        assert PDFParser._decode_ascii85(b"z~>") == b"\x00\x00\x00\x00"

    def test_partial_group(self):
        # 'M' (1 byte) -> '9`' (2 chars) in ASCII85.
        assert PDFParser._decode_ascii85(b"9`~>") == b"M"

    def test_whitespace_and_eod(self):
        data = b"9jq\no^\n~>trailing"
        assert PDFParser._decode_ascii85(data) == b"Man "

    def test_with_prefix(self):
        assert PDFParser._decode_ascii85(b"<~9jqo^~>") == b"Man "


class TestRunLengthDecoder:
    def test_literal_run(self):
        # length=4 (-> copy 5 bytes literally)
        data = bytes([4]) + b"Hello"
        assert PDFParser._decode_run_length(data) == b"Hello"

    def test_repeat_run(self):
        # length=254 -> repeat next byte 257-254=3 times
        data = bytes([254, ord("A")])
        assert PDFParser._decode_run_length(data) == b"AAA"

    def test_eod_marker(self):
        data = bytes([4]) + b"Hello" + bytes([128]) + b"trailing"
        assert PDFParser._decode_run_length(data) == b"Hello"

    def test_mixed(self):
        data = bytes([2]) + b"XYZ" + bytes([253]) + b"!" + bytes([128])
        assert PDFParser._decode_run_length(data) == b"XYZ!!!!"


class TestLzwDecoder:
    def test_round_trip_via_flate_substitution(self):
        # LZW is hard to hand-roll; instead verify decoder by encoding a known
        # short payload and checking it round-trips. We use a tiny payload
        # built from a hand-traced PDF LZW example.
        # "-----A---B" encodes to bytes [0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01]
        # in PDF LZW (9-bit codes). Reference: PDF 1.7 spec, section 7.4.4.4 example.
        encoded = bytes([0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01])
        assert PDFParser._decode_lzw(encoded) == b"-----A---B"

    def test_handles_clear_code(self):
        # Clear code (256) at start resets table; this just confirms no crash.
        # Build minimal stream: clear (256, 9 bits) + literal 'A' (65) + EOD (257)
        # Bits MSB-first: 100000000 001000001 100000001 -> pad to byte boundary.
        bits = "100000000" + "001000001" + "100000001"
        # pad to multiple of 8
        bits += "0" * ((8 - len(bits) % 8) % 8)
        raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
        out = PDFParser._decode_lzw(raw)
        assert out == b"A"


# ---------------------------------------------------------------------------
# Filter chain wiring (through _extract_text)
# ---------------------------------------------------------------------------

class TestFilterChain:
    def test_flate_only(self):
        body = b"BT\n(Hello Flate) Tj\nET\n"
        pdf = _wrap_pdf(zlib.compress(body), b" /Filter /FlateDecode")
        text = PDFParser()._extract_text(pdf)
        assert "Hello Flate" in text

    def test_ascii_hex_only(self):
        body = b"BT\n(HiHex) Tj\nET\n"
        encoded = body.hex().upper().encode("ascii") + b">"
        pdf = _wrap_pdf(encoded, b" /Filter /ASCIIHexDecode")
        text = PDFParser()._extract_text(pdf)
        assert "HiHex" in text

    def test_ascii85_then_flate_chain(self):
        # /Filter [/ASCII85Decode /FlateDecode] — applied left-to-right.
        body = b"BT\n(Chained) Tj\nET\n"
        flated = zlib.compress(body)
        # ascii85-encode the flated bytes
        a85 = _ascii85_encode(flated) + b"~>"
        pdf = _wrap_pdf(a85, b" /Filter [/ASCII85Decode /FlateDecode]")
        text = PDFParser()._extract_text(pdf)
        assert "Chained" in text

    def test_unknown_filter_skipped(self):
        # Unsupported filter -> stream skipped, no garbage emitted.
        pdf = _wrap_pdf(b"garbage", b" /Filter /BogusFilter")
        text = PDFParser()._extract_text(pdf)
        assert text == ""

    def test_image_filter_skipped(self):
        # DCTDecode (JPEG) -> skip, do not emit garbage text.
        pdf = _wrap_pdf(b"\xFF\xD8\xFF\xE0jpegblob", b" /Filter /DCTDecode")
        text = PDFParser()._extract_text(pdf)
        assert text == ""

    def test_filter_chain_parse(self):
        # Direct test of _parse_filter_chain.
        assert PDFParser._parse_filter_chain(b"<< /Length 10 >>") == []
        assert PDFParser._parse_filter_chain(b"<< /Filter /FlateDecode >>") == ["FlateDecode"]
        assert PDFParser._parse_filter_chain(
            b"<< /Filter [/ASCII85Decode /FlateDecode] >>"
        ) == ["ASCII85Decode", "FlateDecode"]
        assert PDFParser._parse_filter_chain(
            b"<< /Filter [/ASCII85Decode /DCTDecode] >>"
        ) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ascii85_encode(data: bytes) -> bytes:
    """Reference ASCII85 encoder for test inputs (matches PDF spec)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i + 4 <= n:
        word = int.from_bytes(data[i:i + 4], "big")
        if word == 0:
            out.append(ord("z"))
        else:
            chars = []
            for _ in range(5):
                word, r = divmod(word, 85)
                chars.append(r + 33)
            out.extend(reversed(chars))
        i += 4
    if i < n:
        rem = n - i
        word = int.from_bytes(data[i:] + b"\x00" * (4 - rem), "big")
        chars = []
        for _ in range(5):
            word, r = divmod(word, 85)
            chars.append(r + 33)
        out.extend(reversed(chars)[: rem + 1] if False else list(reversed(chars))[: rem + 1])
    return bytes(out)
