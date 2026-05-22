"""Tests for PDF CMap / ToUnicode glyph mapping.

Verifies that PDFs using subset fonts (where text streams contain glyph IDs,
not ASCII codes) are correctly remapped to readable Unicode.
"""

import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag.parser import PDFParser  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal PDF builders
# ---------------------------------------------------------------------------

def _make_stream(body: bytes, extra_dict: bytes = b"") -> bytes:
    return (
        b"<< /Length " + str(len(body)).encode("ascii") + extra_dict + b" >>\n"
        b"stream\n" + body + b"\nendstream\n"
    )


def _pdf_with_cmap(content: bytes, cmap_body: bytes) -> bytes:
    """Wrap a content stream + a /ToUnicode CMap stream into a parseable PDF blob.

    The parser scans for `stream` markers globally, so a full xref/trailer is
    unnecessary for these tests.
    """
    cmap_stream = _make_stream(cmap_body, extra_dict=b" /ToUnicode true")
    content_stream = _make_stream(content)
    return b"%PDF-1.4\n" + cmap_stream + content_stream + b"%%EOF\n"


# ---------------------------------------------------------------------------
# CMap parsing
# ---------------------------------------------------------------------------

class TestCMapBuilder:
    def test_bfchar_basic(self):
        cmap_body = (
            b"/CIDInit /ProcSet findresource begin\n"
            b"12 dict begin\n"
            b"begincmap\n"
            b"2 beginbfchar\n"
            b"<0001> <0048>\n"
            b"<0002> <0069>\n"
            b"endbfchar\n"
            b"endcmap\n"
        )
        parser = PDFParser()
        cmap, code_len = parser._build_cmap(_pdf_with_cmap(b"", cmap_body))
        assert cmap[0x0001] == "H"
        assert cmap[0x0002] == "i"
        assert code_len == 2

    def test_bfrange_start_form(self):
        # <lo> <hi> <start> — A..E -> 0x41..0x45
        cmap_body = (
            b"begincmap\n"
            b"1 beginbfrange\n"
            b"<0010> <0014> <0041>\n"
            b"endbfrange\n"
            b"endcmap\n"
        )
        parser = PDFParser()
        cmap, code_len = parser._build_cmap(_pdf_with_cmap(b"", cmap_body))
        assert cmap[0x0010] == "A"
        assert cmap[0x0011] == "B"
        assert cmap[0x0014] == "E"
        assert code_len == 2

    def test_bfrange_array_form(self):
        cmap_body = (
            b"begincmap\n"
            b"1 beginbfrange\n"
            b"<0020> <0022> [<0058> <0059> <005A>]\n"
            b"endbfrange\n"
            b"endcmap\n"
        )
        parser = PDFParser()
        cmap, _ = parser._build_cmap(_pdf_with_cmap(b"", cmap_body))
        assert cmap[0x0020] == "X"
        assert cmap[0x0021] == "Y"
        assert cmap[0x0022] == "Z"

    def test_unicode_multichar_mapping(self):
        # Ligature: <0050> -> "ffi"  (utf-16-be: 0066 0066 0069)
        cmap_body = (
            b"begincmap\n"
            b"1 beginbfchar\n"
            b"<0050> <006600660069>\n"
            b"endbfchar\n"
            b"endcmap\n"
        )
        parser = PDFParser()
        cmap, _ = parser._build_cmap(_pdf_with_cmap(b"", cmap_body))
        assert cmap[0x0050] == "ffi"


# ---------------------------------------------------------------------------
# Apply CMap to extracted strings
# ---------------------------------------------------------------------------

class TestCMapApplication:
    def test_hex_string_with_cmap(self):
        # Glyph IDs 0001..0005 -> "Hello" via bfchar
        cmap_body = (
            b"begincmap\n"
            b"5 beginbfchar\n"
            b"<0001> <0048>\n"
            b"<0002> <0065>\n"
            b"<0003> <006C>\n"
            b"<0004> <006C>\n"
            b"<0005> <006F>\n"
            b"endbfchar\n"
            b"endcmap\n"
        )
        # Content stream: BT <00010002000300040005> Tj ET
        content = b"BT\n<00010002000300040005> Tj\nET\n"
        pdf = _pdf_with_cmap(content, cmap_body)

        parser = PDFParser()
        text = parser._extract_text(pdf)
        assert "Hello" in text

    def test_tj_array_with_cmap(self):
        cmap_body = (
            b"begincmap\n"
            b"1 beginbfrange\n"
            b"<0001> <001A> <0041>\n"
            b"endbfrange\n"
            b"endcmap\n"
        )
        # "CAB" via glyph IDs 0003 0001 0002
        content = b"BT\n[<0003> -50 <0001> <0002>] TJ\nET\n"
        pdf = _pdf_with_cmap(content, cmap_body)

        parser = PDFParser()
        text = parser._extract_text(pdf)
        assert "CAB" in text

    def test_pdf_without_cmap_unchanged(self):
        # Plain ASCII content, no CMap — verify fallback path still works.
        content = b"BT\n(Hello World) Tj\nET\n"
        pdf = b"%PDF-1.4\n" + _make_stream(content) + b"%%EOF\n"

        parser = PDFParser()
        text = parser._extract_text(pdf)
        assert "Hello World" in text

    def test_fallback_for_unmapped_codes(self):
        # 1-byte CMap covers only 0x41 -> 'X'. 0x42 is unmapped -> latin-1 'B'.
        cmap_body = (
            b"begincmap\n"
            b"1 beginbfchar\n"
            b"<41> <0058>\n"  # 'A' (1-byte code) -> 'X'
            b"endbfchar\n"
            b"endcmap\n"
        )
        content = b"BT\n<4142> Tj\nET\n"
        pdf = _pdf_with_cmap(content, cmap_body)

        parser = PDFParser()
        text = parser._extract_text(pdf)
        assert "XB" in text

    def test_full_parse_with_cmap(self):
        """End-to-end: PDFParser.parse() returns chunks with mapped text."""
        # Glyph IDs 0001..0005 -> 'H','e','l','l','o'; repeat to exceed the
        # parser's 10-char minimum sanity check on extracted text.
        cmap_body = (
            b"begincmap\n"
            b"5 beginbfchar\n"
            b"<0001> <0048>\n"
            b"<0002> <0065>\n"
            b"<0003> <006C>\n"
            b"<0004> <006C>\n"
            b"<0005> <006F>\n"
            b"endbfchar\n"
            b"endcmap\n"
        )
        # "HelloHelloHello" via three repeats of glyph sequence 1..5
        content = (
            b"BT\n<000100020003000400050001000200030004000500010002000300040005> Tj\nET\n"
        )
        pdf = _pdf_with_cmap(content, cmap_body)

        parser = PDFParser(chunk_size=128, overlap=0)
        chunks = parser.parse(pdf.decode("latin-1"))
        assert len(chunks) >= 1
        joined = " ".join(c.text for c in chunks)
        assert "Hello" in joined
        assert chunks[0].metadata.get("parser") == "pdf"

# ---------------------------------------------------------------------------
# CMap through compressed (FlateDecode) streams
# ---------------------------------------------------------------------------

class TestCompressedCMap:
    def test_flate_compressed_cmap(self):
        cmap_inner = (
            b"begincmap\n"
            b"1 beginbfchar\n"
            b"<0001> <0041>\n"
            b"endbfchar\n"
            b"endcmap\n"
        )
        compressed = zlib.compress(cmap_inner)
        cmap_stream = (
            b"<< /Length " + str(len(compressed)).encode("ascii")
            + b" /Filter /FlateDecode /ToUnicode true >>\n"
            b"stream\n" + compressed + b"\nendstream\n"
        )
        content_stream = _make_stream(b"BT\n<0001> Tj\nET\n")
        pdf = b"%PDF-1.4\n" + cmap_stream + content_stream + b"%%EOF\n"

        parser = PDFParser()
        text = parser._extract_text(pdf)
        assert "A" in text
