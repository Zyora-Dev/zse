"""Tests for the PDF OCR fallback hook.

ZSE itself stays zero-dep. Users supply their own OCR backend (pytesseract,
EasyOCR, cloud API) via an ``ocr_fn(image_bytes, format_hint) -> str``
callable. The hook only fires when normal text extraction yields nothing —
text-based PDFs never pay the OCR cost.
"""

import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag.parser import PDFParser  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_stream(image_bytes: bytes, image_filter: bytes = b"/DCTDecode") -> bytes:
    return (
        b"1 0 obj\n"
        b"<< /Type /XObject /Subtype /Image /Filter " + image_filter
        + b" /Width 100 /Height 100 /Length "
        + str(len(image_bytes)).encode("ascii") + b" >>\n"
        b"stream\n" + image_bytes + b"\nendstream\nendobj\n"
    )


def _make_text_stream(body: bytes) -> bytes:
    return (
        b"<< /Length " + str(len(body)).encode("ascii") + b" >>\n"
        b"stream\n" + body + b"\nendstream\n"
    )


def _scanned_pdf(image_bytes: bytes, image_filter: bytes = b"/DCTDecode") -> bytes:
    """PDF with one image XObject and no extractable text."""
    return b"%PDF-1.4\n" + _make_image_stream(image_bytes, image_filter) + b"%%EOF\n"


def _text_pdf_with_image(image_bytes: bytes) -> bytes:
    """PDF that already has extractable text *and* an image — OCR must NOT fire."""
    text_body = b"BT 100 700 Td (Real text content here is enough) Tj ET\n"
    return (
        b"%PDF-1.4\n"
        + _make_text_stream(text_body)
        + _make_image_stream(image_bytes)
        + b"%%EOF\n"
    )


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def test_extract_images_finds_jpeg():
    parser = PDFParser()
    jpeg_bytes = b"\xff\xd8\xff\xe0FAKE_JPEG_BYTES\xff\xd9"
    pdf = _scanned_pdf(jpeg_bytes)

    images = parser._extract_images(pdf)
    assert len(images) == 1
    data, hint = images[0]
    assert data == jpeg_bytes
    assert hint == "jpeg"


def test_extract_images_recognizes_filters():
    parser = PDFParser()
    cases = [
        (b"/DCTDecode", "jpeg"),
        (b"/JPXDecode", "jp2"),
        (b"/CCITTFaxDecode", "ccitt"),
        (b"/JBIG2Decode", "jbig2"),
        (b"/FlateDecode", "raw"),
    ]
    for image_filter, expected_hint in cases:
        pdf = _scanned_pdf(b"IMG_BYTES", image_filter=image_filter)
        images = parser._extract_images(pdf)
        assert len(images) == 1, f"failed for {image_filter}"
        assert images[0][1] == expected_hint, (
            f"expected {expected_hint}, got {images[0][1]} for {image_filter}"
        )


def test_extract_images_ignores_non_image_streams():
    parser = PDFParser()
    pdf = (
        b"%PDF-1.4\n"
        + _make_text_stream(b"BT (Hello) Tj ET")
        + _make_image_stream(b"IMG")
        + b"%%EOF\n"
    )
    images = parser._extract_images(pdf)
    assert len(images) == 1
    assert images[0][0] == b"IMG"


# ---------------------------------------------------------------------------
# OCR fallback wiring
# ---------------------------------------------------------------------------

def test_ocr_fires_on_image_only_pdf():
    calls = []

    def fake_ocr(image_bytes, hint):
        calls.append((image_bytes[:10], hint))
        return "Extracted from image: hello world via OCR"

    parser = PDFParser(ocr_fn=fake_ocr)
    pdf = _scanned_pdf(b"\xff\xd8\xff\xe0IMG_PAYLOAD\xff\xd9")

    chunks = parser.parse(pdf.decode("latin-1"))
    assert chunks, "should produce chunks via OCR"
    assert any("hello world" in c.text for c in chunks)
    assert all(c.metadata.get("pdf_ocr") is True for c in chunks)
    assert len(calls) == 1
    assert calls[0][1] == "jpeg"


def test_ocr_does_not_fire_when_text_extractable():
    """If the PDF already has extractable text, OCR must NOT be invoked."""
    calls = []

    def boom(image_bytes, hint):
        calls.append((image_bytes, hint))
        return "SHOULD NOT APPEAR"

    parser = PDFParser(ocr_fn=boom)
    pdf = _text_pdf_with_image(b"\xff\xd8IMG\xff\xd9")
    chunks = parser.parse(pdf.decode("latin-1"))

    assert chunks
    assert not calls, "OCR fired on a text-based PDF"
    assert all(c.metadata.get("pdf_ocr") is not True for c in chunks)
    full_text = "\n".join(c.text for c in chunks)
    assert "SHOULD NOT APPEAR" not in full_text
    assert "Real text content" in full_text


def test_ocr_metadata_override_takes_precedence():
    """``metadata['ocr_fn']`` overrides the parser-instance ocr_fn."""
    instance_calls = []
    override_calls = []

    def instance_ocr(image_bytes, hint):
        instance_calls.append(hint)
        return "from instance"

    def override_ocr(image_bytes, hint):
        override_calls.append(hint)
        return "from override callable"

    parser = PDFParser(ocr_fn=instance_ocr)
    pdf = _scanned_pdf(b"IMG_DATA")
    chunks = parser.parse(pdf.decode("latin-1"), metadata={"ocr_fn": override_ocr})

    assert any("override callable" in c.text for c in chunks)
    assert override_calls == ["jpeg"]
    assert instance_calls == []


def test_ocr_accepts_single_arg_callable():
    """Backward-compat: a callable that only takes image_bytes still works."""
    def simple_ocr(image_bytes):
        return "single-arg ocr says hello there"

    parser = PDFParser(ocr_fn=simple_ocr)
    pdf = _scanned_pdf(b"IMG_BYTES")
    chunks = parser.parse(pdf.decode("latin-1"))

    assert any("single-arg ocr" in c.text for c in chunks)


def test_ocr_swallows_per_image_errors():
    """A bad OCR call on one image must not kill the whole document."""
    def flaky_ocr(image_bytes, hint):
        if image_bytes.startswith(b"BAD"):
            raise RuntimeError("boom")
        return "good image extracted text content"

    pdf = (
        b"%PDF-1.4\n"
        + _make_image_stream(b"BAD_BYTES_HERE")
        + _make_image_stream(b"GOOD_BYTES_HERE")
        + b"%%EOF\n"
    )
    parser = PDFParser(ocr_fn=flaky_ocr)
    chunks = parser.parse(pdf.decode("latin-1"))

    assert chunks
    full = "\n".join(c.text for c in chunks)
    assert "good image extracted" in full


def test_no_ocr_callable_returns_failure_chunk():
    """Without an OCR backend, image-only PDFs still produce the legacy
    failure-marker chunk so callers can detect the issue."""
    parser = PDFParser()
    pdf = _scanned_pdf(b"\xff\xd8IMG\xff\xd9")
    chunks = parser.parse(pdf.decode("latin-1"))

    assert len(chunks) == 1
    assert chunks[0].metadata.get("error") == "extraction_failed"
    assert not chunks[0].metadata.get("pdf_ocr")
