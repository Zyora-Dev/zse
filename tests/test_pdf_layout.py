"""Tests for PDF positional text reflow (layout clustering).

Verifies that text fragments emitted in arbitrary draw order — which is the
norm for multi-column papers, tables, and PDFs produced by Word / LaTeX — are
reflowed back into reading order via text-matrix tracking + column detection.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag.parser import PDFParser  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream(body: bytes) -> bytes:
    return (
        b"<< /Length " + str(len(body)).encode("ascii") + b" >>\n"
        b"stream\n" + body + b"\nendstream\n"
    )


def _wrap(content: bytes) -> bytes:
    return b"%PDF-1.4\n" + _make_stream(content) + b"%%EOF\n"


def _extract(content: bytes) -> str:
    parser = PDFParser()
    return parser._extract_text(_wrap(content))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_basic_operators():
    toks = PDFParser._tokenize_content_stream(
        "BT /F1 12 Tf 100 200 Td (Hello) Tj ET"
    )
    assert toks == ["BT", "/F1", "12", "Tf", "100", "200", "Td", "(Hello)", "Tj", "ET"]


def test_tokenizer_string_with_spaces_and_operators_inside():
    toks = PDFParser._tokenize_content_stream("BT (Hello BT ET Tj) Tj ET")
    # The string must be one atomic token even though it contains operator-like words.
    assert toks == ["BT", "(Hello BT ET Tj)", "Tj", "ET"]


def test_tokenizer_balanced_parens_in_string():
    toks = PDFParser._tokenize_content_stream("(a(b)c) Tj")
    assert toks == ["(a(b)c)", "Tj"]


def test_tokenizer_tj_array_is_one_token():
    toks = PDFParser._tokenize_content_stream("[(Hel) -20 (lo)] TJ")
    assert toks == ["[(Hel) -20 (lo)]", "TJ"]


def test_tokenizer_hex_string():
    toks = PDFParser._tokenize_content_stream("<48656c6c6f> Tj")
    assert toks == ["<48656c6c6f>", "Tj"]


def test_tokenizer_comment_skipped():
    toks = PDFParser._tokenize_content_stream("BT % comment here\n (X) Tj ET")
    assert toks == ["BT", "(X)", "Tj", "ET"]


# ---------------------------------------------------------------------------
# Single-column reading order
# ---------------------------------------------------------------------------

def test_single_column_top_to_bottom():
    """Three lines at decreasing y should come out in top-to-bottom order
    even when drawn bottom-to-top in the content stream."""
    content = (
        b"BT 100 100 Td (third) Tj ET\n"
        b"BT 100 200 Td (second) Tj ET\n"
        b"BT 100 300 Td (first) Tj ET\n"
    )
    out = _extract(content)
    lines = [l for l in out.split("\n") if l]
    assert lines == ["first", "second", "third"]


def test_same_line_left_to_right():
    """Fragments at the same y but different x should be ordered left-to-right
    regardless of draw order."""
    content = (
        b"BT 300 500 Td (world) Tj ET\n"
        b"BT 100 500 Td (Hello) Tj ET\n"
    )
    out = _extract(content)
    assert "Hello" in out and "world" in out
    assert out.index("Hello") < out.index("world")


def test_word_spacing_inserted_on_x_gap():
    """Two fragments at the same y with a meaningful x gap get joined with a space."""
    content = (
        b"BT 100 500 Td (Hello) Tj ET\n"
        b"BT 200 500 Td (World) Tj ET\n"
    )
    out = _extract(content)
    assert "Hello World" in out or "Hello  World" in out


def test_tm_operator_sets_position():
    """Tm directly sets the text matrix translation."""
    content = (
        b"BT 1 0 0 1 100 100 Tm (bottom) Tj ET\n"
        b"BT 1 0 0 1 100 400 Tm (top) Tj ET\n"
    )
    out = _extract(content)
    lines = [l for l in out.split("\n") if l]
    assert lines == ["top", "bottom"]


# ---------------------------------------------------------------------------
# Multi-column reflow
# ---------------------------------------------------------------------------

def test_two_column_reflow():
    """A two-column layout — left col x≈100, right col x≈400 — interleaved in
    draw order should reflow so the entire left column precedes the right."""
    # Build 6 lines per column at descending y. Interleave the draw order.
    lines = []
    for i in range(6):
        y = 700 - i * 30
        lines.append(b"BT 100 " + str(y).encode("ascii") + b" Td (L" + str(i).encode("ascii") + b") Tj ET\n")
        lines.append(b"BT 400 " + str(y).encode("ascii") + b" Td (R" + str(i).encode("ascii") + b") Tj ET\n")
    out = _extract(b"".join(lines))

    # All L's must come before all R's.
    left_positions = [out.index(f"L{i}") for i in range(6)]
    right_positions = [out.index(f"R{i}") for i in range(6)]
    assert max(left_positions) < min(right_positions), (
        f"Columns not separated:\n{out}"
    )
    # Within each column, ordering top-to-bottom.
    assert left_positions == sorted(left_positions)
    assert right_positions == sorted(right_positions)


def test_single_column_not_misdetected_as_multi():
    """A normal single-column doc with a narrow indented line should NOT
    trigger column split."""
    content = []
    for i in range(8):
        y = 700 - i * 20
        # All lines start at x=100 except one indented at x=140.
        x = 140 if i == 3 else 100
        content.append(
            f"BT {x} {y} Td (line{i}) Tj ET\n".encode("ascii")
        )
    out = _extract(b"".join(content))
    lines = [l for l in out.split("\n") if l]
    # Should appear in original top-to-bottom order, no column split.
    assert lines == [f"line{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# TJ array + kerning
# ---------------------------------------------------------------------------

def test_tj_array_concatenates_with_kerning_ignored():
    content = b"BT 100 500 Td [(Hel) -50 (lo) -50 ( World)] TJ ET\n"
    out = _extract(content)
    assert "Hello World" in out or "Hello  World" in out


# ---------------------------------------------------------------------------
# Backward-compatible smoke test: legacy line-based input still works
# ---------------------------------------------------------------------------

def test_legacy_line_formatted_stream_still_extracts():
    """The previous extractor was line-based. Ensure streams that already
    formatted each operator on its own line still produce readable text."""
    content = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"100 700 Td\n"
        b"(Hello) Tj\n"
        b"100 680 Td\n"
        b"(World) Tj\n"
        b"ET\n"
    )
    out = _extract(content)
    assert "Hello" in out
    assert "World" in out
