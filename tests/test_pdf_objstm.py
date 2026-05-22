"""Tests for PDF 1.5+ compressed object streams (/ObjStm).

The classic raw byte scan ``N 0 obj << ... >>`` misses dicts that modern
writers (qpdf, Word 2019+, recent LaTeX) tuck inside /ObjStm streams. The
practical breakage is the /Encrypt dict — when it lives in an ObjStm,
``detect_encryption`` would previously return None and the parser would
silently emit garbage for the entire document.
"""

import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag import pdf_crypto  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_objstm(objects: dict[int, bytes]) -> bytes:
    """Build an /ObjStm stream body containing the given ``obj_num -> body``
    map. Returns the full ``N 0 obj << ... >> stream ... endstream endobj``
    indirect object as raw PDF bytes."""
    # Compute concatenated bodies + per-object offsets.
    nums = sorted(objects.keys())
    parts = []
    offsets = []
    cur = 0
    for num in nums:
        body = objects[num]
        offsets.append((num, cur))
        parts.append(body)
        cur += len(body)
    concatenated = b"".join(parts)

    index_parts = []
    for num, off in offsets:
        index_parts.append(f"{num} {off}".encode("ascii"))
    index = b" ".join(index_parts) + b"\n"
    first = len(index)

    body = zlib.compress(index + concatenated)
    header = (
        b"<< /Type /ObjStm /N " + str(len(nums)).encode("ascii")
        + b" /First " + str(first).encode("ascii")
        + b" /Filter /FlateDecode /Length " + str(len(body)).encode("ascii") + b" >>"
    )
    return b"7 0 obj\n" + header + b"\nstream\n" + body + b"\nendstream\nendobj\n"


# ---------------------------------------------------------------------------
# ObjStm parsing
# ---------------------------------------------------------------------------

def test_iter_objstm_streams_finds_objstm():
    objstm = _build_objstm({3: b"<< /Type /Dummy >>", 5: b"<< /Foo 42 >>"})
    pdf = b"%PDF-1.5\n" + objstm + b"trailer << /Size 0 >>\n%%EOF\n"

    found = list(pdf_crypto._iter_objstm_streams(pdf))
    assert len(found) == 1
    header, body = found[0]
    assert b"/Type /ObjStm" in header
    assert b"<< /Type /Dummy >>" in body
    assert b"<< /Foo 42 >>" in body


def test_resolve_in_object_streams_locates_object():
    objstm = _build_objstm({
        3: b"<< /Type /Dummy >>",
        5: b"<< /Encrypt /False >>",
        9: b"<< /Foo 42 >>",
    })
    pdf = b"%PDF-1.5\n" + objstm + b"trailer << /Size 0 >>\n%%EOF\n"

    body3 = pdf_crypto._resolve_in_object_streams(pdf, 3)
    body5 = pdf_crypto._resolve_in_object_streams(pdf, 5)
    body9 = pdf_crypto._resolve_in_object_streams(pdf, 9)
    missing = pdf_crypto._resolve_in_object_streams(pdf, 99)

    assert body3 == b"<< /Type /Dummy >>"
    assert body5 == b"<< /Encrypt /False >>"
    assert body9 == b"<< /Foo 42 >>"
    assert missing is None


def test_resolve_handles_corrupt_objstm_gracefully():
    # Header lies about /N and /First — parser should refuse, not crash.
    bad = (
        b"7 0 obj\n<< /Type /ObjStm /N 5 /First 9999 /Filter /FlateDecode "
        b"/Length 12 >>\nstream\n"
        + zlib.compress(b"garbage data")
        + b"\nendstream\nendobj\n"
    )
    pdf = b"%PDF-1.5\n" + bad + b"trailer << /Size 0 >>\n%%EOF\n"
    assert pdf_crypto._resolve_in_object_streams(pdf, 3) is None


# ---------------------------------------------------------------------------
# /Encrypt dict inside /ObjStm — the real bug this closes
# ---------------------------------------------------------------------------

def test_encrypt_dict_resolved_from_objstm():
    """If the /Encrypt object lives inside an /ObjStm (PDF 1.5+ behavior),
    ``detect_encryption`` must still recover the dict instead of returning
    None and silently treating the document as plaintext."""
    encrypt_dict_body = (
        b"<< /Filter /Standard /V 2 /R 3 /Length 128 /P -4"
        b" /O <" + b"00" * 32 + b">"
        b" /U <" + b"00" * 32 + b"> >>"
    )
    objstm = _build_objstm({2: encrypt_dict_body})
    file_id = b"00" * 16
    trailer = (
        b"trailer\n<< /Encrypt 2 0 R /ID [<" + file_id + b"> <" + file_id
        + b">] >>\n%%EOF\n"
    )
    pdf = b"%PDF-1.5\n" + objstm + trailer

    enc = pdf_crypto.detect_encryption(pdf)
    assert enc is not None, "Encrypt-in-ObjStm should now be detected"
    assert enc.version == 2
    assert enc.revision == 3
    assert enc.length_bits == 128


def test_top_level_encrypt_dict_still_works():
    """Sanity: classic (non-ObjStm) encryption detection is unchanged."""
    pdf = (
        b"%PDF-1.4\n"
        b"2 0 obj << /Filter /Standard /V 1 /R 2 /Length 40 /P -4"
        b" /O <" + b"00" * 32 + b"> /U <" + b"00" * 32 + b"> >> endobj\n"
        b"trailer << /Encrypt 2 0 R /ID [<" + b"00" * 16 + b"> <" + b"00" * 16
        + b">] >>\n%%EOF\n"
    )
    enc = pdf_crypto.detect_encryption(pdf)
    assert enc is not None
    assert enc.version == 1
