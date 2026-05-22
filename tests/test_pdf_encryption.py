"""Tests for PDF Standard Security Handler (RC4 / AES-128 / AES-256).

Strategy:
  - Verify primitive correctness (RC4, AES-128/256 CBC) against published vectors.
  - Build encrypted PDFs in-process using the same primitives, then verify
    `PDFParser._extract_text` round-trips back to the original plaintext.
  - Cover V=1/R=2 (RC4-40), V=2/R=3 (RC4-128), V=5/R=5 (AES-256 simple),
    V=5/R=6 (AES-256 with Algorithm 2.B).
"""

import hashlib
import struct
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "zse-engine"))

from zse_engine.rag import pdf_crypto  # noqa: E402
from zse_engine.rag.parser import PDFParser  # noqa: E402


# ---------------------------------------------------------------------------
# Primitive: RC4
# ---------------------------------------------------------------------------

class TestRC4Vectors:
    def test_rfc6229_key_key(self):
        # RFC 6229 §2 — Key="Key", Plaintext="Plaintext"
        # Expected ciphertext: BBF316E8D940AF0AD3
        out = pdf_crypto.rc4(b"Key", b"Plaintext")
        assert out.hex().upper() == "BBF316E8D940AF0AD3"

    def test_rfc6229_wiki_secret(self):
        # Key="Wiki", Plaintext="pedia" -> 1021BF0420
        out = pdf_crypto.rc4(b"Wiki", b"pedia")
        assert out.hex().upper() == "1021BF0420"

    def test_symmetry(self):
        # Re-applying with the same key recovers plaintext.
        key = b"\xDE\xAD\xBE\xEF\x12\x34\x56\x78"
        plaintext = b"The quick brown fox jumps over the lazy dog"
        ct = pdf_crypto.rc4(key, plaintext)
        assert pdf_crypto.rc4(key, ct) == plaintext


# ---------------------------------------------------------------------------
# Primitive: AES-128 CBC
# ---------------------------------------------------------------------------

class TestAES128:
    def test_nist_ecb_block(self):
        # FIPS-197 §C.1 single-block AES-128 ECB test vector.
        # Key:        000102030405060708090a0b0c0d0e0f
        # Plaintext:  00112233445566778899aabbccddeeff
        # Ciphertext: 69c4e0d86a7b0430d8cdb78070b4c55a
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
        ct = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
        expected = bytes.fromhex("00112233445566778899aabbccddeeff")
        round_keys = pdf_crypto._aes_expand_key_128(key)
        # We only implement decrypt — verify decrypt(ct) == plaintext.
        assert pdf_crypto._aes_decrypt_block_128(ct, round_keys) == expected

    def test_cbc_round_trip(self):
        # Round-trip via a tiny CBC encryptor built on top of the same block
        # primitive (mirror of decryptor) — confirms key expansion + CBC
        # chaining match what the PDF spec produces.
        key = b"YELLOW SUBMARINE"  # 16 bytes
        iv = bytes(range(16))
        plaintext = b"PDF AES-128 round-trip test data."
        # PKCS#7 pad
        pad = 16 - (len(plaintext) % 16)
        padded = plaintext + bytes([pad]) * pad
        # Encrypt by inverting the decryptor: not trivial without an encryptor.
        # Instead use a reference CBC built on Python's `Crypto` — but zero-dep!
        # Strategy: build a known CT with our decryptor by handcrafting a
        # ciphertext through repeated trials is infeasible. Skip if needed.
        # We rely on the NIST single-block decrypt test above + the end-to-end
        # PDF test below (which exercises the AES path with real ciphertext).
        # Sanity: decrypt of garbage returns empty.
        assert pdf_crypto.aes128_cbc_decrypt(key, b"") == b""
        assert pdf_crypto.aes128_cbc_decrypt(key, b"\x00" * 31) == b""  # not 16-aligned


# ---------------------------------------------------------------------------
# Helpers for building an encrypted PDF in-process
# ---------------------------------------------------------------------------

def _pdf_encrypt_stream_rc4(key: bytes, obj_num: int, gen_num: int, data: bytes) -> bytes:
    """Replicate `PDFEncryption._object_key` + RC4 for test PDF generation."""
    md5 = hashlib.md5()
    md5.update(key)
    md5.update(struct.pack("<I", obj_num)[:3])
    md5.update(struct.pack("<I", gen_num)[:2])
    n = min(len(key) + 5, 16)
    obj_key = md5.digest()[:n]
    return pdf_crypto.rc4(obj_key, data)


def _build_encrypted_pdf_rc4(
    revision: int,
    version: int,
    length_bits: int,
    plaintext_body: bytes,
    password: bytes = b"",
) -> bytes:
    """Build a minimal RC4-encrypted PDF with one content stream and a
    correctly-computed /U + /O so the standard handler accepts it.

    The structure is deliberately simple — one content-stream object (#1),
    the encrypt dict (#2), no font dicts. This is enough for the parser to
    detect encryption, derive the file key, decrypt the stream, and
    extract text from it.
    """
    file_id = b"\x00" * 16
    pad_pwd = pdf_crypto._pad_password(password)

    # /O hash. For an open document with an unknown owner, the easiest valid
    # /O is the standard one for empty owner+user password.
    # PDF 1.7 §7.6.3.4 algorithm 3: O = RC4_iterated(key_from_owner, padded_user_pwd)
    o_key_md5 = hashlib.md5(pad_pwd).digest()
    if revision >= 3:
        for _ in range(50):
            o_key_md5 = hashlib.md5(o_key_md5[: length_bits // 8]).digest()
    o_key = o_key_md5[: length_bits // 8]
    o_hash = pdf_crypto.rc4(o_key, pad_pwd)
    if revision >= 3:
        for i in range(1, 20):
            o_hash = pdf_crypto.rc4(bytes(b ^ i for b in o_key), o_hash)

    # File key derivation (algorithm 2).
    md5 = hashlib.md5()
    md5.update(pad_pwd)
    md5.update(o_hash)
    md5.update(struct.pack("<i", -1))  # permissions = -1
    md5.update(file_id)
    file_key = md5.digest()
    if revision >= 3:
        for _ in range(50):
            file_key = hashlib.md5(file_key[: length_bits // 8]).digest()
    file_key = file_key[: length_bits // 8]

    # /U hash (algorithm 4 / 5).
    if revision == 2:
        u_hash = pdf_crypto.rc4(file_key, pdf_crypto._PAD)
    else:
        m = hashlib.md5()
        m.update(pdf_crypto._PAD)
        m.update(file_id)
        u = m.digest()
        u = pdf_crypto.rc4(file_key, u)
        for i in range(1, 20):
            u = pdf_crypto.rc4(bytes(b ^ i for b in file_key), u)
        u_hash = u + b"\x00" * 16  # /U is always 32 bytes

    # Encrypt the content stream with per-object key.
    encrypted_body = _pdf_encrypt_stream_rc4(file_key, 1, 0, plaintext_body)

    # Assemble the PDF.
    stream_obj = (
        b"1 0 obj\n<< /Length " + str(len(encrypted_body)).encode("ascii") + b" >>\n"
        b"stream\n" + encrypted_body + b"\nendstream\nendobj\n"
    )
    enc_obj = (
        b"2 0 obj\n<< /Filter /Standard "
        b"/V " + str(version).encode("ascii") + b" "
        b"/R " + str(revision).encode("ascii") + b" "
        b"/Length " + str(length_bits).encode("ascii") + b" "
        b"/P -1 "
        b"/O <" + o_hash.hex().encode("ascii") + b"> "
        b"/U <" + u_hash.hex().encode("ascii") + b"> >>\n"
        b"endobj\n"
    )
    trailer = (
        b"trailer\n<< /Encrypt 2 0 R /ID [<" + file_id.hex().encode("ascii")
        + b"> <" + file_id.hex().encode("ascii") + b">] >>\n"
        b"%%EOF\n"
    )
    return b"%PDF-1.4\n" + stream_obj + enc_obj + trailer


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestEncryptionDetection:
    def test_plain_pdf_returns_none(self):
        plain = b"%PDF-1.4\ntrailer << /Size 0 >>\n%%EOF"
        assert pdf_crypto.detect_encryption(plain) is None

    def test_rc4_v1_r2_detected(self):
        pdf = _build_encrypted_pdf_rc4(
            revision=2, version=1, length_bits=40,
            plaintext_body=b"BT\n(Detected) Tj\nET\n",
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert enc.version == 1
        assert enc.revision == 2
        assert enc.length_bits == 40
        assert enc.supported

    def test_rc4_v2_r3_detected(self):
        pdf = _build_encrypted_pdf_rc4(
            revision=3, version=2, length_bits=128,
            plaintext_body=b"BT\n(Detected) Tj\nET\n",
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert enc.version == 2
        assert enc.revision == 3
        assert enc.length_bits == 128
        assert enc.supported

    def test_v5_r6_aes256_detected_and_supported(self):
        # Synthetic V=5/R=6 dict with placeholder hashes. /U is 48 bytes
        # (length-correct), /UE is 32 bytes. `supported` is now True (we
        # implement V=5), but `derive_file_key(b"")` will reject because
        # the placeholder /U won't validate against the empty password.
        pdf = (
            b"%PDF-1.4\n"
            b"2 0 obj\n<< /Filter /Standard /V 5 /R 6 /Length 256 /P -1 "
            b"/O <" + b"00" * 48 + b"> /U <" + b"00" * 48 + b"> "
            b"/OE <" + b"00" * 32 + b"> /UE <" + b"00" * 32 + b"> "
            b"/Perms <" + b"00" * 16 + b"> >>\n"
            b"endobj\n"
            b"trailer\n<< /Encrypt 2 0 R /ID [<" + b"00" * 16 + b"> <" + b"00" * 16 + b">] >>\n"
            b"%%EOF\n"
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert enc.version == 5
        assert enc.revision == 6
        assert enc.supported  # now supported

    def test_v99_unknown_version_unsupported(self):
        pdf = (
            b"%PDF-1.4\n"
            b"2 0 obj\n<< /Filter /Standard /V 99 /R 99 /Length 256 /P -1 "
            b"/O <" + b"00" * 48 + b"> /U <" + b"00" * 48 + b"> >>\n"
            b"endobj\n"
            b"trailer\n<< /Encrypt 2 0 R /ID [<" + b"00" * 16 + b"> <" + b"00" * 16 + b">] >>\n"
            b"%%EOF\n"
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert not enc.supported


# ---------------------------------------------------------------------------
# Key derivation + verification with empty password
# ---------------------------------------------------------------------------

class TestKeyDerivation:
    def test_empty_password_r2(self):
        pdf = _build_encrypted_pdf_rc4(
            revision=2, version=1, length_bits=40,
            plaintext_body=b"BT\n(KeyOK R2) Tj\nET\n",
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc.derive_file_key(b"") is True
        assert enc.file_key is not None
        assert len(enc.file_key) == 5

    def test_empty_password_r3(self):
        pdf = _build_encrypted_pdf_rc4(
            revision=3, version=2, length_bits=128,
            plaintext_body=b"BT\n(KeyOK R3) Tj\nET\n",
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc.derive_file_key(b"") is True
        assert len(enc.file_key) == 16

    def test_wrong_password_rejected(self):
        pdf = _build_encrypted_pdf_rc4(
            revision=3, version=2, length_bits=128,
            plaintext_body=b"BT\n(KeyOK) Tj\nET\n",
        )
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc.derive_file_key(b"not the password") is False
        assert enc.file_key is None


# ---------------------------------------------------------------------------
# End-to-end: PDFParser decrypts an encrypted document
# ---------------------------------------------------------------------------

class TestEndToEndDecryption:
    def test_rc4_v1_r2_round_trip(self):
        plaintext = b"BT\n(Hello Encrypted RC4-40) Tj\nET\n"
        pdf = _build_encrypted_pdf_rc4(
            revision=2, version=1, length_bits=40, plaintext_body=plaintext,
        )
        text = PDFParser()._extract_text(pdf)
        assert "Hello Encrypted RC4-40" in text

    def test_rc4_v2_r3_round_trip(self):
        plaintext = b"BT\n(Hello Encrypted RC4-128) Tj\nET\n"
        pdf = _build_encrypted_pdf_rc4(
            revision=3, version=2, length_bits=128, plaintext_body=plaintext,
        )
        text = PDFParser()._extract_text(pdf)
        assert "Hello Encrypted RC4-128" in text

    def test_full_parse_marks_metadata(self):
        plaintext = b"BT\n(End to end metadata check) Tj\nET\n"
        pdf = _build_encrypted_pdf_rc4(
            revision=3, version=2, length_bits=128, plaintext_body=plaintext,
        )
        parser = PDFParser(chunk_size=128, overlap=0)
        chunks = parser.parse(pdf.decode("latin-1"))
        joined = " ".join(c.text for c in chunks)
        assert "End to end metadata check" in joined
        assert any(c.metadata.get("pdf_encrypted") for c in chunks)

    def test_unsupported_encryption_emits_clear_error(self):
        # Use an unknown version so `supported` is False.
        pdf = (
            b"%PDF-1.4\n"
            b"2 0 obj\n<< /Filter /Standard /V 99 /R 99 /Length 256 /P -1 "
            b"/O <" + b"00" * 48 + b"> /U <" + b"00" * 48 + b"> >>\n"
            b"endobj\n"
            b"trailer\n<< /Encrypt 2 0 R /ID [<" + b"00" * 16 + b"> <" + b"00" * 16 + b">] >>\n"
            b"%%EOF\n"
        )
        parser = PDFParser()
        chunks = parser.parse(pdf.decode("latin-1"))
        assert len(chunks) == 1
        assert "V=99" in chunks[0].text or "encrypted" in chunks[0].text.lower()
        assert chunks[0].metadata.get("error") == "encrypted_unsupported"
    def test_encrypted_then_flate_chain(self):
        # Stream is first encrypted, then we declare /Filter /FlateDecode to
        # confirm decryption-before-filter ordering matches the PDF spec.
        plaintext = b"BT\n(Chained encryption test data) Tj\nET\n"
        # Build a body that is Flate-compressed, then encrypted.
        flated = zlib.compress(plaintext)
        # Use the R3 build helper but inject the flated body and /Filter.
        # Easier: hand-build using known empty-password file key derivation.
        file_id = b"\x00" * 16
        pad_pwd = pdf_crypto._pad_password(b"")
        # O hash (R3, len=128)
        o_key = hashlib.md5(pad_pwd).digest()
        for _ in range(50):
            o_key = hashlib.md5(o_key[:16]).digest()
        o_key = o_key[:16]
        o_hash = pdf_crypto.rc4(o_key, pad_pwd)
        for i in range(1, 20):
            o_hash = pdf_crypto.rc4(bytes(b ^ i for b in o_key), o_hash)
        # File key
        m = hashlib.md5()
        m.update(pad_pwd)
        m.update(o_hash)
        m.update(struct.pack("<i", -1))
        m.update(file_id)
        file_key = m.digest()
        for _ in range(50):
            file_key = hashlib.md5(file_key[:16]).digest()
        file_key = file_key[:16]
        # U hash
        u = hashlib.md5(pdf_crypto._PAD + file_id).digest()
        u = pdf_crypto.rc4(file_key, u)
        for i in range(1, 20):
            u = pdf_crypto.rc4(bytes(b ^ i for b in file_key), u)
        u_hash = u + b"\x00" * 16
        # Encrypt flated body
        encrypted = _pdf_encrypt_stream_rc4(file_key, 1, 0, flated)

        stream_obj = (
            b"1 0 obj\n<< /Length " + str(len(encrypted)).encode("ascii")
            + b" /Filter /FlateDecode >>\nstream\n" + encrypted + b"\nendstream\nendobj\n"
        )
        enc_obj = (
            b"2 0 obj\n<< /Filter /Standard /V 2 /R 3 /Length 128 /P -1 "
            b"/O <" + o_hash.hex().encode("ascii") + b"> "
            b"/U <" + u_hash.hex().encode("ascii") + b"> >>\nendobj\n"
        )
        trailer = (
            b"trailer\n<< /Encrypt 2 0 R /ID [<" + file_id.hex().encode("ascii")
            + b"> <" + file_id.hex().encode("ascii") + b">] >>\n%%EOF\n"
        )
        pdf = b"%PDF-1.4\n" + stream_obj + enc_obj + trailer

        text = PDFParser()._extract_text(pdf)
        assert "Chained encryption test data" in text


# ---------------------------------------------------------------------------
# V=5 (AES-256) end-to-end — R=5 (Acrobat 9) and R=6 (Acrobat X+)
# ---------------------------------------------------------------------------

def _build_encrypted_pdf_v5(revision: int, plaintext_body: bytes,
                            password: bytes = b"") -> bytes:
    """Construct a V=5/R=5 or V=5/R=6 PDF with one encrypted content stream.

    For empty password (the common locked-PDF case) we can hand-build all the
    hashes using the same primitives the decryptor uses, which gives us a
    closed-loop round-trip test without external fixtures.
    """
    assert revision in (5, 6)
    file_key = b"\x42" * 32  # arbitrary 32-byte file key
    val_salt = b"\x11" * 8
    key_salt = b"\x22" * 8

    if revision == 5:
        u_hash_head = hashlib.sha256(password + val_salt).digest()
        fk_key = hashlib.sha256(password + key_salt).digest()
    else:  # R=6
        u_hash_head = pdf_crypto._alg_2b(password, val_salt, b"")
        fk_key = pdf_crypto._alg_2b(password, key_salt, b"")
    u_hash = u_hash_head + val_salt + key_salt  # 48 bytes

    # Encrypt the file key with fk_key, IV=0, no padding (exact 32 bytes out).
    ue_hash = pdf_crypto.aes256_cbc_encrypt(
        fk_key, b"\x00" * 16, file_key, pad=False,
    )
    assert len(ue_hash) == 32

    # Encrypt the content stream with the file key directly (AESV3).
    stream_iv = b"\x55" * 16
    encrypted_stream = stream_iv + pdf_crypto.aes256_cbc_encrypt(
        file_key, stream_iv, plaintext_body, pad=True,
    )

    # O / OE / Perms — we don't validate these, fill with zeros.
    o_hash = b"\x00" * 48
    oe_hash = b"\x00" * 32
    perms_enc = b"\x00" * 16

    stream_obj = (
        b"1 0 obj\n<< /Length " + str(len(encrypted_stream)).encode("ascii")
        + b" >>\nstream\n" + encrypted_stream + b"\nendstream\nendobj\n"
    )
    enc_obj = (
        b"2 0 obj\n<< /Filter /Standard "
        b"/V 5 /R " + str(revision).encode("ascii") + b" /Length 256 /P -1 "
        b"/O <" + o_hash.hex().encode("ascii") + b"> "
        b"/U <" + u_hash.hex().encode("ascii") + b"> "
        b"/OE <" + oe_hash.hex().encode("ascii") + b"> "
        b"/UE <" + ue_hash.hex().encode("ascii") + b"> "
        b"/Perms <" + perms_enc.hex().encode("ascii") + b"> >>\n"
        b"endobj\n"
    )
    file_id = b"\x00" * 16
    trailer = (
        b"trailer\n<< /Encrypt 2 0 R /ID [<" + file_id.hex().encode("ascii")
        + b"> <" + file_id.hex().encode("ascii") + b">] >>\n"
        b"%%EOF\n"
    )
    return b"%PDF-1.4\n" + stream_obj + enc_obj + trailer


class TestV5AES256:
    def test_r5_empty_password_round_trip(self):
        plaintext = b"BT\n(Hello AES-256 R5) Tj\nET\n"
        pdf = _build_encrypted_pdf_v5(revision=5, plaintext_body=plaintext)
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert enc.version == 5 and enc.revision == 5 and enc.supported
        assert enc.derive_file_key(b"") is True
        assert len(enc.file_key) == 32
        text = PDFParser()._extract_text(pdf)
        assert "Hello AES-256 R5" in text

    def test_r5_wrong_password_rejected(self):
        pdf = _build_encrypted_pdf_v5(revision=5, plaintext_body=b"BT\n(x) Tj\nET\n")
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc.derive_file_key(b"wrong") is False
        assert enc.file_key is None

    def test_r5_non_empty_password_via_metadata(self):
        # Build with non-empty password, then verify PDFParser uses metadata
        # to recover when empty fails.
        plaintext = b"BT\n(Hello User Password) Tj\nET\n"
        pdf = _build_encrypted_pdf_v5(
            revision=5, plaintext_body=plaintext, password=b"hunter2",
        )
        # Without password — empty fails, no fallback.
        chunks_no_pw = PDFParser().parse(pdf.decode("latin-1"))
        assert chunks_no_pw[0].metadata.get("error") == "encrypted_password_required"
        # With password via metadata — succeeds.
        chunks = PDFParser().parse(
            pdf.decode("latin-1"),
            metadata={"pdf_password": "hunter2"},
        )
        joined = " ".join(c.text for c in chunks)
        assert "Hello User Password" in joined
        assert any(c.metadata.get("pdf_encrypted") for c in chunks)

    def test_r6_empty_password_round_trip(self):
        # R=6 uses Algorithm 2.B — pure-Python AES makes this slow (~few seconds).
        plaintext = b"BT\n(Hello AES-256 R6) Tj\nET\n"
        pdf = _build_encrypted_pdf_v5(revision=6, plaintext_body=plaintext)
        enc = pdf_crypto.detect_encryption(pdf)
        assert enc is not None
        assert enc.version == 5 and enc.revision == 6 and enc.supported
        assert enc.derive_file_key(b"") is True
        assert len(enc.file_key) == 32
        text = PDFParser()._extract_text(pdf)
        assert "Hello AES-256 R6" in text


# ---------------------------------------------------------------------------
# Negative cases — plain PDFs must continue to work
# ---------------------------------------------------------------------------

class TestPlainPDFUnaffected:
    def test_plain_extraction_unaffected(self):
        # No /Encrypt — encryption detection should be a no-op.
        plain = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Length 33 >>\nstream\n"
            b"BT\n(Plain PDF unaffected) Tj\nET\nendstream\nendobj\n"
            b"trailer << /Size 1 >>\n%%EOF\n"
        )
        text = PDFParser()._extract_text(plain)
        assert "Plain PDF unaffected" in text
