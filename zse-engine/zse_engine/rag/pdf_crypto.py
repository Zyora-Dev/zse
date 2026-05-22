"""Pure-Python PDF Standard Security Handler — zero deps.

Implements just enough of PDF 1.7 §7.6 to decrypt the common cases seen in
the wild:

- V=1, R=2  (RC4-40)
- V=2, R=3  (RC4-128)
- V=4, R=4  (RC4-128 or AES-128 via crypt filters)

Restricted to **empty user password** — covers permissions-locked PDFs which
account for the vast majority of "encrypted" PDFs encountered during RAG
ingest. V=5 (AES-256, R=5/R=6) is detected but reported as unsupported so
the caller can fall back gracefully instead of returning garbage.

All primitives (MD5, RC4, AES-128 CBC) are implemented with stdlib only
(`hashlib` for MD5; RC4 and AES are pure Python here).
"""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Standard PDF password padding (PDF 1.7 §7.6.3.3, table 21)
# ---------------------------------------------------------------------------

_PAD = bytes([
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41,
    0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80,
    0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
])


def _pad_password(password: bytes) -> bytes:
    """Pad/truncate to exactly 32 bytes per PDF spec."""
    if len(password) >= 32:
        return password[:32]
    return password + _PAD[: 32 - len(password)]


# ---------------------------------------------------------------------------
# RC4 stream cipher (Alleged RC4 / ARCFOUR)
# ---------------------------------------------------------------------------

def rc4(key: bytes, data: bytes) -> bytes:
    """RC4 keystream applied to `data`. Symmetric — same call decrypts."""
    s = list(range(256))
    j = 0
    keylen = len(key)
    for i in range(256):
        j = (j + s[i] + key[i % keylen]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray(len(data))
    i = j = 0
    for k, byte in enumerate(data):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out[k] = byte ^ s[(s[i] + s[j]) & 0xFF]
    return bytes(out)


# ---------------------------------------------------------------------------
# AES-128 (decrypt only, CBC mode with PKCS#7 padding)
#
# Compact reference implementation. Used only when a PDF declares /V 4 with
# an AES crypt filter. For everything else (RC4) we never touch this code.
# ---------------------------------------------------------------------------

_AES_SBOX = bytes([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
])
_AES_INV_SBOX = bytes(_AES_SBOX.index(b) for b in range(256))
_AES_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36)


def _xtime(b: int) -> int:
    return ((b << 1) ^ 0x1b) & 0xFF if b & 0x80 else (b << 1) & 0xFF


def _gmul(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a = _xtime(a)
        b >>= 1
    return p


def _aes_expand_key_128(key: bytes) -> list:
    """Expand a 16-byte key into 11 round keys (16 bytes each = 176 bytes)."""
    assert len(key) == 16
    rk = bytearray(key)
    for i in range(10):
        # Last 4 bytes
        t0, t1, t2, t3 = rk[-4], rk[-3], rk[-2], rk[-1]
        # RotWord + SubWord + Rcon
        t0, t1, t2, t3 = (
            _AES_SBOX[t1] ^ _AES_RCON[i],
            _AES_SBOX[t2],
            _AES_SBOX[t3],
            _AES_SBOX[t0],
        )
        for j in range(4):
            base = len(rk) - 16
            rk.extend([
                rk[base] ^ t0,
                rk[base + 1] ^ t1,
                rk[base + 2] ^ t2,
                rk[base + 3] ^ t3,
            ])
            t0 = rk[-4]
            t1 = rk[-3]
            t2 = rk[-2]
            t3 = rk[-1]
    return [bytes(rk[i:i + 16]) for i in range(0, 176, 16)]


def _aes_add_round_key(state: bytearray, rk: bytes) -> None:
    for i in range(16):
        state[i] ^= rk[i]


def _aes_inv_sub_bytes(state: bytearray) -> None:
    for i in range(16):
        state[i] = _AES_INV_SBOX[state[i]]


def _aes_inv_shift_rows(state: bytearray) -> None:
    # Column-major layout. Inverse of left-rotate is right-rotate.
    state[1], state[5], state[9], state[13] = state[13], state[1], state[5], state[9]
    state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
    state[3], state[7], state[11], state[15] = state[7], state[11], state[15], state[3]


def _aes_inv_mix_columns(state: bytearray) -> None:
    for c in range(4):
        o = c * 4
        a, b, cc, d = state[o], state[o + 1], state[o + 2], state[o + 3]
        state[o] = _gmul(a, 14) ^ _gmul(b, 11) ^ _gmul(cc, 13) ^ _gmul(d, 9)
        state[o + 1] = _gmul(a, 9) ^ _gmul(b, 14) ^ _gmul(cc, 11) ^ _gmul(d, 13)
        state[o + 2] = _gmul(a, 13) ^ _gmul(b, 9) ^ _gmul(cc, 14) ^ _gmul(d, 11)
        state[o + 3] = _gmul(a, 11) ^ _gmul(b, 13) ^ _gmul(cc, 9) ^ _gmul(d, 14)


def _aes_decrypt_block_128(block: bytes, round_keys: list) -> bytes:
    state = bytearray(block)
    _aes_add_round_key(state, round_keys[10])
    for r in range(9, 0, -1):
        _aes_inv_shift_rows(state)
        _aes_inv_sub_bytes(state)
        _aes_add_round_key(state, round_keys[r])
        _aes_inv_mix_columns(state)
    _aes_inv_shift_rows(state)
    _aes_inv_sub_bytes(state)
    _aes_add_round_key(state, round_keys[0])
    return bytes(state)


def aes128_cbc_decrypt(key: bytes, data: bytes) -> bytes:
    """AES-128 CBC decrypt. PDF prefixes ciphertext with a 16-byte IV.

    Applies PKCS#7 unpadding on the last block. Returns empty bytes on any
    structural error rather than raising.
    """
    return aes_cbc_decrypt(_aes_expand_key_128(key), data)


# ---------------------------------------------------------------------------
# AES-256 + forward AES (needed for V=5 / R=5 / R=6 — Acrobat 9+)
# ---------------------------------------------------------------------------

def _aes_sub_bytes(state: bytearray) -> None:
    for i in range(16):
        state[i] = _AES_SBOX[state[i]]


def _aes_shift_rows(state: bytearray) -> None:
    state[1], state[5], state[9], state[13] = state[5], state[9], state[13], state[1]
    state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
    state[3], state[7], state[11], state[15] = state[15], state[3], state[7], state[11]


def _aes_mix_columns(state: bytearray) -> None:
    for c in range(4):
        o = c * 4
        a, b, cc, d = state[o], state[o + 1], state[o + 2], state[o + 3]
        state[o] = _gmul(a, 2) ^ _gmul(b, 3) ^ cc ^ d
        state[o + 1] = a ^ _gmul(b, 2) ^ _gmul(cc, 3) ^ d
        state[o + 2] = a ^ b ^ _gmul(cc, 2) ^ _gmul(d, 3)
        state[o + 3] = _gmul(a, 3) ^ b ^ cc ^ _gmul(d, 2)


def _aes_encrypt_block(block: bytes, round_keys: list) -> bytes:
    """Forward AES block encrypt. Works for AES-128 (Nr=10) and AES-256 (Nr=14)
    by looking at len(round_keys)."""
    nr = len(round_keys) - 1
    state = bytearray(block)
    _aes_add_round_key(state, round_keys[0])
    for r in range(1, nr):
        _aes_sub_bytes(state)
        _aes_shift_rows(state)
        _aes_mix_columns(state)
        _aes_add_round_key(state, round_keys[r])
    _aes_sub_bytes(state)
    _aes_shift_rows(state)
    _aes_add_round_key(state, round_keys[nr])
    return bytes(state)


def _aes_decrypt_block(block: bytes, round_keys: list) -> bytes:
    """Generic AES block decrypt (Nr inferred from round-key count)."""
    nr = len(round_keys) - 1
    state = bytearray(block)
    _aes_add_round_key(state, round_keys[nr])
    for r in range(nr - 1, 0, -1):
        _aes_inv_shift_rows(state)
        _aes_inv_sub_bytes(state)
        _aes_add_round_key(state, round_keys[r])
        _aes_inv_mix_columns(state)
    _aes_inv_shift_rows(state)
    _aes_inv_sub_bytes(state)
    _aes_add_round_key(state, round_keys[0])
    return bytes(state)


def _aes_expand_key_256(key: bytes) -> list:
    """Expand 32-byte key into 15 round keys (240 bytes total)."""
    assert len(key) == 32
    rk = bytearray(key)
    rcon_idx = 0
    while len(rk) < 240:
        t = list(rk[-4:])
        if len(rk) % 32 == 0:
            t = [
                _AES_SBOX[t[1]] ^ _AES_RCON[rcon_idx],
                _AES_SBOX[t[2]],
                _AES_SBOX[t[3]],
                _AES_SBOX[t[0]],
            ]
            rcon_idx += 1
        elif len(rk) % 32 == 16:
            t = [_AES_SBOX[b] for b in t]
        base = len(rk) - 32
        for j in range(4):
            rk.append(rk[base + j] ^ t[j])
    return [bytes(rk[i:i + 16]) for i in range(0, 240, 16)]


def aes_cbc_encrypt(round_keys: list, iv: bytes, data: bytes, pad: bool = True) -> bytes:
    """AES-CBC encrypt with optional PKCS#7 padding. `iv` is NOT prepended to
    the output (caller decides — PDF stream encryption prepends, file-key
    encryption does not)."""
    if pad:
        n = 16 - (len(data) % 16)
        data = data + bytes([n]) * n
    elif len(data) % 16 != 0:
        raise ValueError("data must be 16-aligned when pad=False")
    out = bytearray()
    prev = iv
    for i in range(0, len(data), 16):
        block = bytes(a ^ b for a, b in zip(data[i:i + 16], prev))
        ct = _aes_encrypt_block(block, round_keys)
        out.extend(ct)
        prev = ct
    return bytes(out)


def aes_cbc_decrypt(
    round_keys: list,
    data: bytes,
    iv: Optional[bytes] = None,
    strip_pad: bool = True,
) -> bytes:
    """AES-CBC decrypt. If `iv` is None, the first 16 bytes of `data` are
    treated as the IV (PDF stream encryption mode). Otherwise `data` is pure
    ciphertext (PDF file-key encryption mode, IV is usually zero)."""
    if iv is None:
        if len(data) < 32 or len(data) % 16 != 0:
            return b""
        iv = data[:16]
        ct = data[16:]
    else:
        if len(data) % 16 != 0:
            return b""
        ct = data
    out = bytearray()
    prev = iv
    for i in range(0, len(ct), 16):
        block = ct[i:i + 16]
        dec = _aes_decrypt_block(block, round_keys)
        out.extend(bytes(a ^ b for a, b in zip(dec, prev)))
        prev = block
    if strip_pad and out:
        pad = out[-1]
        if 1 <= pad <= 16 and all(b == pad for b in out[-pad:]):
            out = out[:-pad]
    return bytes(out)


def aes128_cbc_encrypt(key: bytes, iv: bytes, data: bytes, pad: bool = True) -> bytes:
    return aes_cbc_encrypt(_aes_expand_key_128(key), iv, data, pad=pad)


def aes256_cbc_encrypt(key: bytes, iv: bytes, data: bytes, pad: bool = True) -> bytes:
    return aes_cbc_encrypt(_aes_expand_key_256(key), iv, data, pad=pad)


def aes256_cbc_decrypt(
    key: bytes,
    data: bytes,
    iv: Optional[bytes] = None,
    strip_pad: bool = True,
) -> bytes:
    return aes_cbc_decrypt(_aes_expand_key_256(key), data, iv=iv, strip_pad=strip_pad)


def _alg_2b(password: bytes, salt: bytes, u_value: bytes) -> bytes:
    """ISO 32000-2 §7.6.4.3.3 Algorithm 2.B — R=6 hash computation.

    Used for both user/owner password validation and file-key wrapping in
    AES-256 / R=6 (Acrobat X and later).
    """
    K = hashlib.sha256(password + salt + u_value).digest()
    round_no = 0
    while True:
        K1 = (password + K + u_value) * 64
        rk = _aes_expand_key_128(K[:16])
        E = aes_cbc_encrypt(rk, K[16:32], K1, pad=False)
        s = sum(E[:16]) % 3
        if s == 0:
            K = hashlib.sha256(E).digest()
        elif s == 1:
            K = hashlib.sha384(E).digest()
        else:
            K = hashlib.sha512(E).digest()
        if round_no >= 64 and E[-1] <= round_no - 32:
            break
        round_no += 1
        if round_no > 1024:
            # Safety guard — algorithm always terminates well before this.
            break
    return K[:32]


# ---------------------------------------------------------------------------
# PDF object / trailer parsing helpers (lightweight — regex on raw bytes)
# ---------------------------------------------------------------------------

_RE_TRAILER_BLOCK = re.compile(rb"trailer\s*<<(.*?)>>", re.DOTALL)
_RE_ENCRYPT_REF = re.compile(rb"/Encrypt\s+(\d+)\s+(\d+)\s+R")
_RE_ID_PAIR = re.compile(rb"/ID\s*\[\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\]")
_RE_OBJ = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj", re.DOTALL)


def _find_trailer_dict(raw: bytes) -> Optional[bytes]:
    """Return the bytes of the trailer dict (between << and >>), or None."""
    # Prefer the last trailer block — PDFs may have multiple (linearized).
    matches = list(_RE_TRAILER_BLOCK.finditer(raw))
    if not matches:
        # Cross-reference stream form (PDF 1.5): /Encrypt lives in the XRef
        # stream's dict. Fall back to scanning the whole file for it.
        return raw
    return matches[-1].group(1)


def _find_id_pair(raw: bytes) -> Optional[Tuple[bytes, bytes]]:
    m = _RE_ID_PAIR.search(raw)
    if not m:
        return None
    try:
        return bytes.fromhex(m.group(1).decode("ascii")), bytes.fromhex(
            m.group(2).decode("ascii")
        )
    except Exception:
        return None


def _find_encrypt_dict(raw: bytes, obj_num: int) -> Optional[bytes]:
    """Locate the /Encrypt indirect object and return its dict bytes.

    Tries the classic top-level form (``N 0 obj << ... >>``) first; if not
    found, falls back to scanning PDF 1.5+ compressed object streams
    (``/Type /ObjStm``), since modern writers (qpdf, Word 2019+, recent
    LaTeX) routinely tuck the /Encrypt dict inside one.
    """
    pat = re.compile(
        rb"\b" + str(obj_num).encode("ascii") + rb"\s+\d+\s+obj\s*<<(.*?)>>",
        re.DOTALL,
    )
    m = pat.search(raw)
    if m:
        return m.group(1)

    # PDF 1.5 compressed object stream fallback.
    body = _resolve_in_object_streams(raw, obj_num)
    if body is None:
        return None
    m = re.search(rb"<<(.*?)>>", body, re.DOTALL)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# PDF 1.5+ compressed object stream (/ObjStm) support
# ---------------------------------------------------------------------------

def _iter_objstm_streams(raw: bytes) -> Iterable[Tuple[bytes, bytes]]:
    """Yield (header_dict_bytes, decoded_body) for every /Type /ObjStm stream
    we can locate by raw byte scan. FlateDecode is the only filter the spec
    permits on object streams in practice; we honour it and skip otherwise.
    """
    i = 0
    n = len(raw)
    while True:
        stream_start = raw.find(b"stream\r\n", i)
        if stream_start == -1:
            stream_start = raw.find(b"stream\n", i)
        if stream_start == -1:
            return
        # Resolve where the stream body actually begins.
        if raw[stream_start + 6:stream_start + 8] == b"\r\n":
            data_start = stream_start + 8
        else:
            data_start = stream_start + 7
        stream_end = raw.find(b"endstream", data_start)
        if stream_end == -1:
            return

        # Look at the preceding object dict.
        header_start = raw.rfind(b"<<", max(0, stream_start - 4000), stream_start)
        if header_start == -1:
            i = stream_end + 9
            continue
        header = raw[header_start:stream_start]
        if b"/Type /ObjStm" not in header and b"/Type/ObjStm" not in header:
            i = stream_end + 9
            continue

        body = raw[data_start:stream_end]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        elif body.endswith(b"\n") or body.endswith(b"\r"):
            body = body[:-1]

        # Only FlateDecode is permitted on /ObjStm. If absent, treat as raw.
        if b"/FlateDecode" in header or b"/Fl " in header or b"/Fl\n" in header:
            try:
                body = zlib.decompress(body)
            except Exception:
                i = stream_end + 9
                continue

        yield header, body
        i = stream_end + 9


def _resolve_in_object_streams(raw: bytes, obj_num: int) -> Optional[bytes]:
    """Return the raw body bytes of object ``obj_num`` if it lives inside any
    /ObjStm in this PDF, else None.

    Object streams contain (at /First bytes from the start) a small index of
    ``objnum offset`` integer pairs, followed by the concatenated object
    bodies. We parse the index, locate ``obj_num``, and return its slice.
    """
    target = str(obj_num).encode("ascii")
    for header, body in _iter_objstm_streams(raw):
        n_match = re.search(rb"/N\s+(\d+)", header)
        first_match = re.search(rb"/First\s+(\d+)", header)
        if not n_match or not first_match:
            continue
        try:
            count = int(n_match.group(1))
            first_offset = int(first_match.group(1))
        except ValueError:
            continue
        if first_offset < 0 or first_offset > len(body) or count <= 0:
            continue

        # Tokenize the index: 2*count integers (objnum, byteoffset).
        index_bytes = body[:first_offset]
        nums = re.findall(rb"-?\d+", index_bytes)
        if len(nums) < 2 * count:
            continue
        entries: List[Tuple[int, int]] = []
        ok = True
        for k in range(count):
            try:
                entries.append((int(nums[2 * k]), int(nums[2 * k + 1])))
            except ValueError:
                ok = False
                break
        if not ok:
            continue

        # Locate the target object.
        for k, (num, offset) in enumerate(entries):
            if num != obj_num:
                continue
            start = first_offset + offset
            if k + 1 < len(entries):
                end = first_offset + entries[k + 1][1]
            else:
                end = len(body)
            if 0 <= start < end <= len(body):
                return body[start:end]
        # Quick prefilter: if the obj_num doesn't appear in the index at all,
        # skip the (already-done) linear scan.
        if target not in index_bytes:
            continue
    return None


def _dict_int(dict_bytes: bytes, name: bytes, default: int = 0) -> int:
    m = re.search(rb"/" + name + rb"\s+(-?\d+)", dict_bytes)
    return int(m.group(1)) if m else default


def _dict_hex(dict_bytes: bytes, name: bytes) -> Optional[bytes]:
    """Extract /Name <hex> or /Name (literal) from a dict."""
    # Hex form.
    m = re.search(rb"/" + name + rb"\s*<([0-9A-Fa-f\s]+)>", dict_bytes)
    if m:
        try:
            return bytes.fromhex(re.sub(rb"\s+", b"", m.group(1)).decode("ascii"))
        except Exception:
            return None
    # Literal-string form. Decode PDF string escapes (octal + simple).
    m = re.search(rb"/" + name + rb"\s*\(((?:[^()\\]|\\.)*)\)", dict_bytes)
    if m:
        return _decode_pdf_literal(m.group(1))
    return None


def _decode_pdf_literal(raw: bytes) -> bytes:
    """Decode a PDF literal-string body (between '(' and ')')."""
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c != 0x5C:  # not backslash
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        nxt = raw[i]
        if nxt in b"nrtbf":
            out.append({b"n": 0x0A, b"r": 0x0D, b"t": 0x09, b"b": 0x08, b"f": 0x0C}[bytes([nxt])])
            i += 1
        elif nxt in b"()\\":
            out.append(nxt)
            i += 1
        elif 0x30 <= nxt <= 0x37:  # octal
            octal = bytes([nxt])
            i += 1
            for _ in range(2):
                if i < n and 0x30 <= raw[i] <= 0x37:
                    octal += bytes([raw[i]])
                    i += 1
                else:
                    break
            out.append(int(octal, 8) & 0xFF)
        else:
            out.append(nxt)
            i += 1
    return bytes(out)


# ---------------------------------------------------------------------------
# Encryption detection + key derivation
# ---------------------------------------------------------------------------

class PDFEncryption:
    """Parsed /Encrypt dict + derived file key for empty user password."""

    def __init__(
        self,
        version: int,
        revision: int,
        length_bits: int,
        permissions: int,
        owner_hash: bytes,
        user_hash: bytes,
        file_id: bytes,
        stream_filter: str = "V2",
        string_filter: str = "V2",
        encrypt_metadata: bool = True,
        ue_hash: bytes = b"",
        oe_hash: bytes = b"",
        perms_enc: bytes = b"",
    ):
        self.version = version
        self.revision = revision
        self.length_bits = length_bits
        self.key_length = length_bits // 8
        self.permissions = permissions
        self.owner_hash = owner_hash
        self.user_hash = user_hash
        self.file_id = file_id
        self.stream_filter = stream_filter
        self.string_filter = string_filter
        self.encrypt_metadata = encrypt_metadata
        self.ue_hash = ue_hash
        self.oe_hash = oe_hash
        self.perms_enc = perms_enc
        self.file_key: Optional[bytes] = None

    @property
    def supported(self) -> bool:
        if self.version in (1, 2, 4) and self.revision in (2, 3, 4):
            return True
        # V=5 (AES-256) — both R=5 (Acrobat 9, deprecated) and R=6 (Acrobat X+).
        if self.version == 5 and self.revision in (5, 6):
            return len(self.user_hash) >= 48 and len(self.ue_hash) == 32
        return False

    @property
    def uses_aes_for_streams(self) -> bool:
        if self.version == 5:
            return True
        return self.version == 4 and self.stream_filter.upper() == "AESV2"

    @property
    def uses_aes_for_strings(self) -> bool:
        if self.version == 5:
            return True
        return self.version == 4 and self.string_filter.upper() == "AESV2"

    def derive_file_key(self, password: bytes = b"") -> bool:
        """Try to derive the file encryption key for the given user password.

        Returns True on success, False if the password is wrong / unsupported.
        """
        if not self.supported:
            return False
        if self.version == 5:
            return self._derive_file_key_v5(password)
        padded = _pad_password(password)
        md5 = hashlib.md5()
        md5.update(padded)
        md5.update(self.owner_hash)
        md5.update(struct.pack("<i", self.permissions))
        md5.update(self.file_id)
        if self.revision >= 4 and not self.encrypt_metadata:
            md5.update(b"\xff\xff\xff\xff")
        key = md5.digest()
        if self.revision >= 3:
            for _ in range(50):
                key = hashlib.md5(key[: self.key_length]).digest()
        key = key[: self.key_length]
        if self._verify_user_password(key, padded):
            self.file_key = key
            return True
        return False

    def _derive_file_key_v5(self, password: bytes) -> bool:
        """ISO 32000-2 §7.6.4.3.2/.3 — V=5 user-password path.

        /U layout (48 bytes total):
          [0:32]   user hash
          [32:40]  validation salt
          [40:48]  key salt
        File key = AES-256-CBC-decrypt(/UE, key=hash(password||key_salt), IV=0).
        """
        if len(self.user_hash) < 48 or len(self.ue_hash) != 32:
            return False
        val_salt = self.user_hash[32:40]
        key_salt = self.user_hash[40:48]
        if self.revision == 5:
            if hashlib.sha256(password + val_salt).digest() != self.user_hash[:32]:
                return False
            fk_key = hashlib.sha256(password + key_salt).digest()
        elif self.revision == 6:
            if _alg_2b(password, val_salt, b"") != self.user_hash[:32]:
                return False
            fk_key = _alg_2b(password, key_salt, b"")
        else:
            return False
        fk = aes256_cbc_decrypt(
            fk_key, self.ue_hash, iv=b"\x00" * 16, strip_pad=False
        )
        if len(fk) != 32:
            return False
        self.file_key = fk
        return True

    def _verify_user_password(self, key: bytes, padded_password: bytes) -> bool:
        if self.revision == 2:
            expected = rc4(key, _PAD)
            return expected == self.user_hash
        # R3 / R4: U is MD5(PAD + ID) then RC4-iterated, then padded with junk.
        md5 = hashlib.md5()
        md5.update(_PAD)
        md5.update(self.file_id)
        u = md5.digest()
        u = rc4(key, u)
        for i in range(1, 20):
            xor_key = bytes(b ^ i for b in key)
            u = rc4(xor_key, u)
        # Spec says only the first 16 bytes must match.
        return u[:16] == self.user_hash[:16]

    # ------------------------------------------------------------------
    # Per-object key derivation + stream/string decryption
    # ------------------------------------------------------------------

    def _object_key(self, obj_num: int, gen_num: int, for_aes: bool) -> bytes:
        if self.file_key is None:
            raise RuntimeError("file key not derived")
        md5 = hashlib.md5()
        md5.update(self.file_key)
        md5.update(struct.pack("<I", obj_num)[:3])
        md5.update(struct.pack("<I", gen_num)[:2])
        if for_aes:
            md5.update(b"sAlT")
        n = min(self.key_length + 5, 16)
        return md5.digest()[:n]

    def decrypt_stream(self, obj_num: int, gen_num: int, data: bytes) -> bytes:
        if self.file_key is None:
            return b""
        # AESV3 (V=5) uses the file key directly — no per-object derivation.
        if self.version == 5:
            return aes256_cbc_decrypt(self.file_key, data)
        use_aes = self.uses_aes_for_streams
        key = self._object_key(obj_num, gen_num, for_aes=use_aes)
        if use_aes:
            return aes128_cbc_decrypt(key, data)
        return rc4(key, data)


# ---------------------------------------------------------------------------
# Public detection + parsing
# ---------------------------------------------------------------------------

def detect_encryption(raw: bytes) -> Optional[PDFEncryption]:
    """Parse the /Encrypt dict from a PDF. Returns None if the file is not
    encrypted, or if the encryption is malformed enough that we can't tell.
    """
    trailer = _find_trailer_dict(raw) or raw
    enc_ref = _RE_ENCRYPT_REF.search(trailer)
    if not enc_ref:
        return None

    obj_num = int(enc_ref.group(1))
    enc_dict = _find_encrypt_dict(raw, obj_num)
    if enc_dict is None:
        return None

    file_id_pair = _find_id_pair(raw)
    file_id = file_id_pair[0] if file_id_pair else b""

    version = _dict_int(enc_dict, b"V", 0)
    revision = _dict_int(enc_dict, b"R", 0)
    length_bits = _dict_int(enc_dict, b"Length", 40)
    if version >= 2 and length_bits == 40:
        # /Length defaults to 40 only for V=1.
        length_bits = 128
    if version == 5:
        length_bits = 256
    permissions = _dict_int(enc_dict, b"P", -1)
    owner_hash = _dict_hex(enc_dict, b"O") or b""
    user_hash = _dict_hex(enc_dict, b"U") or b""
    ue_hash = _dict_hex(enc_dict, b"UE") or b""
    oe_hash = _dict_hex(enc_dict, b"OE") or b""
    perms_enc = _dict_hex(enc_dict, b"Perms") or b""

    # V<5 requires a file ID for the key derivation. V=5 does not.
    if version != 5 and not file_id_pair:
        return None

    # Crypt filter name lookup (V=4 / V=5).
    if version == 5:
        stream_filter = "AESV3"
        string_filter = "AESV3"
    else:
        stream_filter = "V2"
        string_filter = "V2"
    if version in (4, 5):
        stm = re.search(rb"/StmF\s*/(\w+)", enc_dict)
        if stm:
            cf_name = stm.group(1)
            cf = re.search(
                rb"/" + cf_name + rb"\s*<<[^>]*?/CFM\s*/(\w+)",
                enc_dict,
                re.DOTALL,
            )
            if cf:
                stream_filter = cf.group(1).decode("ascii", errors="ignore")
        strf = re.search(rb"/StrF\s*/(\w+)", enc_dict)
        if strf:
            cf_name = strf.group(1)
            cf = re.search(
                rb"/" + cf_name + rb"\s*<<[^>]*?/CFM\s*/(\w+)",
                enc_dict,
                re.DOTALL,
            )
            if cf:
                string_filter = cf.group(1).decode("ascii", errors="ignore")

    em = re.search(rb"/EncryptMetadata\s+(true|false)", enc_dict)
    encrypt_metadata = (em is None) or (em.group(1) == b"true")

    return PDFEncryption(
        version=version,
        revision=revision,
        length_bits=length_bits,
        permissions=permissions,
        owner_hash=owner_hash,
        user_hash=user_hash,
        file_id=file_id,
        stream_filter=stream_filter,
        string_filter=string_filter,
        encrypt_metadata=encrypt_metadata,
        ue_hash=ue_hash,
        oe_hash=oe_hash,
        perms_enc=perms_enc,
    )


def find_stream_object_numbers(raw: bytes) -> Dict[int, Tuple[int, int]]:
    """Map stream-data-start offsets to their enclosing (obj_num, gen_num).

    Lets the parser look up which object owns each stream so we can derive
    the per-object decryption key.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", raw):
        obj_num = int(m.group(1))
        gen_num = int(m.group(2))
        # Find the first 'stream' marker after this header, before the next 'endobj'.
        body_start = m.end()
        endobj = raw.find(b"endobj", body_start)
        if endobj == -1:
            continue
        sm = raw.find(b"stream", body_start, endobj)
        if sm == -1:
            continue
        if raw[sm + 6:sm + 8] == b"\r\n":
            data_start = sm + 8
        elif raw[sm + 6:sm + 7] in (b"\r", b"\n"):
            data_start = sm + 7
        else:
            continue
        out[data_start] = (obj_num, gen_num)
    return out
