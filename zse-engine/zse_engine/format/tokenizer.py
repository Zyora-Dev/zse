"""ZSE BPE Tokenizer — Zero-dependency tokenizer extracted from HuggingFace.

Extracts vocab + merges from HF tokenizer.json, provides runtime encode/decode.
No sentencepiece, no tiktoken — pure Python BPE.

Supports:
    - Byte-level BPE (GPT-2 / Llama style)
    - Pre-tokenization via regex splitting
    - Special tokens (bos, eos, pad, unk)
    - Serialization to/from ZSE binary format
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from zse_engine.format import serializer


# --------------------------------------------------------------------------- #
# Byte-level BPE helpers (GPT-2 style byte encoding)
# --------------------------------------------------------------------------- #

def _bytes_to_unicode() -> Dict[int, str]:
    """Build the byte-to-unicode mapping used by GPT-2 / Llama tokenizers.

    Printable ASCII bytes map to themselves; other bytes are shifted to
    the Unicode range starting at 256 so every byte has a visible char.
    """
    bs = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('\xa1'), ord('\xac') + 1))
        + list(range(ord('\xae'), ord('\xff') + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


# --------------------------------------------------------------------------- #
# Pre-tokenization regex (GPT-2 / Llama style)
# --------------------------------------------------------------------------- #

# Splits on contractions, words, numbers, and non-whitespace non-word chars
_GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
)


# --------------------------------------------------------------------------- #
# Tokenizer dataclass
# --------------------------------------------------------------------------- #

@dataclass
class SpecialTokens:
    """Special token IDs."""
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
    unk_id: int = 0

    def to_dict(self) -> dict:
        return {
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SpecialTokens':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        vocab: token string → token ID
        merges: ordered list of BPE merge pairs
        special_tokens: special token IDs
        added_tokens: extra tokens added after base vocab (e.g. <|im_start|>)
    """
    vocab: Dict[str, int] = field(default_factory=dict)
    merges: List[Tuple[str, str]] = field(default_factory=list)
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    added_tokens: Dict[str, int] = field(default_factory=dict)

    # --- Internal caches (rebuilt on load) ---
    _id_to_token: Dict[int, str] = field(default_factory=dict, repr=False)
    _bpe_ranks: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)
    _cache: Dict[str, str] = field(default_factory=dict, repr=False)
    # Added tokens pattern for fast detection
    _added_pattern: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        self._rebuild_caches()

    def _rebuild_caches(self):
        """Rebuild internal lookup structures."""
        self._id_to_token = {v: k for k, v in self.vocab.items()}
        # Also include added tokens in reverse map
        for tok, tid in self.added_tokens.items():
            self._id_to_token[tid] = tok
        self._bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}
        # Build regex for added tokens (sorted longest-first for greedy match)
        if self.added_tokens:
            sorted_toks = sorted(self.added_tokens.keys(), key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_toks]
            self._added_pattern = re.compile('|'.join(escaped))
        else:
            self._added_pattern = None

    # ------------------------------------------------------------------ #
    # BPE core
    # ------------------------------------------------------------------ #

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all adjacent symbol pairs in a word."""
        pairs = set()
        prev = word[0]
        for s in word[1:]:
            pairs.add((prev, s))
            prev = s
        return pairs

    def _bpe(self, token: str) -> str:
        """Apply BPE merges to a single pre-tokenized token."""
        if token in self._cache:
            return self._cache[token]

        word = tuple(token)
        if len(word) <= 1:
            self._cache[token] = token
            return token

        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break

            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = len(self.merges)  # higher than any valid rank
            for pair in pairs:
                rank = self._bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # No more merges applicable

            # Merge the best pair
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                # Find next occurrence of first
                found = False
                for j in range(i, len(word)):
                    if word[j] == first:
                        new_word.extend(word[i:j])
                        i = j
                        found = True
                        break
                if not found:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

        result = ' '.join(word)
        self._cache[token] = result
        return result

    # ------------------------------------------------------------------ #
    # Encode
    # ------------------------------------------------------------------ #

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token IDs
        """
        ids = []

        if add_bos:
            ids.append(self.special_tokens.bos_id)

        # Handle added tokens by splitting text around them
        if self._added_pattern:
            parts = self._added_pattern.split(text)
            matches = self._added_pattern.findall(text)

            for i, part in enumerate(parts):
                if part:
                    ids.extend(self._encode_chunk(part))
                if i < len(matches):
                    ids.append(self.added_tokens[matches[i]])
        else:
            ids.extend(self._encode_chunk(text))

        if add_eos:
            ids.append(self.special_tokens.eos_id)

        return ids

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a text chunk (no added tokens) to IDs."""
        ids = []
        # Pre-tokenize with GPT-2 regex
        for match in _GPT2_PAT.finditer(text):
            token = match.group()
            # Byte-encode: convert each byte to its unicode char
            bpe_token = ''.join(BYTE_ENCODER.get(b, chr(b)) for b in token.encode('utf-8'))
            # Apply BPE merges
            bpe_result = self._bpe(bpe_token)
            for piece in bpe_result.split(' '):
                token_id = self.vocab.get(piece)
                if token_id is not None:
                    ids.append(token_id)
                else:
                    ids.append(self.special_tokens.unk_id)
        return ids

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special: If True, skip BOS/EOS/PAD tokens

        Returns:
            Decoded string
        """
        special_ids = {
            self.special_tokens.bos_id,
            self.special_tokens.eos_id,
            self.special_tokens.pad_id,
        }

        tokens = []
        for tid in ids:
            if skip_special and tid in special_ids:
                continue
            token = self._id_to_token.get(tid)
            if token is not None:
                tokens.append(token)

        text = ''.join(tokens)

        # Byte-decode: convert unicode chars back to bytes
        byte_list = []
        for ch in text:
            if ch in BYTE_DECODER:
                byte_list.append(BYTE_DECODER[ch])
            else:
                # Fallback for chars not in the byte mapping
                byte_list.extend(ch.encode('utf-8'))

        return bytes(byte_list).decode('utf-8', errors='replace')

    # ------------------------------------------------------------------ #
    # Serialization (ZSE binary format)
    # ------------------------------------------------------------------ #

    def serialize(self) -> bytes:
        """Serialize tokenizer to ZSE binary format."""
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens.to_dict(),
            "added_tokens": self.added_tokens,
        }
        return serializer.encode(data)

    @classmethod
    def deserialize(cls, data: bytes) -> 'BPETokenizer':
        """Deserialize tokenizer from ZSE binary format."""
        d = serializer.decode(data)
        tok = cls(
            vocab=d["vocab"],
            merges=[tuple(m) for m in d["merges"]],
            special_tokens=SpecialTokens.from_dict(d.get("special_tokens", {})),
            added_tokens=d.get("added_tokens", {}),
        )
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.added_tokens)

    # ------------------------------------------------------------------ #
    # Extract from HuggingFace tokenizer.json
    # ------------------------------------------------------------------ #

    @classmethod
    def from_hf_tokenizer_json(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from a HuggingFace tokenizer.json file.

        This is the primary way to create a tokenizer for conversion.
        Only needed at convert-time, not at inference runtime.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract vocab
        model = data.get("model", {})
        vocab = model.get("vocab", {})

        # Extract merges
        raw_merges = model.get("merges", [])
        merges = []
        for m in raw_merges:
            if isinstance(m, str):
                parts = m.split(' ', 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                merges.append((m[0], m[1]))

        # Extract special tokens
        special = SpecialTokens()
        added_tokens_list = data.get("added_tokens", [])
        added_tokens = {}

        for tok_info in added_tokens_list:
            content = tok_info.get("content", "")
            tid = tok_info.get("id", 0)
            is_special = tok_info.get("special", False)

            if is_special:
                content_lower = content.lower()
                if "bos" in content_lower or content == "<s>" or "beginoftext" in content_lower.replace("|", "").replace("_", ""):
                    special.bos_id = tid
                elif ("eos" in content_lower or content == "</s>"
                      or "endoftext" in content_lower.replace("|", "").replace("_", "")
                      or content == "<|endoftext|>"):
                    special.eos_id = tid
                elif "pad" in content_lower or content == "<pad>":
                    special.pad_id = tid
                elif "unk" in content_lower or content == "<unk>":
                    special.unk_id = tid

            # Track added tokens not in base vocab
            if content not in vocab:
                added_tokens[content] = tid

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special,
            added_tokens=added_tokens,
        )

    @classmethod
    def from_hf_dir(cls, model_dir: str) -> 'BPETokenizer':
        """Load tokenizer from a HuggingFace model directory.

        Looks for tokenizer.json (fast tokenizer) first.
        Also reads tokenizer_config.json for eos/bos token IDs as fallback.
        """
        import os
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"No tokenizer.json found in {model_dir}. "
                f"ZSE requires a HuggingFace fast tokenizer (tokenizer.json)."
            )

        tok = cls.from_hf_tokenizer_json(tokenizer_path)

        # Fallback: read tokenizer_config.json for explicit eos/bos token IDs
        # This handles models like Qwen2 where eos_token = "<|endoftext|>"
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    tc = json.load(f)

                # Look up eos_token string → ID
                eos_token = tc.get("eos_token")
                if isinstance(eos_token, dict):
                    eos_token = eos_token.get("content", eos_token)
                if isinstance(eos_token, str) and eos_token:
                    eos_id = tok.vocab.get(eos_token)
                    if eos_id is None and eos_token in tok.added_tokens:
                        eos_id = tok.added_tokens[eos_token]
                    if eos_id is not None:
                        tok.special_tokens.eos_id = eos_id

                bos_token = tc.get("bos_token")
                if isinstance(bos_token, dict):
                    bos_token = bos_token.get("content", bos_token)
                if isinstance(bos_token, str) and bos_token:
                    bos_id = tok.vocab.get(bos_token)
                    if bos_id is None and bos_token in tok.added_tokens:
                        bos_id = tok.added_tokens[bos_token]
                    if bos_id is not None:
                        tok.special_tokens.bos_id = bos_id
            except Exception:
                pass  # Non-critical — pattern matching may have already worked

        return tok
