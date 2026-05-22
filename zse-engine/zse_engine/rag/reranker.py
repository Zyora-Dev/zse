"""ZSE LLM Reranker — Cross-encoder reranking via the loaded LLM.

After initial retrieval returns top-K candidates, this reranker uses the
inference LLM as a cross-encoder to score (query, doc) relevance.

Method: prompt-based scoring with log-probability extraction.
We prompt: "Query: <q>\\nDocument: <d>\\nRelevant? Answer Yes or No.\\n"
then take logprob(YES_TOKEN) - logprob(NO_TOKEN) from the final logits.

Cost: one prefill per (query, doc) pair. For K=20 candidates this is acceptable
as a post-retrieval refinement step. Use only when high precision matters.

Trade-off vs dedicated cross-encoder models: slightly weaker quality (LLM not
specifically tuned for ranking) but zero extra weights / VRAM.
"""

import math
import struct
from typing import List, Tuple, Optional, Dict


_PROMPT_TEMPLATE = (
    "You are a relevance grader. Decide if the document is relevant to the query.\n"
    "Query: {q}\n"
    "Document: {d}\n"
    "Answer with a single word, Yes or No.\nAnswer:"
)


class LLMReranker:
    """Rerank (chunk_id, doc_text) candidates using LLM logprob scoring.

    Args:
        model_runner: ZSE ModelRunner instance (must expose prefill + hidden_size)
        tokenizer: ZSE tokenizer (for encoding the rerank prompt)
        doc_max_tokens: Truncate document text to this many tokens for the prompt
    """

    _RERANK_SLOT_BASE = 2_000_000_000
    _RERANK_SLOT_COUNT = 16

    def __init__(self, model_runner, tokenizer, doc_max_tokens: int = 384):
        self._runner = model_runner
        self._tokenizer = tokenizer
        self._doc_max_tokens = doc_max_tokens
        self._slot_cursor = 0
        # Resolve "Yes" / "No" token IDs (lazy — first call)
        self._yes_token: Optional[int] = None
        self._no_token: Optional[int] = None

    def _next_slot(self) -> int:
        s = self._RERANK_SLOT_BASE + self._slot_cursor
        self._slot_cursor = (self._slot_cursor + 1) % self._RERANK_SLOT_COUNT
        return s

    def _resolve_yes_no_tokens(self):
        """Resolve the token IDs for 'Yes' and 'No' (cached after first call)."""
        if self._yes_token is not None and self._no_token is not None:
            return
        if self._tokenizer is None:
            return
        # Try several common encodings. We want a single-token mapping.
        for cand in (" Yes", "Yes", " yes"):
            try:
                ids = self._tokenizer.encode(cand) if hasattr(self._tokenizer, "encode") else self._tokenizer(cand)
                if ids and len(ids) >= 1:
                    self._yes_token = ids[-1]
                    break
            except Exception:
                continue
        for cand in (" No", "No", " no"):
            try:
                ids = self._tokenizer.encode(cand) if hasattr(self._tokenizer, "encode") else self._tokenizer(cand)
                if ids and len(ids) >= 1:
                    self._no_token = ids[-1]
                    break
            except Exception:
                continue

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text by token count using tokenizer."""
        if not self._tokenizer or not text:
            return text or ""
        try:
            ids = self._tokenizer.encode(text) if hasattr(self._tokenizer, "encode") else self._tokenizer(text)
        except Exception:
            return text
        if not ids or len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        if hasattr(self._tokenizer, "decode"):
            try:
                return self._tokenizer.decode(ids)
            except Exception:
                return text[: max_tokens * 4]
        return text[: max_tokens * 4]

    def score_pair(self, query: str, doc_text: str) -> float:
        """Score one (query, doc) pair. Returns logprob(Yes) - logprob(No).

        Higher = more relevant. Returns 0.0 on any error so callers can fall
        back gracefully on the original retrieval score.
        """
        self._resolve_yes_no_tokens()
        if self._yes_token is None or self._no_token is None:
            return 0.0
        if self._tokenizer is None:
            return 0.0

        prompt = _PROMPT_TEMPLATE.format(
            q=query.strip(),
            d=self._truncate(doc_text, self._doc_max_tokens),
        )
        try:
            token_ids = self._tokenizer.encode(prompt) if hasattr(self._tokenizer, "encode") else self._tokenizer(prompt)
        except Exception:
            return 0.0
        if not token_ids:
            return 0.0
        # Safety cap on prompt length
        if len(token_ids) > 1024:
            token_ids = token_ids[:1024]

        slot = self._next_slot()
        try:
            logits_bytes = self._runner.prefill(list(token_ids), seq_id=slot)
        except Exception:
            try:
                self._runner._kv_cache.free_sequence(slot)
            except Exception:
                pass
            return 0.0
        finally:
            try:
                self._runner._kv_cache.free_sequence(slot)
            except Exception:
                pass

        if not logits_bytes:
            return 0.0
        V = self._runner.vocab_size
        # Logits are fp16, length V
        try:
            logits = struct.unpack(f'<{V}e', logits_bytes[: V * 2])
        except Exception:
            return 0.0

        # Stable log-softmax over only the two tokens we care about:
        # logprob(yes) - logprob(no) = z_yes - z_no  (denominators cancel)
        # but logits are bounded — clip extreme to avoid overflow
        z_yes = float(logits[self._yes_token])
        z_no = float(logits[self._no_token])
        return z_yes - z_no

    def rerank(
        self, query: str, candidates: List[Tuple[int, str]], top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """Rerank (chunk_id, doc_text) pairs. Returns sorted [(chunk_id, score)] top_n."""
        if not candidates:
            return []
        scored = []
        for cid, text in candidates:
            s = self.score_pair(query, text)
            scored.append((cid, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
