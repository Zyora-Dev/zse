"""Tests for modern RAG additions: dense embeddings, RRF fusion, LLM rerank.

Uses a FakeModelRunner that mimics the ModelRunner interface so we can test
the wiring + math without actually loading an LLM.
"""

import os
import sqlite3
import struct
import tempfile
import unittest

from zse_engine.rag.dense_embedder import DenseEmbedder, dense_cosine
from zse_engine.rag.reranker import LLMReranker
from zse_engine.rag.engine import RAGEngine
from zse_engine.rag.store import RAGStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Word-hash tokenizer with stable encode/decode."""
    vocab_size = 32000

    def encode(self, text):
        if not text:
            return []
        out = []
        for w in text.lower().split():
            h = 0
            for ch in w:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            out.append(h % self.vocab_size)
        return out

    def decode(self, ids):
        return "tok " * len(ids)

    def __call__(self, text):
        return self.encode(text)


class FakeKVCache:
    def allocate_sequence(self, seq_id, prompt_tokens=None):
        pass

    def mark_active(self, seq_id):
        pass

    def free_sequence(self, seq_id):
        pass


class FakeModelRunner:
    """Deterministic fake embedder: dense vector = bag-of-words sketch.

    For two semantically similar inputs (overlapping tokens) the cosine is high;
    for unrelated inputs it is low. Good enough to test wiring + math.
    """
    hidden_size = 64
    vocab_size = 32000

    def __init__(self):
        self._kv_cache = FakeKVCache()
        # vector seed per token-id (stable hash)
        self._tok_vecs = {}

    def _vec_for_token(self, tid):
        v = self._tok_vecs.get(tid)
        if v is not None:
            return v
        import math
        rng = (tid * 2654435761) & 0xFFFFFFFF
        out = []
        for _ in range(self.hidden_size):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(((rng & 0xFFFF) / 32768.0) - 1.0)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        out = [x / norm for x in out]
        self._tok_vecs[tid] = out
        return out

    def embed_pooled(self, token_ids, seq_id):
        import math
        pooled = [0.0] * self.hidden_size
        for tid in token_ids:
            v = self._vec_for_token(tid)
            for i in range(self.hidden_size):
                pooled[i] += v[i]
        n = max(len(token_ids), 1)
        pooled = [x / n for x in pooled]
        norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
        pooled = [x / norm for x in pooled]
        return struct.pack(f'<{self.hidden_size}e', *pooled)

    def prefill(self, token_ids, seq_id, lora_adapter=None):
        """Return fp16 logits that favor 'Yes' if query/doc share tokens.

        Heuristic: the prompt has 'Query:' and 'Document:' segments. We
        approximate relevance by checking if any non-stopword token appears
        in both — boost Yes-token logit. For tests, we pass tokens directly.
        """
        # We use the last 4 token IDs as a proxy. In real ZSE these would be
        # actual logits; here we manufacture a vocab-sized vector with a small
        # boost on token 'YES_ID' if heuristic fires.
        V = self.vocab_size
        out = [0.0] * V
        # Yes/No tokens (whatever tokenizer maps "Yes"/"No" to)
        yes_id = FakeTokenizer().encode("Yes")[-1]
        no_id = FakeTokenizer().encode("No")[-1]
        # Heuristic: count distinct token IDs that appear >=2 times in prompt
        from collections import Counter
        c = Counter(token_ids)
        overlap = sum(1 for v in c.values() if v >= 2)
        if overlap >= 2:
            out[yes_id] = 5.0
            out[no_id] = -2.0
        else:
            out[yes_id] = -1.0
            out[no_id] = 3.0
        return struct.pack(f'<{V}e', *out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDenseEmbedder(unittest.TestCase):
    def setUp(self):
        self.runner = FakeModelRunner()
        self.tok = FakeTokenizer()
        self.emb = DenseEmbedder(self.runner, self.tok)

    def test_embed_returns_correct_size(self):
        v = self.emb.embed("hello world")
        self.assertIsNotNone(v)
        self.assertEqual(len(v), self.runner.hidden_size * 2)

    def test_similar_texts_have_higher_cosine(self):
        a = self.emb.embed("the quick brown fox jumps over the lazy dog")
        b = self.emb.embed("the quick brown fox jumped over a lazy dog")
        c = self.emb.embed("quantum mechanics describes subatomic particles")
        sim_ab = dense_cosine(a, b)
        sim_ac = dense_cosine(a, c)
        self.assertGreater(sim_ab, sim_ac)

    def test_l2_normalized(self):
        v = self.emb.embed("test text")
        n = self.runner.hidden_size
        floats = struct.unpack(f'<{n}e', v)
        import math
        norm = math.sqrt(sum(x * x for x in floats))
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_empty_text_returns_none(self):
        self.assertIsNone(self.emb.embed(""))
        self.assertIsNone(self.emb.embed("   "))

    def test_query_cache(self):
        v1 = self.emb.embed_query("repeated query")
        v2 = self.emb.embed_query("repeated query")
        self.assertEqual(v1, v2)


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.runner = FakeModelRunner()
        self.tok = FakeTokenizer()
        self.rr = LLMReranker(self.runner, self.tok, doc_max_tokens=64)

    def test_score_pair_returns_finite(self):
        s = self.rr.score_pair("python programming", "python is a programming language")
        self.assertIsInstance(s, float)

    def test_rerank_orders_candidates(self):
        # Doc with high overlap should rank above non-overlapping
        cands = [
            (1, "machine learning is fun"),
            (2, "machine learning machine learning machine"),
            (3, "cooking pasta tonight"),
        ]
        out = self.rr.rerank("machine learning", cands, top_n=3)
        self.assertEqual(len(out), 3)
        # Result is list of (cid, score) sorted descending
        self.assertEqual(out[0][1], max(s for _, s in out))


class TestRRFFusion(unittest.TestCase):
    """Test the RRF fusion math directly via a tiny RAGEngine instance."""

    def setUp(self):
        from zse_engine.server.database import ServerDatabase
        self._tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self._tmp.close()
        self._db = ServerDatabase(self._tmp.name)
        store = RAGStore(self._db)
        self.engine = RAGEngine(store=store, tokenizer=FakeTokenizer())

    def tearDown(self):
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_dense_schema_migration(self):
        # Verify the dense_vector column was added
        cols = {row[1] for row in self.engine._store._conn.execute(
            "PRAGMA table_info(rag_chunks)"
        ).fetchall()}
        self.assertIn("dense_vector", cols)

    def test_set_model_runner_enables_dense(self):
        self.assertFalse(self.engine.has_dense())
        runner = FakeModelRunner()
        self.engine.set_model_runner(runner)
        self.assertTrue(self.engine.has_dense())

    def test_search_with_dense_and_rerank_end_to_end(self):
        # Wire fake model
        runner = FakeModelRunner()
        self.engine.set_model_runner(runner)

        # Ingest a few docs
        docs = {
            "ml.txt": b"machine learning models learn from data using algorithms",
            "cook.txt": b"cooking pasta requires water salt and time",
            "ai.txt": b"artificial intelligence and machine learning are related fields",
        }
        for name, content in docs.items():
            self.engine.ingest(name, content)

        # Search for ML — both ml.txt and ai.txt should be more relevant than cook.txt
        results = self.engine.search(
            "machine learning algorithms", top_k=3,
            use_dense=True, use_rerank=True,
        )
        self.assertGreater(len(results), 0)
        # Top-ranked should be one of the ML-related docs
        top_doc = results[0]["doc_name"]
        self.assertIn(top_doc, ("ml.txt", "ai.txt"))

    def test_search_falls_back_without_model_runner(self):
        # Without dense wiring, search still works (keyword-only)
        self.engine.ingest("a.txt", b"the quick brown fox jumps over the lazy dog")
        results = self.engine.search("quick fox", top_k=3, use_dense=True)
        # use_dense silently falls back to False; should still get results
        self.assertGreater(len(results), 0)

    def test_rrf_score_ranges(self):
        # RRF scores should be in (0, 1) per rank contribution
        self.engine.ingest("a.txt", b"alpha beta gamma delta")
        self.engine.ingest("b.txt", b"alpha beta epsilon zeta")
        results = self.engine.search("alpha beta", top_k=2, fusion="rrf")
        for r in results:
            self.assertGreater(r["score"], 0.0)
            self.assertLess(r["score"], 1.0)


class TestBackfillDense(unittest.TestCase):
    def setUp(self):
        from zse_engine.server.database import ServerDatabase
        self._tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self._tmp.close()
        self._db = ServerDatabase(self._tmp.name)
        store = RAGStore(self._db)
        self.engine = RAGEngine(store=store, tokenizer=FakeTokenizer())

    def tearDown(self):
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_backfill_returns_zero_without_runner(self):
        self.engine.ingest("a.txt", b"content one")
        self.assertEqual(self.engine.backfill_dense_embeddings(), 0)

    def test_backfill_embeds_existing_chunks(self):
        # Ingest WITHOUT dense (no model_runner wired yet)
        self.engine.ingest("a.txt", b"content for backfill test one")
        self.engine.ingest("b.txt", b"content for backfill test two")

        # Now wire model runner & backfill
        self.engine.set_model_runner(FakeModelRunner())
        n = self.engine.backfill_dense_embeddings()
        self.assertGreaterEqual(n, 2)

        # Subsequent backfill should be no-op
        self.assertEqual(self.engine.backfill_dense_embeddings(), 0)


if __name__ == "__main__":
    unittest.main()
