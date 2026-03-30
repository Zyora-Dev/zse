"""
Embedding Engine — Generate vector embeddings for semantic blocks.

Supports:
1. sentence-transformers (high quality, recommended)
2. TF-IDF fallback (zero dependencies, decent for simple use)

The embedder operates on semantic blocks (compressed text), not raw
document text — meaning each embedding captures the distilled meaning.
"""

import re
from typing import List, Optional

import numpy as np


class Embedder:
    """
    Embedding engine with sentence-transformers support + TF-IDF fallback.

    Usage:
        emb = Embedder()                  # TF-IDF fallback
        emb = Embedder("all-MiniLM-L6-v2")  # sentence-transformers
        vectors = emb.embed(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self.model_name = model_name or ""
        self.dimension = dimension
        self._model = None
        self._use_st = False

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self._use_st = True
                self.dimension = self._model.get_sentence_embedding_dimension()
                self.model_name = model_name
            except (ImportError, Exception):
                # Fallback to TF-IDF
                self._use_st = False
                self.model_name = "tfidf-fallback"
                import warnings
                warnings.warn(
                    f"sentence-transformers not available — falling back to TF-IDF embeddings. "
                    f"TF-IDF embeddings are NOT compatible with neural embeddings. "
                    f"Documents ingested with TF-IDF will give poor search results if later "
                    f"searched with a neural model. Install sentence-transformers: "
                    f"pip install sentence-transformers",
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def name(self) -> str:
        return self.model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of texts.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self._use_st and self._model is not None:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        return self._tfidf_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (dimension,)."""
        return self.embed([query])[0]

    # ----- TF-IDF Fallback -----

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """Basic TF-IDF embedding when sentence-transformers unavailable."""
        # Build vocabulary
        all_words: set = set()
        tokenized = []
        for text in texts:
            words = self._tokenize(text)
            tokenized.append(words)
            all_words.update(words)

        vocab = sorted(all_words)[:self.dimension]
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for i, words in enumerate(tokenized):
            if not words:
                continue
            counts: dict = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                if w in word_to_idx:
                    embeddings[i, word_to_idx[w]] = c / len(words)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        return embeddings

    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "of", "to", "in",
        "for", "on", "with", "at", "by", "from", "as", "or", "and", "but",
        "if", "than", "so", "no", "not", "only", "own", "same", "too",
        "very", "just", "this", "that", "these", "those", "it", "its",
    })

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self._STOPWORDS and len(w) > 2]
