"""
ZSE RAG Module - Document-aware context retrieval.

Provides:
- Document upload and management
- Smart text chunking
- Embedding generation (using loaded model or sentence-transformers)
- Vector similarity search
- Context injection for chat

Storage:
- SQLite for document metadata
- NumPy for embeddings (simple but effective)
"""

import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Document:
    """A document in the RAG store."""
    id: str
    name: str
    content: str
    chunk_count: int
    created_at: str
    file_type: str
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A text chunk from a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('embedding', None)  # Don't include embedding in JSON
        return d


@dataclass
class SearchResult:
    """A search result with similarity score."""
    chunk: Chunk
    document: Document
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "document": {
                "id": self.document.id,
                "name": self.document.name,
            },
            "score": float(self.score),
        }


class TextChunker:
    """Smart text chunking with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into chunks with positions.
        
        Returns:
            List of (chunk_text, start_char, end_char)
        """
        chunks = []
        self._split_recursive(text, chunks, 0)
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        chunks: List[Tuple[str, int, int]],
        offset: int,
        sep_idx: int = 0,
    ):
        """Recursively split text using separators."""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append((text, offset, offset + len(text)))
            return
        
        # Try each separator
        for i in range(sep_idx, len(self.separators)):
            sep = self.separators[i]
            
            if not sep:
                # Final fallback: hard split
                for j in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk = text[j:j + self.chunk_size]
                    if chunk.strip():
                        chunks.append((chunk, offset + j, offset + j + len(chunk)))
                return
            
            if sep in text:
                parts = text.split(sep)
                current_chunk = ""
                current_start = offset
                
                for part in parts:
                    if len(current_chunk) + len(part) + len(sep) <= self.chunk_size:
                        current_chunk += part + sep
                    else:
                        if current_chunk.strip():
                            chunks.append((
                                current_chunk.rstrip(sep),
                                current_start,
                                current_start + len(current_chunk) - len(sep),
                            ))
                        
                        # Start new chunk with overlap
                        overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                        current_start = offset + text.find(part)
                        current_chunk = overlap_text + part + sep
                
                if current_chunk.strip():
                    chunks.append((
                        current_chunk.rstrip(sep),
                        current_start,
                        current_start + len(current_chunk) - len(sep),
                    ))
                return


class SimpleEmbedder:
    """Simple embedding using TF-IDF + SVD or sentence-transformers.
    
    Falls back to a basic TF-IDF approach if sentence-transformers not available.
    """
    
    def __init__(self, model_name: Optional[str] = None, dimension: int = 384):
        self.dimension = dimension
        self.model = None
        self.use_sentence_transformers = False
        
        # Try to load sentence-transformers
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.use_sentence_transformers = True
                self.dimension = self.model.get_sentence_embedding_dimension()
            except ImportError:
                pass
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        if self.use_sentence_transformers and self.model:
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            # Simple TF-IDF-like embedding
            return self._tfidf_embed(texts)
    
    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """Basic TF-IDF-inspired embedding."""
        # Build vocabulary from texts
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)
        
        vocab = sorted(all_words)[:self.dimension]  # Limit vocab size
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        
        # Create embeddings
        embeddings = np.zeros((len(texts), self.dimension))
        
        for i, text in enumerate(texts):
            words = self._tokenize(text)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            for word, count in word_counts.items():
                if word in word_to_idx:
                    # TF-IDF-like score (simplified)
                    tf = count / len(words) if words else 0
                    embeddings[i, word_to_idx[word]] = tf
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords and short words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'or', 'and', 'but', 'if', 'than', 'so', 'no',
                     'not', 'only', 'own', 'same', 'too', 'very', 'just', 'this',
                     'that', 'these', 'those', 'it', 'its'}
        return [w for w in words if w not in stopwords and len(w) > 2]


class RAGStore:
    """SQLite + NumPy based RAG storage."""
    
    DEFAULT_DB_PATH = Path.home() / ".zse" / "rag.db"
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: Optional[str] = None,
    ):
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.db_path.with_suffix('.npy')
        
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = SimpleEmbedder(embedding_model)
        
        self._lock = Lock()
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        self._init_db()
        self._load_embeddings()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        file_type TEXT,
                        file_size INTEGER,
                        content_hash TEXT,
                        metadata TEXT DEFAULT '{}'
                    );
                    
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER,
                        start_char INTEGER,
                        end_char INTEGER,
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                            ON DELETE CASCADE
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_chunks_document 
                        ON chunks(document_id);
                    CREATE INDEX IF NOT EXISTS idx_documents_hash 
                        ON documents(content_hash);
                """)
                conn.commit()
            finally:
                conn.close()
    
    def _load_embeddings(self):
        """Load embeddings from disk."""
        if self.embeddings_path.exists():
            try:
                data = np.load(str(self.embeddings_path), allow_pickle=True).item()
                self._embeddings_cache = data
            except Exception:
                self._embeddings_cache = {}
    
    def _save_embeddings(self):
        """Save embeddings to disk."""
        np.save(str(self.embeddings_path), self._embeddings_cache)
    
    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------
    
    def add_document(
        self,
        name: str,
        content: str,
        file_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Add a document and generate embeddings."""
        # Check for duplicate
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with self._lock:
            conn = self._get_conn()
            try:
                existing = conn.execute(
                    "SELECT id FROM documents WHERE content_hash = ?",
                    (content_hash,)
                ).fetchone()
                
                if existing:
                    raise ValueError(f"Document with same content already exists: {existing['id']}")
            finally:
                conn.close()
        
        # Create document
        now = datetime.utcnow().isoformat()
        doc = Document(
            id=str(uuid.uuid4()),
            name=name,
            content=content,
            chunk_count=0,
            created_at=now,
            file_type=file_type,
            file_size=len(content.encode()),
            metadata=metadata or {},
        )
        
        # Chunk the content
        chunk_data = self.chunker.chunk(content)
        chunks = []
        
        for i, (chunk_text, start, end) in enumerate(chunk_data):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                content=chunk_text,
                chunk_index=i,
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
        
        doc.chunk_count = len(chunks)
        
        # Generate embeddings
        if chunks:
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedder.embed(chunk_texts)
            
            for chunk, emb in zip(chunks, embeddings):
                self._embeddings_cache[chunk.id] = emb
        
        # Save to database
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO documents 
                    (id, name, content, chunk_count, created_at, file_type, file_size, content_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.id, doc.name, doc.content, doc.chunk_count,
                    doc.created_at, doc.file_type, doc.file_size,
                    content_hash, json.dumps(doc.metadata)
                ))
                
                for chunk in chunks:
                    conn.execute("""
                        INSERT INTO chunks 
                        (id, document_id, content, chunk_index, start_char, end_char)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.id, chunk.document_id, chunk.content,
                        chunk.chunk_index, chunk.start_char, chunk.end_char
                    ))
                
                conn.commit()
            finally:
                conn.close()
        
        # Save embeddings
        self._save_embeddings()
        
        return doc
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                if not row:
                    return None
                return self._row_to_document(row)
            finally:
                conn.close()
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """List all documents."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM documents 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset)).fetchall()
                return [self._row_to_document(row) for row in rows]
            finally:
                conn.close()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        with self._lock:
            conn = self._get_conn()
            try:
                # Get chunk IDs first
                chunk_ids = [row['id'] for row in conn.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
                ).fetchall()]
                
                # Remove embeddings
                for chunk_id in chunk_ids:
                    self._embeddings_cache.pop(chunk_id, None)
                
                # Delete from DB (cascades to chunks)
                cursor = conn.execute(
                    "DELETE FROM documents WHERE id = ?", (doc_id,)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    self._save_embeddings()
                    return True
                return False
            finally:
                conn.close()
    
    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score (0-1)
            document_ids: Limit search to specific documents
        """
        if not self._embeddings_cache:
            return []
        
        # Get query embedding
        query_embedding = self.embedder.embed([query])[0]
        
        # Get chunks to search
        with self._lock:
            conn = self._get_conn()
            try:
                if document_ids:
                    placeholders = ','.join('?' * len(document_ids))
                    rows = conn.execute(f"""
                        SELECT c.*, d.name as doc_name 
                        FROM chunks c JOIN documents d ON c.document_id = d.id
                        WHERE c.document_id IN ({placeholders})
                    """, document_ids).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT c.*, d.name as doc_name 
                        FROM chunks c JOIN documents d ON c.document_id = d.id
                    """).fetchall()
                
                chunks = []
                for row in rows:
                    chunk = Chunk(
                        id=row['id'],
                        document_id=row['document_id'],
                        content=row['content'],
                        chunk_index=row['chunk_index'],
                        start_char=row['start_char'],
                        end_char=row['end_char'],
                    )
                    if chunk.id in self._embeddings_cache:
                        chunk.embedding = self._embeddings_cache[chunk.id]
                        chunks.append((chunk, row['doc_name']))
            finally:
                conn.close()
        
        if not chunks:
            return []
        
        # Calculate similarities
        results = []
        for chunk, doc_name in chunks:
            if chunk.embedding is not None:
                # Cosine similarity
                score = np.dot(query_embedding, chunk.embedding)
                
                if score >= min_score:
                    doc = self.get_document(chunk.document_id)
                    if doc:
                        results.append(SearchResult(chunk, doc, float(score)))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 1000,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[SearchResult]]:
        """Get context string for injection into prompts.
        
        Returns:
            (context_string, search_results)
        """
        results = self.search(query, top_k=top_k * 2, document_ids=document_ids)
        
        if not results:
            return "", []
        
        # Build context, respecting token limit (rough estimate: 4 chars = 1 token)
        context_parts = []
        used_results = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        for result in results:
            chunk_text = f"[{result.document.name}]: {result.chunk.content}"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            used_results.append(result)
            total_chars += len(chunk_text)
            
            if len(used_results) >= top_k:
                break
        
        context = "\n\n".join(context_parts)
        return context, used_results
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row['id'],
            name=row['name'],
            content=row['content'],
            chunk_count=row['chunk_count'],
            created_at=row['created_at'],
            file_type=row['file_type'],
            file_size=row['file_size'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG store statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                total_size = conn.execute(
                    "SELECT COALESCE(SUM(file_size), 0) FROM documents"
                ).fetchone()[0]
                
                return {
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "embeddings": len(self._embeddings_cache),
                    "total_size_bytes": total_size,
                    "db_path": str(self.db_path),
                }
            finally:
                conn.close()


# Global instance
_rag_store: Optional[RAGStore] = None
_store_lock = Lock()


def get_rag_store(
    db_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> RAGStore:
    """Get the global RAG store instance."""
    global _rag_store
    with _store_lock:
        if _rag_store is None:
            _rag_store = RAGStore(db_path, embedding_model=embedding_model)
        return _rag_store


def init_rag_store(
    db_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> RAGStore:
    """Initialize/reinitialize the RAG store."""
    global _rag_store
    with _store_lock:
        _rag_store = RAGStore(db_path, embedding_model=embedding_model)
        return _rag_store
