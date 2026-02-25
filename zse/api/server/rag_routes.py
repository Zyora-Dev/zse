"""
ZSE RAG API Routes - Document-aware context retrieval.

Provides REST API for:
- Document upload and management
- Semantic search
- Context retrieval for chat augmentation
"""

import io
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field

from zse.api.server.rag import get_rag_store, Document, SearchResult


# =============================================================================
# Request/Response Models
# =============================================================================

class AddDocumentRequest(BaseModel):
    """Request to add a document by content."""
    name: str
    content: str
    file_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Document response."""
    id: str
    name: str
    chunk_count: int
    created_at: str
    file_type: str
    file_size: int
    metadata: Dict[str, Any]


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentResponse]
    total: int


class SearchRequest(BaseModel):
    """Search request."""
    query: str
    top_k: int = 5
    min_score: float = 0.0
    document_ids: Optional[List[str]] = None


class ChunkResponse(BaseModel):
    """Chunk in search result."""
    id: str
    document_id: str
    content: str
    chunk_index: int


class SearchResultResponse(BaseModel):
    """Single search result."""
    chunk: ChunkResponse
    document_name: str
    document_id: str
    score: float


class SearchResponse(BaseModel):
    """Search results."""
    results: List[SearchResultResponse]
    query: str


class ContextRequest(BaseModel):
    """Context retrieval request."""
    query: str
    top_k: int = 3
    max_tokens: int = 1000
    document_ids: Optional[List[str]] = None


class ContextResponse(BaseModel):
    """Context for chat injection."""
    context: str
    sources: List[SearchResultResponse]


class RAGStatsResponse(BaseModel):
    """RAG store statistics."""
    documents: int
    chunks: int
    embeddings: int
    total_size_bytes: int
    db_path: str


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# =============================================================================
# Document Endpoints
# =============================================================================

@router.post("/documents", response_model=DocumentResponse)
async def add_document(request: AddDocumentRequest):
    """Add a document by content."""
    store = get_rag_store()
    
    try:
        doc = store.add_document(
            name=request.name,
            content=request.content,
            file_type=request.file_type,
            metadata=request.metadata,
        )
        return DocumentResponse(
            id=doc.id,
            name=doc.name,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            file_type=doc.file_type,
            file_size=doc.file_size,
            metadata=doc.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(default=None),
):
    """Upload a document file.
    
    Supported formats: .txt, .md, .pdf (text extraction), .json
    """
    store = get_rag_store()
    
    # Read file content
    content_bytes = await file.read()
    filename = file.filename or "untitled"
    
    # Determine file type and extract text
    file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'txt'
    
    if file_ext in ('txt', 'md', 'markdown'):
        content = content_bytes.decode('utf-8', errors='ignore')
        file_type = 'text' if file_ext == 'txt' else 'markdown'
    
    elif file_ext == 'json':
        content = content_bytes.decode('utf-8', errors='ignore')
        file_type = 'json'
    
    elif file_ext == 'pdf':
        # Try to extract text from PDF
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            content = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            file_type = 'pdf'
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="PDF support requires pypdf: pip install pypdf"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    
    else:
        # Try to decode as text
        try:
            content = content_bytes.decode('utf-8', errors='ignore')
            file_type = 'text'
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Document is empty")
    
    # Parse metadata if provided
    meta = {}
    if metadata:
        try:
            import json
            meta = json.loads(metadata)
        except Exception:
            pass
    
    try:
        doc = store.add_document(
            name=filename,
            content=content,
            file_type=file_type,
            metadata=meta,
        )
        return DocumentResponse(
            id=doc.id,
            name=doc.name,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            file_type=doc.file_type,
            file_size=doc.file_size,
            metadata=doc.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List all documents."""
    store = get_rag_store()
    documents = store.list_documents(limit=limit, offset=offset)
    
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                name=d.name,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
                file_type=d.file_type,
                file_size=d.file_size,
                metadata=d.metadata,
            )
            for d in documents
        ],
        total=len(documents),
    )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a document by ID."""
    store = get_rag_store()
    doc = store.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=doc.id,
        name=doc.name,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
        file_type=doc.file_type,
        file_size=doc.file_size,
        metadata=doc.metadata,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    store = get_rag_store()
    
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"status": "deleted", "id": doc_id}


# =============================================================================
# Search Endpoints
# =============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks."""
    store = get_rag_store()
    
    results = store.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        document_ids=request.document_ids,
    )
    
    return SearchResponse(
        results=[
            SearchResultResponse(
                chunk=ChunkResponse(
                    id=r.chunk.id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    chunk_index=r.chunk.chunk_index,
                ),
                document_name=r.document.name,
                document_id=r.document.id,
                score=r.score,
            )
            for r in results
        ],
        query=request.query,
    )


@router.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """Get context for chat augmentation.
    
    Returns a formatted context string ready to inject into prompts,
    along with source information.
    """
    store = get_rag_store()
    
    context, results = store.get_context(
        query=request.query,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        document_ids=request.document_ids,
    )
    
    return ContextResponse(
        context=context,
        sources=[
            SearchResultResponse(
                chunk=ChunkResponse(
                    id=r.chunk.id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    chunk_index=r.chunk.chunk_index,
                ),
                document_name=r.document.name,
                document_id=r.document.id,
                score=r.score,
            )
            for r in results
        ],
    )


# =============================================================================
# Stats Endpoint
# =============================================================================

@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG store statistics."""
    store = get_rag_store()
    stats = store.get_stats()
    return RAGStatsResponse(**stats)
