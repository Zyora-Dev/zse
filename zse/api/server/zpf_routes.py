"""
ZSE ZPF API Routes — .zpf-powered RAG endpoints.

Coexists with the existing /api/rag/ routes. These new routes use the
semantic compression pipeline (.zpf) for higher quality retrieval.

Endpoints:
    POST   /api/zpf/ingest          — Upload & ingest a document
    POST   /api/zpf/search          — Semantic search
    POST   /api/zpf/context         — Get LLM-ready context
    GET    /api/zpf/documents       — List ingested documents
    DELETE /api/zpf/documents/{id}  — Remove a document
    GET    /api/zpf/stats           — Pipeline statistics
    POST   /api/zpf/convert         — Convert file to .zpf (download)
    GET    /api/zpf/inspect/{id}    — Inspect a .zpf file
"""

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# =============================================================================
# Request/Response Models
# =============================================================================


class ZPFDocumentResponse(BaseModel):
    doc_id: str
    block_count: int
    total_tokens: int


class ZPFDocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total: int


class ZPFSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None
    score_threshold: float = 0.0


class ZPFSearchResultItem(BaseModel):
    doc_id: str
    block_idx: int
    score: float
    content: str
    block_type: int
    block_type_name: str
    summary: str


class ZPFSearchResponse(BaseModel):
    results: List[ZPFSearchResultItem]
    query: str


class ZPFContextRequest(BaseModel):
    query: str
    max_tokens: int = 2000
    top_k: int = 10
    doc_id: Optional[str] = None


class ZPFContextResponse(BaseModel):
    context: str
    block_count: int
    total_tokens_used: int


class ZPFStatsResponse(BaseModel):
    total_documents: int
    total_blocks: int
    embedding_model: str
    embedding_dim: int
    store_dir: str
    zpf_dir: str


class ZPFInspectResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    created_at: str
    block_count: int
    total_tokens: int
    compression_ratio: float
    token_efficiency: float
    blocks: List[Dict[str, Any]]


# =============================================================================
# Router
# =============================================================================

zpf_router = APIRouter(prefix="/api/zpf", tags=["ZPF RAG"])


def _get_pipeline():
    """Lazy import to avoid startup cost."""
    from zse.core.zrag.pipeline import get_rag_pipeline

    return get_rag_pipeline()


# =============================================================================
# Ingest
# =============================================================================


@zpf_router.post("/ingest", response_model=ZPFDocumentResponse)
async def zpf_ingest(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
):
    """Upload and ingest a document with .zpf semantic compression.

    Supports: PDF, DOCX, HTML, TXT, CSV, JSON, MD, .zpf
    """
    pipeline = _get_pipeline()

    content_bytes = await file.read()
    filename = file.filename or "untitled.txt"

    meta = {}
    if metadata:
        try:
            import json as _json

            meta = _json.loads(metadata)
        except Exception:
            pass

    try:
        doc_id = pipeline.ingest_bytes(
            data=content_bytes,
            filename=filename,
            title=title,
            metadata=meta,
        )

        docs = pipeline.list_documents()
        doc_info = next((d for d in docs if d["doc_id"] == doc_id), {})

        return ZPFDocumentResponse(
            doc_id=doc_id,
            block_count=doc_info.get("block_count", 0),
            total_tokens=doc_info.get("total_tokens", 0),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# =============================================================================
# Search
# =============================================================================


@zpf_router.post("/search", response_model=ZPFSearchResponse)
async def zpf_search(request: ZPFSearchRequest):
    """Semantic search across .zpf-ingested documents."""
    pipeline = _get_pipeline()

    from zse.core.zrag.zpf_spec import BlockType

    results = pipeline.search(
        query=request.query,
        top_k=request.top_k,
        doc_filter=request.doc_id,
        score_threshold=request.score_threshold,
    )

    return ZPFSearchResponse(
        results=[
            ZPFSearchResultItem(
                doc_id=r.doc_id,
                block_idx=r.block_idx,
                score=r.score,
                content=r.content,
                block_type=r.block_type,
                block_type_name=BlockType(r.block_type).name if r.block_type <= 10 else "TEXT",
                summary=r.summary,
            )
            for r in results
        ],
        query=request.query,
    )


# =============================================================================
# Context
# =============================================================================


@zpf_router.post("/context", response_model=ZPFContextResponse)
async def zpf_context(request: ZPFContextRequest):
    """Get LLM-ready context from .zpf semantic blocks."""
    pipeline = _get_pipeline()

    context = pipeline.get_context(
        query=request.query,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
        doc_filter=request.doc_id,
    )

    token_estimate = len(context) // 4 if context else 0

    return ZPFContextResponse(
        context=context,
        block_count=context.count("---") + 1 if context else 0,
        total_tokens_used=token_estimate,
    )


# =============================================================================
# Documents
# =============================================================================


@zpf_router.get("/documents", response_model=ZPFDocumentListResponse)
async def zpf_list_documents():
    """List all documents in the .zpf RAG store."""
    pipeline = _get_pipeline()
    docs = pipeline.list_documents()

    return ZPFDocumentListResponse(
        documents=docs,
        total=len(docs),
    )


@zpf_router.delete("/documents/{doc_id}")
async def zpf_delete_document(doc_id: str):
    """Remove a document from the .zpf RAG store."""
    pipeline = _get_pipeline()
    removed = pipeline.remove(doc_id)

    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted", "doc_id": doc_id, "blocks_removed": removed}


# =============================================================================
# Stats
# =============================================================================


@zpf_router.get("/stats", response_model=ZPFStatsResponse)
async def zpf_stats():
    """Get .zpf RAG pipeline statistics."""
    pipeline = _get_pipeline()
    return ZPFStatsResponse(**pipeline.stats)


# =============================================================================
# Convert (download .zpf)
# =============================================================================


@zpf_router.post("/convert")
async def zpf_convert(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
):
    """Convert a document to .zpf and return it for download.

    Does NOT add to the RAG store — just converts and returns the .zpf file.
    """
    pipeline = _get_pipeline()

    content_bytes = await file.read()
    filename = file.filename or "untitled.txt"

    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input
        input_path = os.path.join(tmpdir, filename)
        with open(input_path, "wb") as f:
            f.write(content_bytes)

        # Convert
        output_name = Path(filename).stem + ".zpf"
        output_path = os.path.join(tmpdir, output_name)

        try:
            pipeline.convert(input_path, output_path=output_path, title=title)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read and stream back
        with open(output_path, "rb") as f:
            zpf_bytes = f.read()

    return StreamingResponse(
        io.BytesIO(zpf_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={output_name}"},
    )
