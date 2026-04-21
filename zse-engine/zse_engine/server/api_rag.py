"""ZSE Server RAG API — Document upload, search, and management endpoints.

Endpoints:
    POST   /v1/rag/upload          — Upload and ingest a document
    PUT    /v1/rag/document/{id}   — Replace/update a document
    GET    /v1/rag/documents       — List all documents
    GET    /v1/rag/document/{id}   — Get document details
    DELETE /v1/rag/document/{id}   — Remove a document
    POST   /v1/rag/search          — Search for relevant chunks (hybrid, filterable)
    GET    /v1/rag/stats           — RAG store statistics
"""

import base64

from zse_engine.server.router import Request, Response
from zse_engine.server.auth import AuthManager


class RAGAPI:
    """RAG endpoint handlers.

    Args:
        auth: AuthManager instance
        rag_engine: RAGEngine instance
    """

    def __init__(self, auth: AuthManager, rag_engine=None):
        self._auth = auth
        self._rag = rag_engine

    def register(self, router):
        """Register RAG routes."""
        router.post("/v1/rag/upload", self.handle_upload)
        router.put("/v1/rag/document/{id}", self.handle_replace)
        router.get("/v1/rag/documents", self.handle_list)
        router.get("/v1/rag/document/{id}", self.handle_get)
        router.delete("/v1/rag/document/{id}", self.handle_delete)
        router.post("/v1/rag/search", self.handle_search)
        router.get("/v1/rag/stats", self.handle_stats)

    def _check_auth(self, request: Request):
        """Authenticate and return (auth_result, error_response)."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return None, Response.error(auth.error, auth.status_code)
        return auth, None

    async def handle_upload(self, request: Request) -> Response:
        """Upload and ingest a document.

        Body: {
            "filename": "data.json",
            "content": "<base64-encoded file content>",
            "chunk_size": 512,    (optional)
            "overlap": 64         (optional)
        }
        """
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.error("RAG engine not available", 503)

        body = request.json
        filename = body.get("filename", "")
        content_b64 = body.get("content", "")

        if not filename:
            return Response.error("'filename' field required")
        if not content_b64:
            return Response.error("'content' field required (base64-encoded)")

        try:
            content = base64.b64decode(content_b64)
        except Exception:
            return Response.error("Invalid base64 content")

        chunk_size = body.get("chunk_size")
        overlap = body.get("overlap")

        try:
            result = self._rag.ingest(
                filename=filename,
                content=content,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if "error" in result:
                return Response.error(result["error"])
            return Response.json(result, status=201)
        except Exception as e:
            return Response.error(f"Ingestion failed: {str(e)}", 500)

    async def handle_replace(self, request: Request) -> Response:
        """Replace/update an existing document.

        Deletes the old document and re-ingests with new content.

        Body: {
            "filename": "data.json",  (optional, keeps original name if omitted)
            "content": "<base64-encoded file content>",
            "chunk_size": 512,    (optional)
            "overlap": 64         (optional)
        }
        """
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.error("RAG engine not available", 503)

        doc_id_str = request.path_params.get("id", "")
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            return Response.error("Invalid document ID")

        # Get existing document for filename
        existing = self._rag.get_document(doc_id)
        if not existing:
            return Response.error("Document not found", 404)

        body = request.json
        filename = body.get("filename", existing["name"])
        content_b64 = body.get("content", "")

        if not content_b64:
            return Response.error("'content' field required (base64-encoded)")

        try:
            content = base64.b64decode(content_b64)
        except Exception:
            return Response.error("Invalid base64 content")

        # Delete old
        self._rag.remove_document(doc_id)

        # Ingest new
        chunk_size = body.get("chunk_size")
        overlap = body.get("overlap")

        try:
            result = self._rag.ingest(
                filename=filename,
                content=content,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if "error" in result:
                return Response.error(result["error"])
            result["replaced_doc_id"] = doc_id
            return Response.json(result)
        except Exception as e:
            return Response.error(f"Ingestion failed: {str(e)}", 500)

    async def handle_list(self, request: Request) -> Response:
        """List all RAG documents."""
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.json({"documents": []})

        docs = self._rag.list_documents()
        return Response.json({"documents": docs})

    async def handle_get(self, request: Request) -> Response:
        """Get document details with chunk previews."""
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.error("RAG engine not available", 503)

        doc_id_str = request.path_params.get("id", "")
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            return Response.error("Invalid document ID")

        doc = self._rag.get_document(doc_id)
        if not doc:
            return Response.error("Document not found", 404)

        return Response.json(doc)

    async def handle_delete(self, request: Request) -> Response:
        """Remove a document."""
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.error("RAG engine not available", 503)

        doc_id_str = request.path_params.get("id", "")
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            return Response.error("Invalid document ID")

        success = self._rag.remove_document(doc_id)
        if success:
            return Response.json({"deleted": True, "doc_id": doc_id})
        return Response.error("Document not found", 404)

    async def handle_search(self, request: Request) -> Response:
        """Search RAG store for relevant chunks.

        Supports metadata filtering and multi-query decomposition.

        Body: {
            "query": "search text",
            "top_k": 5,                        (optional, default 5)
            "doc_type": "json",                (optional, filter by doc type)
            "doc_name": "data.json",           (optional, filter by doc name)
            "doc_id": 3,                       (optional, filter by doc ID)
            "multi_query": true                (optional, decompose query into sub-queries)
        }
        """
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.json({"results": []})

        body = request.json
        query = body.get("query", "")
        top_k = body.get("top_k", 5)

        if not query:
            return Response.error("'query' field required")

        # Metadata filters
        filters = {}
        if "doc_type" in body:
            filters["doc_type"] = body["doc_type"]
        if "doc_name" in body:
            filters["doc_name"] = body["doc_name"]
        if "doc_id" in body:
            filters["doc_id"] = body["doc_id"]

        # Multi-query decomposition
        multi_query = body.get("multi_query", False)

        if multi_query:
            results = self._rag.multi_query_search(query, top_k=top_k, filters=filters)
        elif filters:
            results = self._rag.search(query, top_k=top_k, filters=filters)
        else:
            results = self._rag.search(query, top_k=top_k)

        return Response.json({"results": results, "query": query})

    async def handle_stats(self, request: Request) -> Response:
        """RAG store statistics."""
        auth, err = self._check_auth(request)
        if err:
            return err

        if not self._rag:
            return Response.json({
                "document_count": 0, "chunk_count": 0,
                "total_original_tokens": 0, "total_compressed_tokens": 0,
                "token_savings_pct": 0.0,
            })

        return Response.json(self._rag.stats())
