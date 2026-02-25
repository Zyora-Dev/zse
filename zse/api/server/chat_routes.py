"""
ZSE Chat API Routes - Enhanced chat with persistence.

Provides REST API for:
- Conversation management (CRUD)
- Message management
- Chat with streaming
- Export/Import
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zse.api.server.chat_store import (
    get_chat_store,
    Conversation,
    Message,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    title: str = "New Chat"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class UpdateConversationRequest(BaseModel):
    """Request to update a conversation."""
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class AddMessageRequest(BaseModel):
    """Request to add a message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    title: str
    model: Optional[str]
    system_prompt: Optional[str]
    created_at: str
    updated_at: str
    message_count: int
    total_tokens: int
    settings: Dict[str, Any]


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    tokens: Optional[int]
    latency_ms: Optional[float]
    metadata: Dict[str, Any]


class ConversationListResponse(BaseModel):
    """List of conversations."""
    conversations: List[ConversationResponse]
    total: int


class MessageListResponse(BaseModel):
    """List of messages."""
    messages: List[MessageResponse]
    conversation_id: str


class ExportResponse(BaseModel):
    """Export response."""
    format: str
    content: str


class ChatStatsResponse(BaseModel):
    """Chat store statistics."""
    conversations: int
    messages: int
    total_tokens: int
    db_path: str
    db_size_mb: float


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# =============================================================================
# Conversation Endpoints
# =============================================================================

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    store = get_chat_store()
    conv = store.create_conversation(
        title=request.title,
        model=request.model,
        system_prompt=request.system_prompt,
        settings=request.settings,
    )
    return ConversationResponse(**conv.to_dict())


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all conversations."""
    store = get_chat_store()
    conversations = store.list_conversations(limit=limit, offset=offset)
    return ConversationListResponse(
        conversations=[ConversationResponse(**c.to_dict()) for c in conversations],
        total=len(conversations),
    )


@router.get("/conversations/{conv_id}", response_model=ConversationResponse)
async def get_conversation(conv_id: str):
    """Get a specific conversation."""
    store = get_chat_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(**conv.to_dict())


@router.patch("/conversations/{conv_id}", response_model=ConversationResponse)
async def update_conversation(conv_id: str, request: UpdateConversationRequest):
    """Update a conversation."""
    store = get_chat_store()
    conv = store.update_conversation(
        conv_id=conv_id,
        title=request.title,
        model=request.model,
        system_prompt=request.system_prompt,
        settings=request.settings,
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(**conv.to_dict())


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    store = get_chat_store()
    if not store.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conv_id}


@router.get("/conversations/search/{query}", response_model=ConversationListResponse)
async def search_conversations(
    query: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Search conversations by title or content."""
    store = get_chat_store()
    conversations = store.search_conversations(query, limit=limit)
    return ConversationListResponse(
        conversations=[ConversationResponse(**c.to_dict()) for c in conversations],
        total=len(conversations),
    )


# =============================================================================
# Message Endpoints
# =============================================================================

@router.get("/conversations/{conv_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conv_id: str,
    limit: Optional[int] = Query(default=None, ge=1, le=1000),
):
    """Get all messages in a conversation."""
    store = get_chat_store()
    
    # Verify conversation exists
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = store.get_messages(conv_id, limit=limit)
    return MessageListResponse(
        messages=[MessageResponse(**m.to_dict()) for m in messages],
        conversation_id=conv_id,
    )


@router.post("/conversations/{conv_id}/messages", response_model=MessageResponse)
async def add_message(conv_id: str, request: AddMessageRequest):
    """Add a message to a conversation."""
    store = get_chat_store()
    
    # Verify conversation exists
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    msg = store.add_message(
        conversation_id=conv_id,
        role=request.role,
        content=request.content,
        tokens=request.tokens,
        latency_ms=request.latency_ms,
        metadata=request.metadata,
    )
    return MessageResponse(**msg.to_dict())


@router.delete("/conversations/{conv_id}/messages/{msg_id}")
async def delete_message(conv_id: str, msg_id: str):
    """Delete a specific message."""
    store = get_chat_store()
    if not store.delete_message(msg_id):
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "deleted", "id": msg_id}


@router.delete("/conversations/{conv_id}/messages")
async def clear_messages(conv_id: str):
    """Clear all messages from a conversation."""
    store = get_chat_store()
    
    # Verify conversation exists
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    deleted = store.clear_conversation(conv_id)
    return {"status": "cleared", "deleted": deleted}


# =============================================================================
# Export/Import Endpoints
# =============================================================================

@router.get("/conversations/{conv_id}/export", response_model=ExportResponse)
async def export_conversation(
    conv_id: str,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
):
    """Export a conversation."""
    store = get_chat_store()
    try:
        content = store.export_conversation(conv_id, format=format)
        return ExportResponse(format=format, content=content)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversations/import", response_model=ConversationResponse)
async def import_conversation(data: str):
    """Import a conversation from JSON export."""
    store = get_chat_store()
    try:
        conv = store.import_conversation(data)
        return ConversationResponse(**conv.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid import data: {e}")


# =============================================================================
# Stats Endpoint
# =============================================================================

@router.get("/stats", response_model=ChatStatsResponse)
async def get_chat_stats():
    """Get chat store statistics."""
    store = get_chat_store()
    stats = store.get_stats()
    return ChatStatsResponse(**stats)
