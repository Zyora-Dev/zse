"""
ZSE Chat Store - SQLite-based conversation persistence.

Provides persistent storage for:
- Conversations (chat sessions)
- Messages (user/assistant exchanges)
- Settings (per-conversation parameters)
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    """A single message in a conversation."""
    id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    content: str
    created_at: str
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_api_format(self) -> Dict[str, str]:
        """Convert to OpenAI API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A conversation/chat session."""
    id: str
    title: str
    model: Optional[str]
    system_prompt: Optional[str]
    created_at: str
    updated_at: str
    message_count: int = 0
    total_tokens: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChatStore:
    """SQLite-based chat storage."""
    
    DEFAULT_DB_PATH = Path.home() / ".zse" / "chat.db"
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize chat store.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.zse/chat.db
        """
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    -- Conversations table
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        model TEXT,
                        system_prompt TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        settings TEXT DEFAULT '{}'
                    );
                    
                    -- Messages table
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        tokens INTEGER,
                        latency_ms REAL,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                            ON DELETE CASCADE
                    );
                    
                    -- Indexes
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                        ON messages(conversation_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_conversations_updated 
                        ON conversations(updated_at DESC);
                """)
                conn.commit()
            finally:
                conn.close()
    
    # -------------------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------------------
    
    def create_conversation(
        self,
        title: str = "New Chat",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation."""
        now = datetime.utcnow().isoformat()
        conv = Conversation(
            id=str(uuid.uuid4()),
            title=title,
            model=model,
            system_prompt=system_prompt,
            created_at=now,
            updated_at=now,
            settings=settings or {},
        )
        
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO conversations 
                    (id, title, model, system_prompt, created_at, updated_at, settings)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    conv.id, conv.title, conv.model, conv.system_prompt,
                    conv.created_at, conv.updated_at, json.dumps(conv.settings)
                ))
                conn.commit()
            finally:
                conn.close()
        
        return conv
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conv_id,)
                ).fetchone()
                if not row:
                    return None
                return self._row_to_conversation(row)
            finally:
                conn.close()
    
    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Conversation]:
        """List conversations, newest first."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM conversations 
                    ORDER BY updated_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset)).fetchall()
                return [self._row_to_conversation(row) for row in rows]
            finally:
                conn.close()
    
    def update_conversation(
        self,
        conv_id: str,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Update conversation properties."""
        with self._lock:
            conn = self._get_conn()
            try:
                # Build update query dynamically
                updates = []
                params = []
                
                if title is not None:
                    updates.append("title = ?")
                    params.append(title)
                if model is not None:
                    updates.append("model = ?")
                    params.append(model)
                if system_prompt is not None:
                    updates.append("system_prompt = ?")
                    params.append(system_prompt)
                if settings is not None:
                    updates.append("settings = ?")
                    params.append(json.dumps(settings))
                
                if not updates:
                    return self.get_conversation(conv_id)
                
                updates.append("updated_at = ?")
                params.append(datetime.utcnow().isoformat())
                params.append(conv_id)
                
                conn.execute(f"""
                    UPDATE conversations 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
                
                return self.get_conversation(conv_id)
            finally:
                conn.close()
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?", (conv_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
    
    def search_conversations(self, query: str, limit: int = 20) -> List[Conversation]:
        """Search conversations by title or message content."""
        with self._lock:
            conn = self._get_conn()
            try:
                # Search in titles and messages
                rows = conn.execute("""
                    SELECT DISTINCT c.* FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.title LIKE ? OR m.content LIKE ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit)).fetchall()
                return [self._row_to_conversation(row) for row in rows]
            finally:
                conn.close()
    
    # -------------------------------------------------------------------------
    # Message Operations
    # -------------------------------------------------------------------------
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to a conversation."""
        now = datetime.utcnow().isoformat()
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            tokens=tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO messages 
                    (id, conversation_id, role, content, created_at, tokens, latency_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    msg.id, msg.conversation_id, msg.role, msg.content,
                    msg.created_at, msg.tokens, msg.latency_ms, json.dumps(msg.metadata)
                ))
                
                # Update conversation stats
                conn.execute("""
                    UPDATE conversations 
                    SET message_count = message_count + 1,
                        total_tokens = total_tokens + ?,
                        updated_at = ?
                    WHERE id = ?
                """, (tokens or 0, now, conversation_id))
                
                conn.commit()
            finally:
                conn.close()
        
        return msg
    
    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        before_id: Optional[str] = None,
    ) -> List[Message]:
        """Get messages for a conversation."""
        with self._lock:
            conn = self._get_conn()
            try:
                query = "SELECT * FROM messages WHERE conversation_id = ?"
                params: List[Any] = [conversation_id]
                
                if before_id:
                    query += " AND created_at < (SELECT created_at FROM messages WHERE id = ?)"
                    params.append(before_id)
                
                query += " ORDER BY created_at ASC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_message(row) for row in rows]
            finally:
                conn.close()
    
    def get_conversation_for_api(
        self,
        conversation_id: str,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Get messages in OpenAI API format."""
        conv = self.get_conversation(conversation_id)
        messages = self.get_messages(conversation_id)
        
        result = []
        
        # Add system prompt if exists
        if include_system and conv and conv.system_prompt:
            result.append({"role": "system", "content": conv.system_prompt})
        
        # Add messages
        for msg in messages:
            if msg.role != "system":  # Skip system messages in history
                result.append(msg.to_api_format())
        
        return result
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a specific message."""
        with self._lock:
            conn = self._get_conn()
            try:
                # Get message info first
                row = conn.execute(
                    "SELECT conversation_id, tokens FROM messages WHERE id = ?",
                    (message_id,)
                ).fetchone()
                
                if not row:
                    return False
                
                conv_id, tokens = row["conversation_id"], row["tokens"] or 0
                
                # Delete message
                conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                
                # Update conversation stats
                conn.execute("""
                    UPDATE conversations 
                    SET message_count = message_count - 1,
                        total_tokens = total_tokens - ?,
                        updated_at = ?
                    WHERE id = ?
                """, (tokens, datetime.utcnow().isoformat(), conv_id))
                
                conn.commit()
                return True
            finally:
                conn.close()
    
    def clear_conversation(self, conversation_id: str) -> int:
        """Clear all messages from a conversation."""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                deleted = cursor.rowcount
                
                # Reset conversation stats
                conn.execute("""
                    UPDATE conversations 
                    SET message_count = 0, total_tokens = 0, updated_at = ?
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), conversation_id))
                
                conn.commit()
                return deleted
            finally:
                conn.close()
    
    # -------------------------------------------------------------------------
    # Export/Import
    # -------------------------------------------------------------------------
    
    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
    ) -> str:
        """Export a conversation.
        
        Args:
            conversation_id: Conversation to export
            format: "json" or "markdown"
        """
        conv = self.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = self.get_messages(conversation_id)
        
        if format == "json":
            return json.dumps({
                "conversation": conv.to_dict(),
                "messages": [m.to_dict() for m in messages],
            }, indent=2)
        
        elif format == "markdown":
            lines = [
                f"# {conv.title}",
                f"",
                f"**Model:** {conv.model or 'Not specified'}",
                f"**Created:** {conv.created_at}",
                f"**Messages:** {conv.message_count}",
                f"",
            ]
            
            if conv.system_prompt:
                lines.extend([
                    "## System Prompt",
                    "",
                    f"> {conv.system_prompt}",
                    "",
                ])
            
            lines.append("## Conversation")
            lines.append("")
            
            for msg in messages:
                role_label = "**User:**" if msg.role == "user" else "**Assistant:**"
                lines.append(role_label)
                lines.append("")
                lines.append(msg.content)
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def import_conversation(self, data: str) -> Conversation:
        """Import a conversation from JSON export."""
        parsed = json.loads(data)
        
        conv_data = parsed["conversation"]
        messages_data = parsed["messages"]
        
        # Create new conversation with new ID
        conv = self.create_conversation(
            title=conv_data.get("title", "Imported Chat"),
            model=conv_data.get("model"),
            system_prompt=conv_data.get("system_prompt"),
            settings=conv_data.get("settings", {}),
        )
        
        # Add messages
        for msg_data in messages_data:
            self.add_message(
                conversation_id=conv.id,
                role=msg_data["role"],
                content=msg_data["content"],
                tokens=msg_data.get("tokens"),
                latency_ms=msg_data.get("latency_ms"),
                metadata=msg_data.get("metadata", {}),
            )
        
        return conv
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        """Convert database row to Conversation."""
        return Conversation(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            settings=json.loads(row["settings"]) if row["settings"] else {},
        )
    
    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert database row to Message."""
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            tokens=row["tokens"],
            latency_ms=row["latency_ms"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat store statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                conv_count = conn.execute(
                    "SELECT COUNT(*) FROM conversations"
                ).fetchone()[0]
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0]
                total_tokens = conn.execute(
                    "SELECT COALESCE(SUM(total_tokens), 0) FROM conversations"
                ).fetchone()[0]
                
                return {
                    "conversations": conv_count,
                    "messages": msg_count,
                    "total_tokens": total_tokens,
                    "db_path": str(self.db_path),
                    "db_size_mb": self.db_path.stat().st_size / (1024 * 1024)
                        if self.db_path.exists() else 0,
                }
            finally:
                conn.close()


# Global instance
_chat_store: Optional[ChatStore] = None
_store_lock = Lock()


def get_chat_store(db_path: Optional[str] = None) -> ChatStore:
    """Get the global chat store instance."""
    global _chat_store
    with _store_lock:
        if _chat_store is None:
            _chat_store = ChatStore(db_path)
        return _chat_store


def init_chat_store(db_path: Optional[str] = None) -> ChatStore:
    """Initialize/reinitialize the chat store."""
    global _chat_store
    with _store_lock:
        _chat_store = ChatStore(db_path)
        return _chat_store
