"""
ZSE MCP API Routes - Tool/function calling endpoints.

Provides REST API for:
- Tool listing and management
- Tool execution
- Function calling support for chat
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from zse.api.server.mcp import (
    get_tool_registry,
    register_tool,
    Tool,
    ToolCall,
    ToolResult,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ToolParameterSchema(BaseModel):
    """Tool parameter schema."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Tool schema for API responses."""
    name: str
    description: str
    parameters: List[ToolParameterSchema]


class ToolListResponse(BaseModel):
    """List of tools."""
    tools: List[ToolSchema]
    openai_format: List[Dict[str, Any]]


class ExecuteToolRequest(BaseModel):
    """Request to execute a tool."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResultResponse(BaseModel):
    """Tool execution result."""
    tool_call_id: str
    name: str
    result: Optional[Any]
    error: Optional[str]


class RegisterToolRequest(BaseModel):
    """Request to register a custom tool (no handler - for schema only)."""
    name: str
    description: str
    parameters: List[ToolParameterSchema]


class ParseToolCallsRequest(BaseModel):
    """Request to parse tool calls from text."""
    content: str


class ToolCallSchema(BaseModel):
    """Tool call schema."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ParseToolCallsResponse(BaseModel):
    """Parsed tool calls."""
    tool_calls: List[ToolCallSchema]


class ProcessToolCallsRequest(BaseModel):
    """Request to parse and execute tool calls."""
    content: str
    auto_execute: bool = True


class ProcessToolCallsResponse(BaseModel):
    """Processed tool calls with results."""
    tool_calls: List[ToolCallSchema]
    results: List[ToolResultResponse]


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/tools", tags=["Tools"])


# =============================================================================
# Tool Endpoints
# =============================================================================

@router.get("/", response_model=ToolListResponse)
async def list_tools():
    """List all registered tools."""
    registry = get_tool_registry()
    tools = registry.list_tools()
    
    return ToolListResponse(
        tools=[
            ToolSchema(
                name=t.name,
                description=t.description,
                parameters=[
                    ToolParameterSchema(
                        name=p.name,
                        type=p.type,
                        description=p.description,
                        required=p.required,
                        enum=p.enum,
                        default=p.default,
                    )
                    for p in t.parameters
                ],
            )
            for t in tools
        ],
        openai_format=registry.get_openai_tools(),
    )


@router.get("/{name}", response_model=ToolSchema)
async def get_tool(name: str):
    """Get a specific tool by name."""
    registry = get_tool_registry()
    tool = registry.get(name)
    
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    
    return ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters=[
            ToolParameterSchema(
                name=p.name,
                type=p.type,
                description=p.description,
                required=p.required,
                enum=p.enum,
                default=p.default,
            )
            for p in tool.parameters
        ],
    )


@router.post("/execute", response_model=ToolResultResponse)
async def execute_tool(request: ExecuteToolRequest):
    """Execute a tool with given arguments."""
    registry = get_tool_registry()
    
    tool = registry.get(request.name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
    
    result = registry.execute(request.name, request.arguments)
    
    return ToolResultResponse(
        tool_call_id=result.tool_call_id,
        name=result.name,
        result=result.result,
        error=result.error,
    )


@router.post("/parse", response_model=ParseToolCallsResponse)
async def parse_tool_calls(request: ParseToolCallsRequest):
    """Parse tool calls from model output text."""
    registry = get_tool_registry()
    
    tool_calls = registry.parse_tool_calls(request.content)
    
    return ParseToolCallsResponse(
        tool_calls=[
            ToolCallSchema(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
            )
            for tc in tool_calls
        ],
    )


@router.post("/process", response_model=ProcessToolCallsResponse)
async def process_tool_calls(request: ProcessToolCallsRequest):
    """Parse and optionally execute tool calls from model output."""
    registry = get_tool_registry()
    
    tool_calls, results = registry.process_tool_calls(
        request.content,
        auto_execute=request.auto_execute,
    )
    
    return ProcessToolCallsResponse(
        tool_calls=[
            ToolCallSchema(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
            )
            for tc in tool_calls
        ],
        results=[
            ToolResultResponse(
                tool_call_id=r.tool_call_id,
                name=r.name,
                result=r.result,
                error=r.error,
            )
            for r in results
        ],
    )


@router.post("/register")
async def register_custom_tool(request: RegisterToolRequest):
    """Register a custom tool schema (for external execution).
    
    Note: Custom tools registered this way don't have handlers and
    cannot be executed via /execute. They are for schema purposes only.
    """
    registry = get_tool_registry()
    
    # Check if tool already exists
    existing = registry.get(request.name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Tool '{request.name}' already exists"
        )
    
    # Create tool without handler
    from zse.api.server.mcp import Tool, ToolParameter
    
    tool = Tool(
        name=request.name,
        description=request.description,
        parameters=[
            ToolParameter(
                name=p.name,
                type=p.type,
                description=p.description,
                required=p.required,
                enum=p.enum,
                default=p.default,
            )
            for p in request.parameters
        ],
        handler=None,
    )
    
    registry.register(tool)
    
    return {
        "status": "registered",
        "name": request.name,
        "note": "Custom tools without handlers cannot be executed via /execute",
    }


@router.delete("/{name}")
async def unregister_tool(name: str):
    """Unregister a tool."""
    registry = get_tool_registry()
    
    if not registry.unregister(name):
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    
    return {"status": "unregistered", "name": name}


# =============================================================================
# OpenAI-Compatible Tools Format
# =============================================================================

@router.get("/openai/functions")
async def get_openai_functions():
    """Get tools in OpenAI function calling format.
    
    Use this to pass to chat completions with function calling enabled.
    """
    registry = get_tool_registry()
    return {"functions": registry.get_openai_tools()}
