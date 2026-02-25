"""
ZSE MCP Module - Model Context Protocol support for tool calling.

Provides:
- Tool definitions (JSON schema)
- Function calling parsing
- Built-in tools (calculator, datetime, etc.)
- External MCP server connections (future)

MCP Reference: https://modelcontextprotocol.io/
"""

import json
import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from threading import Lock


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """A tool that can be called by the model."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if self.handler is None:
            raise ValueError(f"Tool '{self.name}' has no handler")
        return self.handler(**kwargs)


@dataclass
class ToolCall:
    """A tool call request from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    name: str
    result: Any
    error: Optional[str] = None
    
    def to_message(self) -> Dict[str, Any]:
        """Convert to message format for chat."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": json.dumps(self.result) if not self.error else f"Error: {self.error}",
        }


class ToolRegistry:
    """Registry and executor for tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._lock = Lock()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        # Calculator
        self.register(Tool(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), powers (**), and common functions (sqrt, sin, cos, tan, log, exp, abs, round).",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate, e.g., '2 + 2 * 3' or 'sqrt(16) + log(100)'",
                ),
            ],
            handler=self._calc_handler,
        ))
        
        # DateTime
        self.register(Tool(
            name="datetime",
            description="Get current date and time, or perform date/time calculations.",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    enum=["now", "date", "time", "timestamp", "weekday"],
                    default="now",
                    required=False,
                ),
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone (e.g., 'UTC', 'America/New_York'). Default is UTC.",
                    required=False,
                    default="UTC",
                ),
            ],
            handler=self._datetime_handler,
        ))
        
        # JSON Parser
        self.register(Tool(
            name="parse_json",
            description="Parse and extract data from JSON strings.",
            parameters=[
                ToolParameter(
                    name="json_string",
                    type="string",
                    description="JSON string to parse",
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Optional JSONPath-like path to extract (e.g., 'data.items[0].name')",
                    required=False,
                ),
            ],
            handler=self._json_handler,
        ))
        
        # String Operations
        self.register(Tool(
            name="string_ops",
            description="Perform string operations like uppercase, lowercase, split, join, replace.",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    enum=["upper", "lower", "title", "split", "join", "replace", "length", "reverse"],
                ),
                ToolParameter(
                    name="text",
                    type="string",
                    description="Text to operate on",
                ),
                ToolParameter(
                    name="arg1",
                    type="string",
                    description="First argument (delimiter for split/join, search for replace)",
                    required=False,
                ),
                ToolParameter(
                    name="arg2",
                    type="string",
                    description="Second argument (replacement for replace)",
                    required=False,
                ),
            ],
            handler=self._string_handler,
        ))
    
    # -------------------------------------------------------------------------
    # Built-in Handlers
    # -------------------------------------------------------------------------
    
    def _calc_handler(self, expression: str) -> Dict[str, Any]:
        """Calculator handler with safe math evaluation."""
        # Define allowed functions
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log10,
            "ln": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
        }
        
        # Sanitize expression - allow only safe characters
        sanitized = re.sub(r'[^0-9+\-*/().,%\s\w]', '', expression)
        
        try:
            result = eval(sanitized, {"__builtins__": {}}, safe_dict)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}
    
    def _datetime_handler(
        self,
        operation: str = "now",
        timezone: str = "UTC",
    ) -> Dict[str, Any]:
        """DateTime handler."""
        try:
            # Get current time in UTC
            now = datetime.now(tz=timezone_map.get(timezone.upper(), timezone_map["UTC"]))
            
            if operation == "now":
                return {"datetime": now.isoformat(), "timezone": timezone}
            elif operation == "date":
                return {"date": now.strftime("%Y-%m-%d"), "timezone": timezone}
            elif operation == "time":
                return {"time": now.strftime("%H:%M:%S"), "timezone": timezone}
            elif operation == "timestamp":
                return {"timestamp": int(now.timestamp()), "timezone": timezone}
            elif operation == "weekday":
                return {
                    "weekday": now.strftime("%A"),
                    "day_number": now.isoweekday(),
                    "timezone": timezone,
                }
            else:
                return {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _json_handler(
        self,
        json_string: str,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """JSON parser handler."""
        try:
            data = json.loads(json_string)
            
            if path:
                # Simple path navigation
                result = data
                for key in path.split('.'):
                    # Handle array indexing
                    if '[' in key and key.endswith(']'):
                        base, idx = key.rstrip(']').split('[')
                        if base:
                            result = result[base]
                        result = result[int(idx)]
                    else:
                        result = result[key]
                return {"path": path, "result": result}
            
            return {"parsed": data}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}
        except (KeyError, IndexError, TypeError) as e:
            return {"error": f"Path error: {e}"}
    
    def _string_handler(
        self,
        operation: str,
        text: str,
        arg1: Optional[str] = None,
        arg2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """String operations handler."""
        try:
            if operation == "upper":
                return {"result": text.upper()}
            elif operation == "lower":
                return {"result": text.lower()}
            elif operation == "title":
                return {"result": text.title()}
            elif operation == "split":
                delimiter = arg1 or " "
                return {"result": text.split(delimiter)}
            elif operation == "join":
                # Expecting text to be a JSON array string
                items = json.loads(text) if text.startswith('[') else text.split()
                delimiter = arg1 or " "
                return {"result": delimiter.join(str(i) for i in items)}
            elif operation == "replace":
                if not arg1:
                    return {"error": "replace requires search argument (arg1)"}
                return {"result": text.replace(arg1, arg2 or "")}
            elif operation == "length":
                return {"result": len(text)}
            elif operation == "reverse":
                return {"result": text[::-1]}
            else:
                return {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"error": str(e)}
    
    # -------------------------------------------------------------------------
    # Registry Operations
    # -------------------------------------------------------------------------
    
    def register(self, tool: Tool):
        """Register a tool."""
        with self._lock:
            self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        
        if tool is None:
            return ToolResult(
                tool_call_id=str(uuid.uuid4()),
                name=name,
                result=None,
                error=f"Tool '{name}' not found",
            )
        
        try:
            result = tool.execute(**arguments)
            return ToolResult(
                tool_call_id=str(uuid.uuid4()),
                name=name,
                result=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=str(uuid.uuid4()),
                name=name,
                result=None,
                error=str(e),
            )
    
    def parse_tool_calls(self, content: str) -> List[ToolCall]:
        """Parse tool calls from model output.
        
        Supports formats:
        - OpenAI: {"name": "...", "arguments": {...}}
        - Claude: <tool_call name="...">...</tool_call>
        - Generic: ```tool\n{"name": "...", "arguments": {...}}\n```
        """
        tool_calls = []
        
        # Try OpenAI JSON format
        try:
            # Look for JSON objects with name and arguments
            json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
            for match in re.finditer(json_pattern, content, re.DOTALL):
                name = match.group(1)
                args_str = match.group(2)
                try:
                    args = json.loads(args_str)
                    tool_calls.append(ToolCall(
                        id=str(uuid.uuid4()),
                        name=name,
                        arguments=args,
                    ))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        
        # Try code block format
        code_pattern = r'```(?:tool|json)?\s*\n?\s*(\{.*?\})\s*\n?```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    tool_calls.append(ToolCall(
                        id=str(uuid.uuid4()),
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                    ))
            except json.JSONDecodeError:
                pass
        
        # Try XML-like format
        xml_pattern = r'<tool_call\s+name="([^"]+)"[^>]*>(.*?)</tool_call>'
        for match in re.finditer(xml_pattern, content, re.DOTALL):
            name = match.group(1)
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str) if args_str else {}
                tool_calls.append(ToolCall(
                    id=str(uuid.uuid4()),
                    name=name,
                    arguments=args,
                ))
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    
    def process_tool_calls(
        self,
        content: str,
        auto_execute: bool = True,
    ) -> Tuple[List[ToolCall], List[ToolResult]]:
        """Parse and optionally execute tool calls from model output.
        
        Returns:
            (tool_calls, tool_results)
        """
        tool_calls = self.parse_tool_calls(content)
        results = []
        
        if auto_execute:
            for call in tool_calls:
                result = self.execute(call.name, call.arguments)
                result.tool_call_id = call.id
                results.append(result)
        
        return tool_calls, results


# Timezone mapping
timezone_map = {
    "UTC": timezone.utc,
}

# Try to add more timezones if zoneinfo is available
try:
    from zoneinfo import ZoneInfo
    timezone_map.update({
        "EST": ZoneInfo("America/New_York"),
        "PST": ZoneInfo("America/Los_Angeles"),
        "CST": ZoneInfo("America/Chicago"),
        "MST": ZoneInfo("America/Denver"),
        "GMT": ZoneInfo("Europe/London"),
        "CET": ZoneInfo("Europe/Paris"),
        "JST": ZoneInfo("Asia/Tokyo"),
    })
except ImportError:
    pass


# Global instance
_tool_registry: Optional[ToolRegistry] = None
_registry_lock = Lock()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _tool_registry
    with _registry_lock:
        if _tool_registry is None:
            _tool_registry = ToolRegistry()
        return _tool_registry


def register_tool(
    name: str,
    description: str,
    parameters: List[Dict[str, Any]],
    handler: Callable[..., Any],
) -> Tool:
    """Register a new tool.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: List of parameter dicts with name, type, description, required
        handler: Function to call when tool is executed
    
    Returns:
        The registered Tool
    """
    registry = get_tool_registry()
    
    params = [
        ToolParameter(
            name=p["name"],
            type=p.get("type", "string"),
            description=p.get("description", ""),
            required=p.get("required", True),
            enum=p.get("enum"),
            default=p.get("default"),
        )
        for p in parameters
    ]
    
    tool = Tool(
        name=name,
        description=description,
        parameters=params,
        handler=handler,
    )
    
    registry.register(tool)
    return tool
