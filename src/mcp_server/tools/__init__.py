# MCP Server Tools - Tool registry and handlers

from typing import Any, Callable

# Tool registry: name -> (handler, schema)
_tool_registry: dict[str, tuple[Callable[[dict[str, Any]], Any], dict[str, Any]]] = {}


def register_tool(
    name: str,
    handler: Callable[[dict[str, Any]], Any],
    description: str = "",
    input_schema: dict[str, Any] | None = None,
) -> None:
    """Register a tool with the MCP server.

    Args:
        name: Tool name
        handler: Async function to handle tool calls, receives arguments dict
        description: Tool description
        input_schema: JSON Schema for tool input
    """
    schema = input_schema or {"type": "object", "properties": {}}
    schema["name"] = name
    schema["description"] = description

    _tool_registry[name] = (handler, schema)
    # Import logger here to avoid circular import
    from observability.logger import get_logger
    logger = get_logger(__name__)
    logger.info(f"Tool registered: {name}")


def get_tool_handler(name: str) -> Callable[[dict[str, Any]], Any] | None:
    """Get tool handler by name.

    Args:
        name: Tool name

    Returns:
        Handler function or None if not found
    """
    entry = _tool_registry.get(name)
    if entry:
        return entry[0]
    return None


def get_registered_tools() -> list[dict[str, Any]]:
    """Get list of all registered tools.

    Returns:
        List of tool schemas
    """
    tools = []
    for name, (_, schema) in _tool_registry.items():
        tools.append({
            "name": name,
            "description": schema.get("description", ""),
            "inputSchema": schema.get("input_schema", schema),
        })
    return tools
