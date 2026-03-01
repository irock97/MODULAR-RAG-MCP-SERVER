"""MCP Server entry point for stdio transport.

This module implements a minimal JSON-RPC 2.0 loop that supports
the MCP `initialize` request. It ensures stdout only contains
protocol messages while all logs go to stderr.

Design Principles:
    - Protocol-First: 严格遵循 MCP 官方规范 (JSON-RPC 2.0)
    - Stdio Transport: stdout 仅输出 MCP 协议消息，logs 到 stderr
    - Minimal: 不依赖 MCP SDK，手动实现协议
"""

from __future__ import annotations

import json
import sys
from typing import Any

from observability.logger import get_logger


DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "1.0.0"


def _build_initialize_result(params: dict[str, Any] | None) -> dict[str, Any]:
    """Build MCP initialize result payload.

    Args:
        params: Initialize request parameters.

    Returns:
        Result payload for the initialize response.
    """
    params = params or {}
    protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": protocol_version,
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {}},
    }


def _write_response(payload: dict[str, Any]) -> None:
    """Write a JSON-RPC response to stdout.

    Args:
        payload: JSON-RPC response payload.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _build_error_response(request_id: Any, code: int, message: str, data: str | None = None) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    Args:
        request_id: Request ID
        code: Error code
        message: Error message
        data: Optional error data

    Returns:
        JSON-RPC error response
    """
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error
    }


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC request.

    Args:
        request: Parsed JSON-RPC request.

    Returns:
        JSON-RPC response payload, or None for notifications.
    """
    method = request.get("method")
    request_id = request.get("id")

    # Handle initialize
    if method == "initialize":
        result = _build_initialize_result(request.get("params"))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # Handle tools/list
    if method == "tools/list":
        from mcp_server.tools import get_registered_tools
        tools = get_registered_tools()
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}

    # Handle tools/call
    if method == "tools/call":
        from mcp_server.tools import get_tool_handler
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handler = get_tool_handler(tool_name)
        if handler is None:
            return _build_error_response(request_id, -32601, f"Tool not found: {tool_name}")

        try:
            import asyncio
            result = asyncio.run(handler(arguments))
            # Convert result to MCP content format
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            else:
                content = [{"type": "text", "text": str(result)}]
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": content}}
        except Exception as e:
            return _build_error_response(request_id, -32603, f"Internal error: {str(e)}")

    # Notifications don't expect a response
    if request_id is None:
        return None

    # Unknown method
    return _build_error_response(request_id, -32601, "Method not found")


def run_stdio_server() -> int:
    """Run MCP server over stdio.

    Returns:
        Exit code.
    """
    logger = get_logger("mcp_server.stdio")
    logger.info("Starting MCP server (stdio transport).")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received on stdin.")
            continue

        response = _handle_request(request)
        if response is not None:
            _write_response(response)
            logger.info("Handled request: %s", request.get("method"))

    logger.info("MCP server shutting down.")
    return 0

"""MCP Server entry point for stdio transport.

This module implements a minimal JSON-RPC 2.0 loop that supports
the MCP `initialize` request. It ensures stdout only contains
protocol messages while all logs go to stderr.

Design Principles:
    - Protocol-First: 严格遵循 MCP 官方规范 (JSON-RPC 2.0)
    - Stdio Transport: stdout 仅输出 MCP 协议消息，logs 到 stderr
    - Minimal: 不依赖 MCP SDK，手动实现协议
"""

from __future__ import annotations

import json
import sys
from typing import Any

from observability.logger import get_logger


DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "1.0.0"


def _build_initialize_result(params: dict[str, Any] | None) -> dict[str, Any]:
    """Build MCP initialize result payload.

    Args:
        params: Initialize request parameters.

    Returns:
        Result payload for the initialize response.
    """
    params = params or {}
    protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": protocol_version,
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {}},
    }


def _write_response(payload: dict[str, Any]) -> None:
    """Write a JSON-RPC response to stdout.

    Args:
        payload: JSON-RPC response payload.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _build_error_response(request_id: Any, code: int, message: str, data: str | None = None) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    Args:
        request_id: Request ID
        code: Error code
        message: Error message
        data: Optional error data

    Returns:
        JSON-RPC error response
    """
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error
    }


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC request.

    Args:
        request: Parsed JSON-RPC request.

    Returns:
        JSON-RPC response payload, or None for notifications.
    """
    method = request.get("method")
    request_id = request.get("id")

    # Handle initialize
    if method == "initialize":
        result = _build_initialize_result(request.get("params"))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # Handle tools/list
    if method == "tools/list":
        from mcp_server.tools import get_registered_tools
        tools = get_registered_tools()
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}

    # Handle tools/call
    if method == "tools/call":
        from mcp_server.tools import get_tool_handler
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handler = get_tool_handler(tool_name)
        if handler is None:
            return _build_error_response(request_id, -32601, f"Tool not found: {tool_name}")

        try:
            import asyncio
            result = asyncio.run(handler(arguments))
            # Convert result to MCP content format
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            else:
                content = [{"type": "text", "text": str(result)}]
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": content}}
        except Exception as e:
            return _build_error_response(request_id, -32603, f"Internal error: {str(e)}")

    # Notifications don't expect a response
    if request_id is None:
        return None

    # Unknown method
    return _build_error_response(request_id, -32601, "Method not found")


def run_stdio_server() -> int:
    """Run MCP server over stdio.

    Returns:
        Exit code.
    """
    logger = get_logger("mcp_server.stdio")
    logger.info("Starting MCP server (stdio transport).")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received on stdin.")
            continue

        response = _handle_request(request)
        if response is not None:
            _write_response(response)
            logger.info("Handled request: %s", request.get("method"))

    logger.info("MCP server shutting down.")
    return 0


def main() -> int:
    """Entry point for stdio MCP server."""
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())

def main() -> int:
    """Entry point for stdio MCP server."""
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
