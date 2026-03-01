"""MCP Server entry point for stdio transport.

This module provides the MCP server that uses the official MCP SDK
for protocol handling and stdio transport.

Design Principles:
    - Stdio Transport: stdout 仅输出 MCP 协议消息，logs 到 stderr
    - Minimal: 使用官方 mcp SDK 处理协议细节
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp_server.protocol_handler import (
    ProtocolHandler,
    create_mcp_server,
    get_protocol_handler,
)


async def run_stdio_server() -> int:
    """Run MCP server over stdio using official MCP SDK."""
    from observability.logger import get_logger

    logger = get_logger("mcp_server.stdio")
    logger.info("Starting MCP server (stdio transport with MCP SDK).")

    # Create protocol handler
    protocol_handler = ProtocolHandler(
        server_name="modular-rag-mcp-server",
        server_version="1.0.0",
    )

    # Create MCP server with protocol handler
    server = create_mcp_server(
        server_name="modular-rag-mcp-server",
        server_version="1.0.0",
        protocol_handler=protocol_handler,
    )

    # Run the server with stdio
    from mcp.server.stdio import stdio_server

    read_stream, write_stream = stdio_server()
    await server.run(
        read_stream,
        write_stream,
        server.create_initialization_options(),
    )

    logger.info("MCP server shutting down.")
    return 0


def main() -> int:
    """Entry point for stdio MCP server."""
    try:
        return asyncio.run(run_stdio_server())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
