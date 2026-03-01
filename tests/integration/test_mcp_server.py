"""Integration tests for MCP Server and Protocol Handler.

These tests verify the MCP server works correctly with Stdio transport
and the protocol handler implements MCP correctly using official SDK.
"""

import pytest
from mcp import types


class TestProtocolHandler:
    """Tests for ProtocolHandler."""

    def test_protocol_handler_imports(self):
        """Test that protocol handler can be imported."""
        from mcp_server.protocol_handler import (
            ProtocolHandler,
            JSONRPCErrorCodes,
            ToolDefinition,
            create_mcp_server,
            get_protocol_handler,
        )

        assert ProtocolHandler is not None
        assert JSONRPCErrorCodes.INVALID_REQUEST == -32600
        assert JSONRPCErrorCodes.METHOD_NOT_FOUND == -32601
        assert JSONRPCErrorCodes.INVALID_PARAMS == -32602
        assert JSONRPCErrorCodes.INTERNAL_ERROR == -32603

    def test_protocol_handler_creation(self):
        """Test creating a ProtocolHandler instance."""
        from mcp_server.protocol_handler import ProtocolHandler

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )

        assert handler.server_name == "test-server"
        assert handler.server_version == "1.0.0"
        assert handler.tools == {}

    def test_get_server_info(self):
        """Test getting server info."""
        from mcp_server.protocol_handler import ProtocolHandler

        handler = ProtocolHandler(
            server_name="modular-rag-mcp-server",
            server_version="1.0.0"
        )

        server_info = handler.get_server_info()
        assert server_info["name"] == "modular-rag-mcp-server"
        assert server_info["version"] == "1.0.0"

    def test_get_capabilities(self):
        """Test getting server capabilities."""
        from mcp_server.protocol_handler import ProtocolHandler

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )

        caps = handler.get_capabilities()
        assert "tools" in caps
        assert caps["tools"] == {}

    def test_get_capabilities_with_tools(self):
        """Test getting server capabilities with registered tools."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def test_handler(args):
            return "test result"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=test_handler,
        )

        caps = handler.get_capabilities()
        assert "tools" in caps
        assert caps["tools"] == {}


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool(self):
        """Test registering a tool."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def greet_handler(args):
            return f"Hello, {args.get('name', 'World')}"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="greet",
            description="Greet someone",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=greet_handler,
        )

        assert "greet" in handler.tools
        assert handler.tools["greet"].name == "greet"
        assert handler.tools["greet"].description == "Greet someone"

    def test_register_duplicate_tool_raises(self):
        """Test registering duplicate tool raises ValueError."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def handler1(args):
            return "handler1"

        async def handler2(args):
            return "handler2"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="duplicate",
            description="First handler",
            input_schema={"type": "object"},
            handler=handler1,
        )

        with pytest.raises(ValueError, match="already registered"):
            handler.register_tool(
                name="duplicate",
                description="Second handler",
                input_schema={"type": "object"},
                handler=handler2,
            )

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def greet_handler(args):
            return "Hello"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="greet",
            description="Greet someone",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            handler=greet_handler,
        )

        schemas = handler.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "greet"
        assert schemas[0].description == "Greet someone"


class TestToolExecution:
    """Test tool execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test executing a tool successfully."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def greet_handler(args):
            return f"Hello, {args.get('name', 'World')}"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="greet",
            description="Greet someone",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            handler=greet_handler,
        )

        result = await handler.execute_tool("greet", {"name": "Alice"})
        assert isinstance(result, types.CallToolResult)
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "Hello, Alice"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing unknown tool returns error."""
        from mcp_server.protocol_handler import ProtocolHandler

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )

        result = await handler.execute_tool("nonexistent", {})
        assert isinstance(result, types.CallToolResult)
        assert result.isError is True
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_params(self):
        """Test executing tool with invalid params returns error."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def broken_handler(args):
            # This handler expects 'name' but we'll pass wrong args
            return args["required_field"]

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="broken",
            description="A broken tool",
            input_schema={"type": "object"},
            handler=broken_handler,
        )

        result = await handler.execute_tool("broken", {})
        assert isinstance(result, types.CallToolResult)
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_execute_tool_returns_string(self):
        """Test tool returning string."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def echo_handler(args):
            return "echo result"

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="echo",
            description="Echo back input",
            input_schema={"type": "object"},
            handler=echo_handler,
        )

        result = await handler.execute_tool("echo", {"text": "hello"})
        assert isinstance(result, types.CallToolResult)
        assert result.isError is False
        assert result.content[0].text == "echo result"

    @pytest.mark.asyncio
    async def test_execute_tool_returns_dict(self):
        """Test tool returning dict."""
        from mcp_server.protocol_handler import ProtocolHandler

        async def data_handler(args):
            return {"key": "value", "number": 42}

        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0"
        )
        handler.register_tool(
            name="data",
            description="Return data",
            input_schema={"type": "object"},
            handler=data_handler,
        )

        result = await handler.execute_tool("data", {})
        assert isinstance(result, types.CallToolResult)
        assert result.isError is False
        # Dict should be converted to string
        assert "key" in result.content[0].text
        assert "value" in result.content[0].text


class TestMCPServerCreation:
    """Test MCP server creation."""

    def test_create_mcp_server(self):
        """Test creating an MCP server."""
        from mcp_server.protocol_handler import (
            create_mcp_server,
            get_protocol_handler,
        )

        server = create_mcp_server(
            server_name="test-server",
            server_version="1.0.0",
        )

        assert server is not None
        handler = get_protocol_handler(server)
        assert handler.server_name == "test-server"
        assert handler.server_version == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
