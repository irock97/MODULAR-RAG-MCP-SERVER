"""Integration tests for MCP Server.

These tests verify the MCP server works correctly with Stdio transport.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestMCPServerBasics:
    """Basic tests for MCP Server."""

    def test_server_module_imports(self):
        """Test that server module can be imported."""
        from mcp_server import main, run_stdio_server
        from mcp_server.server import (
            _build_initialize_result,
            _build_error_response,
            _handle_request,
            main as server_main,
            run_stdio_server,
        )

        assert main is not None
        assert run_stdio_server is not None

    def test_build_initialize_result(self):
        """Test initialize result building."""
        from mcp_server.server import (
            _build_initialize_result,
            SERVER_NAME,
            SERVER_VERSION,
        )

        result = _build_initialize_result(None)

        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == SERVER_NAME
        assert result["serverInfo"]["version"] == SERVER_VERSION
        assert "capabilities" in result

    def test_build_initialize_result_with_params(self):
        """Test initialize result with custom params."""
        from mcp_server.server import _build_initialize_result

        params = {"protocolVersion": "2024-01-01", "capabilities": {}}
        result = _build_initialize_result(params)

        assert result["protocolVersion"] == "2024-01-01"

    def test_build_error_response(self):
        """Test error response building."""
        from mcp_server.server import _build_error_response

        response = _build_error_response(request_id=1, code=-32601, message="Method not found")

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method not found"

    def test_build_error_response_with_data(self):
        """Test error response with data."""
        from mcp_server.server import _build_error_response

        response = _build_error_response(
            request_id=2, code=-32603, message="Internal error", data="some error"
        )

        assert response["error"]["data"] == "some error"

    def test_handle_request_initialize(self):
        """Test handling initialize request."""
        from mcp_server.server import _handle_request

        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        response = _handle_request(request)

        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "serverInfo" in response["result"]

    def test_handle_request_tools_list(self):
        """Test handling tools/list request."""
        from mcp_server.server import _handle_request

        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        response = _handle_request(request)

        assert response is not None
        assert response["result"]["tools"] == []

    def test_handle_request_unknown_method(self):
        """Test handling unknown method."""
        from mcp_server.server import _handle_request

        request = {"jsonrpc": "2.0", "id": 3, "method": "unknown/method", "params": {}}
        response = _handle_request(request)

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_handle_notification_no_response(self):
        """Test that notifications don't get a response."""
        from mcp_server.server import _handle_request

        # Notification has no id
        request = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        response = _handle_request(request)

        assert response is None


class TestToolsRegistry:
    """Test tools registry functions."""

    def test_tools_registry_imports(self):
        """Test that tools registry can be imported."""
        from mcp_server.tools import (
            get_registered_tools,
            get_tool_handler,
            register_tool,
        )

        assert register_tool is not None
        assert get_tool_handler is not None
        assert get_registered_tools is not None

    def test_tool_registration_and_retrieval(self):
        """Test tool registration and retrieval."""
        from mcp_server.tools import (
            _tool_registry,
            get_registered_tools,
            get_tool_handler,
            register_tool,
        )

        # Clear registry
        _tool_registry.clear()

        async def test_handler(args):
            return f"Hello, {args.get('name', 'world')}"

        register_tool(
            name="greet",
            handler=test_handler,
            description="Greet someone",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        handler = get_tool_handler("greet")
        assert handler is not None

        tools = get_registered_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"

        # Cleanup
        _tool_registry.clear()

    def test_handle_tools_call_with_handler(self):
        """Test tools/call with registered handler."""
        from mcp_server.server import _handle_request
        from mcp_server.tools import _tool_registry

        # Clear and register test tool
        _tool_registry.clear()

        async def greet_handler(args):
            return f"Hello, {args.get('name', 'World')}"

        from mcp_server.tools import register_tool
        register_tool("greet", greet_handler, "Greeting tool")

        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "greet", "arguments": {"name": "Alice"}},
        }
        response = _handle_request(request)

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]

        # Cleanup
        _tool_registry.clear()

    def test_handle_tools_call_unknown_tool(self):
        """Test tools/call with unknown tool."""
        from mcp_server.server import _handle_request
        from mcp_server.tools import _tool_registry

        _tool_registry.clear()

        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        }
        response = _handle_request(request)

        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32601


class TestMCPServerExecution:
    """Tests that run MCP server as subprocess."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return project root."""
        return Path(__file__).parent.parent.parent

    def test_server_can_be_invoked(self, project_root):
        """Test that server module can be invoked."""
        env = {"PYTHONPATH": str(project_root / "src")}

        result = subprocess.run(
            [sys.executable, "-c", "from mcp_server import main; print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env={**subprocess.os.environ, **env},
            timeout=10,
        )

        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
