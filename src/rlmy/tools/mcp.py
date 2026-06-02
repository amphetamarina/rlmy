"""
Purpose: MCP server connection management and tool wrapping.
Usage:
    from rlmy.tools.mcp import MCPConnector
    async with MCPConnector() as mcp_tools:
        # mcp_tools is dict[str, MCPToolWrapper]
        all_tools = standard_tools + list(mcp_tools.values())
Key Components:
    MCPConnector — async context manager, loads config from mcp_servers.json
    MCPToolWrapper — callable with typed inspect.Signature for DSPy Tool() integration
Conventions:
    Self-contained — depends only on mcp, asyncio, inspect, json, logging, re, pathlib.
    No dependency on cli_proto, mainmcp, or any other prototype module.
    Config loaded from mcp_servers.json in project root (gitignored, see mcp_servers.example.json).
"""

import asyncio
import inspect
import json
import logging
import re
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

__all__ = ["MCPConnector", "MCPToolWrapper"]

logger = logging.getLogger(__name__)

# Default path: ~/.config/rlmy/mcp_servers.json
# Centralized in the user's config directory (consistent with config.toml and
# sandboxes). Works regardless of install method (uv tool install, pip, editable).
# Falls back to project-root mcp_servers.json for dev convenience if present.
_USER_CONFIG_PATH = Path.home() / ".config" / "rlmy" / "mcp_servers.json"
_DEV_CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp_servers.json"
_DEFAULT_CONFIG_PATH = _USER_CONFIG_PATH if _USER_CONFIG_PATH.exists() else _DEV_CONFIG_PATH


# JSON Schema type -> Python type mapping
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _sanitize_param_name(name: str) -> str:
    """
    Purpose: Convert MCP parameter name to valid Python identifier

    Args:
        name: Original parameter name (may contain dots, dashes, etc.)

    Returns:
        Sanitized name safe for inspect.Parameter

    Conventions:
        - Replaces dots and dashes with underscores
        - Prepends underscore if name starts with digit
    """
    sanitized = re.sub(r"[.\-]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


class MCPToolWrapper:
    """
    Purpose: Callable wrapper that exposes typed signature to inspect.signature()

    Attributes:
        _session: MCP ClientSession for making calls
        _tool_name: Original tool name on the MCP server
        _param_name_map: Maps sanitized param names -> original MCP names
        __name__: Python-friendly function name
        __doc__: Docstring with tool description
        __signature__: inspect.Signature derived from MCP schema

    Usage Patterns:
        wrapper = MCPToolWrapper(session, "search", "mcp_slack", tool_schema)
        result = wrapper(query="hello")

    Conventions:
        - inspect.signature(wrapper) returns typed params from MCP schema
        - No exec/eval - uses Python's inspect.Parameter and Signature
        - Sync call via nest_asyncio-enabled event loop
        - Sanitizes param names with dots/dashes (e.g., 'assignee.any' -> 'assignee_any')
    """

    def __init__(self, session, tool_name: str, server_name: str, tool_schema: dict):
        self._session = session
        self._tool_name = tool_name
        self._param_name_map = {}  # sanitized -> original
        self.__name__ = f"{server_name}_{tool_name}"
        self.__doc__ = self._build_docstring(tool_name, server_name, tool_schema)
        self.__signature__ = self._build_signature(tool_schema)

    def _build_signature(self, schema: dict) -> inspect.Signature:
        """
        Purpose: Build inspect.Signature from MCP JSON schema

        Args:
            schema: inputSchema from MCP tool (JSON Schema format)

        Returns:
            inspect.Signature with typed parameters

        Conventions:
            - Sanitizes param names to valid Python identifiers
            - Stores mapping in _param_name_map for translation at call time
        """
        properties = schema.get("properties", {}) if schema else {}
        required = set(schema.get("required", [])) if schema else set()

        params = []
        for original_name, prop_def in properties.items():
            # Sanitize name for Python, store mapping for call-time translation
            sanitized_name = _sanitize_param_name(original_name)
            self._param_name_map[sanitized_name] = original_name

            py_type = _TYPE_MAP.get(prop_def.get("type", "string"), str)

            if original_name in required:
                param = inspect.Parameter(
                    sanitized_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=py_type,
                )
            else:
                param = inspect.Parameter(
                    sanitized_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=py_type,
                )
            params.append(param)

        # Sort: required first (no default), then optional (has default)
        params.sort(key=lambda p: p.default is not inspect.Parameter.empty)

        return inspect.Signature(params, return_annotation=str)

    def _build_docstring(self, tool_name: str, server_name: str, schema: dict) -> str:
        """Build docstring from MCP schema descriptions"""
        properties = schema.get("properties", {}) if schema else {}
        required = set(schema.get("required", [])) if schema else set()

        lines = [f"MCP tool: {tool_name} from {server_name}"]
        if properties:
            lines.append("")
            lines.append("Args:")
            for name, prop_def in properties.items():
                desc = prop_def.get("description", "No description")
                req_marker = " (required)" if name in required else " (optional)"
                lines.append(f"    {name}: {desc}{req_marker}")
        return "\n".join(lines)

    def __call__(self, *args, **kwargs) -> str:
        """
        Purpose: Sync call to MCP tool via session.call_tool()

        Args:
            *args: Positional arguments (mapped to param names via signature)
            **kwargs: Keyword arguments matching the tool's schema

        Returns:
            Tool result as string, or error message

        Conventions:
            - Translates sanitized param names back to original MCP names
        """
        # Map positional args to parameter names using signature
        param_names = list(self.__signature__.parameters.keys())
        for i, arg in enumerate(args):
            if i < len(param_names):
                kwargs[param_names[i]] = arg

        # Translate sanitized names back to original MCP names and filter None
        arguments = {}
        for sanitized_name, value in kwargs.items():
            if value is not None:
                original_name = self._param_name_map.get(sanitized_name, sanitized_name)
                arguments[original_name] = value

        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self._session.call_tool(self._tool_name, arguments=arguments)
            )
            if hasattr(result, "content") and result.content:
                return str(
                    result.content[0].text
                    if hasattr(result.content[0], "text")
                    else result.content[0]
                )
            return str(result)
        except Exception as e:
            raise RuntimeError(f"MCP tool error: {str(e)}") from e


class MCPConnector:
    """
    Purpose: Manage persistent MCP server connections and create wrapper functions.
    Loads server configuration from mcp_servers.json (see mcp_servers.example.json).

    Attributes:
        servers: StdioServerParameters per server name, loaded from JSON config
        tool_mapping: Whitelist of tool names per server, loaded from JSON config
        sessions_map: Maps server names to their active sessions
        contexts: Context managers for connection cleanup

    Usage Patterns:
        async with MCPConnector() as mcp_tools:
            # mcp_tools: dict[str, MCPToolWrapper]
            ...
    Conventions:
        Creates MCPToolWrapper instances that respect inspect.signature().
        If config file is missing or malformed, degrades gracefully (no tools, no crash).
    """

    def __init__(self, config_path: Path | None = None):
        config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self.servers: dict[str, StdioServerParameters] = {}
        self.tool_mapping: dict[str, list[str]] = {}
        self.sessions_map: dict = {}
        self.contexts: list = []

        if not config_path.exists():
            logger.warning(
                "MCP config not found at %s — no MCP servers will load. "
                "Copy mcp_servers.example.json to mcp_servers.json to configure.",
                config_path,
            )
            return

        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to read MCP config %s: %s — no MCP servers will load.",
                config_path, e,
            )
            return

        for name, cfg in raw.get("servers", {}).items():
            # "command" is required — crash loudly on config authoring error
            params = {"command": cfg["command"], "args": cfg.get("args", [])}
            if "env" in cfg:
                params["env"] = cfg["env"]
            self.servers[name] = StdioServerParameters(**params)
            if "tools" in cfg:
                self.tool_mapping[name] = cfg["tools"]

    def _create_mcp_wrapper(
        self, session, tool_name: str, server_name: str, tool_schema: dict
    ) -> MCPToolWrapper:
        """
        Purpose: Create MCPToolWrapper instance from MCP tool schema

        Args:
            session: MCP ClientSession
            tool_name: Original tool name from server
            server_name: Server identifier
            tool_schema: inputSchema from the MCP tool (JSON Schema format)

        Returns:
            MCPToolWrapper with proper __signature__ for inspect.signature()
        """
        return MCPToolWrapper(session, tool_name, server_name, tool_schema)

    async def __aenter__(self):
        """
        Purpose: Connect to all MCP servers and create wrapper functions
        Returns: Dictionary of tool_name -> callable wrapper functions
        """
        mcp_tools = {}

        for server_name, server_params in self.servers.items():
            try:
                logger.info("Connecting to MCP server: %s", server_name)

                # Create persistent connection
                stdio_ctx = stdio_client(server_params)
                read, write = await stdio_ctx.__aenter__()

                session_ctx = ClientSession(read, write)
                session = await session_ctx.__aenter__()
                await session.initialize()

                # Store contexts for cleanup
                self.contexts.append((stdio_ctx, session_ctx))
                self.sessions_map[server_name] = session

                # List available tools from this server
                response = await session.list_tools()
                available_tools = {t.name for t in response.tools}

                # Create wrappers for allowed tools
                if server_name in self.tool_mapping:
                    for tool_name in self.tool_mapping[server_name]:
                        if tool_name in available_tools:
                            # Find the tool object to get its schema
                            tool_obj = next(
                                (t for t in response.tools if t.name == tool_name), None
                            )
                            tool_schema = (
                                tool_obj.inputSchema
                                if tool_obj and hasattr(tool_obj, "inputSchema")
                                else None
                            )

                            wrapper = self._create_mcp_wrapper(
                                session, tool_name, server_name, tool_schema
                            )
                            mcp_tools[wrapper.__name__] = wrapper
                            logger.info("Loaded tool: %s", wrapper.__name__)
                        else:
                            logger.warning("Tool not found on %s: %s", server_name, tool_name)

            except Exception as e:
                logger.warning("Could not connect to %s: %s", server_name, e)
                continue

        if not mcp_tools:
            logger.warning("No MCP tools loaded. Proceeding with standard tools only.")

        return mcp_tools

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Purpose: Clean up all MCP connections
        """
        logger.debug("Closing MCP connections...")
        for stdio_ctx, session_ctx in reversed(self.contexts):
            try:
                await session_ctx.__aexit__(None, None, None)
                await stdio_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing MCP connection: %s", e)
