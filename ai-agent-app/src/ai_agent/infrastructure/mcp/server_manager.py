"""MCP server management and discovery system."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4
import structlog

from .protocol import (
    MCPClient,
    MCPServerCapabilities,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPMessage,
)

logger = structlog.get_logger()


class MCPServerStatus(str, Enum):
    """MCP server status."""

    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class MCPServerType(str, Enum):
    """MCP server types."""

    PROCESS = "process"
    HTTP = "http"
    WEBSOCKET = "websocket"
    STDIO = "stdio"


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str | None = None
    server_type: MCPServerType = MCPServerType.PROCESS
    endpoint: str = ""
    command: list[str] | None = None
    env: dict[str, str] | None = None
    working_directory: str | None = None
    capabilities: MCPServerCapabilities | None = None
    status: MCPServerStatus = MCPServerStatus.UNKNOWN
    last_health_check: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    updated_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class MCPToolInfo:
    """Information about a tool from an MCP server."""

    server_id: str
    tool: MCPTool
    last_updated: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class MCPResourceInfo:
    """Information about a resource from an MCP server."""

    server_id: str
    resource: MCPResource
    last_updated: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class MCPPromptInfo:
    """Information about a prompt from an MCP server."""

    server_id: str
    prompt: MCPPrompt
    last_updated: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class MCPServerManager:
    """Manages MCP servers and their lifecycle."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerInfo] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> MCPToolInfo
        self._resources: dict[str, MCPResourceInfo] = (
            {}
        )  # resource_uri -> MCPResourceInfo
        self._prompts: dict[str, MCPPromptInfo] = {}  # prompt_name -> MCPPromptInfo
        self._health_check_interval = 30  # 30 seconds
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the MCP server manager."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("MCP server manager started")

    async def stop(self) -> None:
        """Stop the MCP server manager."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all servers
        async with self._lock:
            for server_id in list(self._servers.keys()):
                await self.stop_server(server_id)

        logger.info("MCP server manager stopped")

    async def register_server(
        self,
        name: str,
        server_type: MCPServerType,
        endpoint: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        working_directory: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new MCP server."""
        async with self._lock:
            server_info = MCPServerInfo(
                name=name,
                description=description,
                server_type=server_type,
                endpoint=endpoint,
                command=command,
                env=env,
                working_directory=working_directory,
                metadata=metadata or {},
            )

            self._servers[server_info.id] = server_info

            logger.info(
                "MCP server registered",
                server_id=server_info.id,
                name=name,
                type=server_type.value,
            )

            return server_info.id

    async def unregister_server(self, server_id: str) -> bool:
        """Unregister an MCP server."""
        async with self._lock:
            if server_id not in self._servers:
                return False

            # Stop the server if running
            await self.stop_server(server_id)

            # Remove from all registries
            del self._servers[server_id]

            # Remove associated tools, resources, and prompts
            self._tools = {
                name: info
                for name, info in self._tools.items()
                if info.server_id != server_id
            }
            self._resources = {
                uri: info
                for uri, info in self._resources.items()
                if info.server_id != server_id
            }
            self._prompts = {
                name: info
                for name, info in self._prompts.items()
                if info.server_id != server_id
            }

            logger.info("MCP server unregistered", server_id=server_id)
            return True

    async def start_server(self, server_id: str) -> bool:
        """Start an MCP server."""
        async with self._lock:
            if server_id not in self._servers:
                return False

            server_info = self._servers[server_id]

            if server_info.status in [
                MCPServerStatus.RUNNING,
                MCPServerStatus.STARTING,
            ]:
                return True

            try:
                server_info.status = MCPServerStatus.STARTING
                server_info.error_message = None

                if server_info.server_type == MCPServerType.PROCESS:
                    await self._start_process_server(server_info)
                elif server_info.server_type == MCPServerType.HTTP:
                    await self._start_http_server(server_info)
                elif server_info.server_type == MCPServerType.WEBSOCKET:
                    await self._start_websocket_server(server_info)
                elif server_info.server_type == MCPServerType.STDIO:
                    await self._start_stdio_server(server_info)
                else:
                    raise ValueError(
                        f"Unsupported server type: {server_info.server_type}"
                    )

                server_info.status = MCPServerStatus.RUNNING
                server_info.updated_at = asyncio.get_event_loop().time()

                logger.info(
                    "MCP server started", server_id=server_id, name=server_info.name
                )
                return True

            except Exception as e:
                server_info.status = MCPServerStatus.ERROR
                server_info.error_message = str(e)
                logger.error(
                    "Failed to start MCP server", server_id=server_id, error=str(e)
                )
                return False

    async def stop_server(self, server_id: str) -> bool:
        """Stop an MCP server."""
        async with self._lock:
            if server_id not in self._servers:
                return False

            server_info = self._servers[server_id]

            if server_info.status == MCPServerStatus.STOPPED:
                return True

            try:
                server_info.status = MCPServerStatus.STOPPING

                # Stop process if running
                if server_id in self._processes:
                    process = self._processes[server_id]
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                    del self._processes[server_id]

                # Close client connection
                if server_id in self._clients:
                    # Close client connection (implemented by subclasses)
                    await self._close_client(server_id)
                    del self._clients[server_id]

                server_info.status = MCPServerStatus.STOPPED
                server_info.updated_at = asyncio.get_event_loop().time()

                logger.info(
                    "MCP server stopped", server_id=server_id, name=server_info.name
                )
                return True

            except Exception as e:
                server_info.status = MCPServerStatus.ERROR
                server_info.error_message = str(e)
                logger.error(
                    "Failed to stop MCP server", server_id=server_id, error=str(e)
                )
                return False

    async def restart_server(self, server_id: str) -> bool:
        """Restart an MCP server."""
        await self.stop_server(server_id)
        await asyncio.sleep(1)  # Brief delay
        return await self.start_server(server_id)

    async def get_server(self, server_id: str) -> MCPServerInfo | None:
        """Get server information."""
        return self._servers.get(server_id)

    async def list_servers(self) -> list[MCPServerInfo]:
        """List all registered servers."""
        return list(self._servers.values())

    async def get_server_status(self, server_id: str) -> MCPServerStatus | None:
        """Get server status."""
        server = self._servers.get(server_id)
        return server.status if server else None

    async def health_check_server(self, server_id: str) -> bool:
        """Perform health check on a specific server."""
        if server_id not in self._servers:
            return False

        server_info = self._servers[server_id]

        try:
            if server_info.status != MCPServerStatus.RUNNING:
                return False

            # Check if client exists and is healthy
            if server_id in self._clients:
                client = self._clients[server_id]
                # Perform a simple health check (e.g., list tools)
                await client.list_tools()
                server_info.last_health_check = asyncio.get_event_loop().time()
                return True
            else:
                return False

        except Exception as e:
            logger.warning(
                "Server health check failed", server_id=server_id, error=str(e)
            )
            server_info.status = MCPServerStatus.DISCONNECTED
            server_info.error_message = str(e)
            return False

    async def get_tools(self, server_id: str | None = None) -> list[MCPToolInfo]:
        """Get tools from servers."""
        if server_id:
            return [
                info for info in self._tools.values() if info.server_id == server_id
            ]
        return list(self._tools.values())

    async def get_resources(
        self, server_id: str | None = None
    ) -> list[MCPResourceInfo]:
        """Get resources from servers."""
        if server_id:
            return [
                info for info in self._resources.values() if info.server_id == server_id
            ]
        return list(self._resources.values())

    async def get_prompts(self, server_id: str | None = None) -> list[MCPPromptInfo]:
        """Get prompts from servers."""
        if server_id:
            return [
                info for info in self._prompts.values() if info.server_id == server_id
            ]
        return list(self._prompts.values())

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], server_id: str | None = None
    ) -> dict[str, Any]:
        """Call a tool on a specific server."""
        tool_info = self._tools.get(tool_name)
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found")

        if server_id and tool_info.server_id != server_id:
            raise ValueError(f"Tool '{tool_name}' not found on server '{server_id}'")

        if tool_info.server_id not in self._clients:
            raise ValueError(f"Server '{tool_info.server_id}' not connected")

        client = self._clients[tool_info.server_id]
        return await client.call_tool(tool_name, arguments)

    async def _start_process_server(self, server_info: MCPServerInfo) -> None:
        """Start a process-based MCP server."""
        if not server_info.command:
            raise ValueError("Command is required for process server")

        # Prepare environment
        env = server_info.env or {}

        # Start the process
        process = await asyncio.create_subprocess_exec(
            *server_info.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            env=env,
            cwd=server_info.working_directory,
        )

        self._processes[server_info.id] = process

        # Create MCP client for stdio communication
        client = StdioMCPClient(process)
        await client.initialize()
        self._clients[server_info.id] = client

        # Discover capabilities and tools
        await self._discover_server_capabilities(server_info.id, client)

    async def _start_http_server(self, server_info: MCPServerInfo) -> None:
        """Start an HTTP-based MCP server."""
        # Implementation for HTTP MCP servers
        raise NotImplementedError("HTTP MCP servers not yet implemented")

    async def _start_websocket_server(self, server_info: MCPServerInfo) -> None:
        """Start a WebSocket-based MCP server."""
        # Implementation for WebSocket MCP servers
        raise NotImplementedError("WebSocket MCP servers not yet implemented")

    async def _start_stdio_server(self, server_info: MCPServerInfo) -> None:
        """Start a stdio-based MCP server."""
        # Similar to process server but with different command handling
        await self._start_process_server(server_info)

    async def _close_client(self, server_id: str) -> None:
        """Close client connection."""
        # Implementation depends on client type
        pass

    async def _discover_server_capabilities(
        self, server_id: str, client: MCPClient
    ) -> None:
        """Discover server capabilities and register tools/resources/prompts."""
        try:
            # Discover tools
            tools = await client.list_tools()
            for tool in tools:
                tool_info = MCPToolInfo(server_id=server_id, tool=tool)
                self._tools[tool.name] = tool_info

            # Discover resources
            resources = await client.list_resources()
            for resource in resources:
                resource_info = MCPResourceInfo(server_id=server_id, resource=resource)
                self._resources[resource.uri] = resource_info

            # Discover prompts
            prompts = await client.list_prompts()
            for prompt in prompts:
                prompt_info = MCPPromptInfo(server_id=server_id, prompt=prompt)
                self._prompts[prompt.name] = prompt_info

            logger.info(
                "Server capabilities discovered",
                server_id=server_id,
                tools=len(tools),
                resources=len(resources),
                prompts=len(prompts),
            )

        except Exception as e:
            logger.error(
                "Failed to discover server capabilities",
                server_id=server_id,
                error=str(e),
            )

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                async with self._lock:
                    for server_id in list(self._servers.keys()):
                        if self._servers[server_id].status == MCPServerStatus.RUNNING:
                            await self.health_check_server(server_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))


class StdioMCPClient(MCPClient):
    """MCP client for stdio communication."""

    def __init__(self, process: asyncio.subprocess.Process):
        super().__init__()
        self.process = process
        self._message_queue: asyncio.Queue[MCPMessage] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None

    async def _send_message(self, message: MCPMessage) -> None:
        """Send message via stdio."""
        if not self.process.stdin:
            raise RuntimeError("Process stdin not available")

        message_json = message.to_json() + "\n"
        self.process.stdin.write(message_json.encode())
        if hasattr(self.process.stdin, "drain"):
            await self.process.stdin.drain()

    async def _read_messages(self) -> None:
        """Read messages from stdio."""
        if not self.process.stdout:
            return

        while True:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break

                message_json = line.decode().strip()
                if message_json:
                    message = MCPMessage.from_json(message_json)
                    await self.handle_message(message)

            except Exception as e:
                logger.error("Error reading from process", error=str(e))
                break

    async def initialize(self) -> None:
        """Initialize the stdio client."""
        await super().initialize()

        # Start reading messages
        self._reader_task = asyncio.create_task(self._read_messages())
