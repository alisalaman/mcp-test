"""MCP client with connection pooling and management."""

import asyncio
from dataclasses import dataclass
from typing import Any
import structlog

from .protocol import MCPClient, MCPMessage
from .server_manager import MCPServerInfo, MCPServerStatus

logger = structlog.get_logger()


@dataclass
class MCPConnection:
    """MCP connection information."""

    server_id: str
    client: MCPClient
    created_at: float
    last_used: float
    is_healthy: bool = True
    error_count: int = 0
    max_errors: int = 5


class MCPConnectionPool:
    """Connection pool for MCP clients."""

    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the connection pool."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("MCP connection pool started")

    async def stop(self) -> None:
        """Stop the connection pool."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for connection in self._connections.values():
                await self._close_connection(connection)
            self._connections.clear()

        logger.info("MCP connection pool stopped")

    async def get_connection(self, server_id: str) -> MCPConnection | None:
        """Get a connection for a server."""
        async with self._lock:
            connection = self._connections.get(server_id)

            if connection and connection.is_healthy:
                connection.last_used = asyncio.get_event_loop().time()
                return connection

            return None

    async def create_connection(
        self, server_id: str, server_info: MCPServerInfo
    ) -> MCPConnection | None:
        """Create a new connection for a server."""
        async with self._lock:
            # Check if connection already exists
            if server_id in self._connections:
                return self._connections[server_id]

            # Check connection limit
            if len(self._connections) >= self.max_connections:
                # Remove oldest connection
                oldest_connection = min(
                    self._connections.values(), key=lambda c: c.last_used
                )
                await self._close_connection(oldest_connection)
                del self._connections[oldest_connection.server_id]

            try:
                # Create client based on server type
                if server_info.server_type.value == "process":
                    client = await self._create_process_client(server_info)
                elif server_info.server_type.value == "http":
                    client = await self._create_http_client(server_info)
                elif server_info.server_type.value == "websocket":
                    client = await self._create_websocket_client(server_info)
                else:
                    logger.error(
                        "Unsupported server type", server_type=server_info.server_type
                    )
                    return None

                # Initialize the client
                await client.initialize()

                # Create connection
                current_time = asyncio.get_event_loop().time()
                connection = MCPConnection(
                    server_id=server_id,
                    client=client,
                    created_at=current_time,
                    last_used=current_time,
                    is_healthy=True,
                )

                self._connections[server_id] = connection

                logger.info("MCP connection created", server_id=server_id)
                return connection

            except Exception as e:
                logger.error(
                    "Failed to create MCP connection", server_id=server_id, error=str(e)
                )
                return None

    async def close_connection(self, server_id: str) -> bool:
        """Close a specific connection."""
        async with self._lock:
            connection = self._connections.pop(server_id, None)
            if connection:
                await self._close_connection(connection)
                return True
            return False

    async def health_check_connection(self, server_id: str) -> bool:
        """Perform health check on a connection."""
        async with self._lock:
            connection = self._connections.get(server_id)
            if not connection:
                return False

            try:
                # Simple health check - try to list tools
                await connection.client.list_tools()
                connection.is_healthy = True
                connection.error_count = 0
                return True

            except Exception as e:
                connection.error_count += 1
                connection.is_healthy = False

                if connection.error_count >= connection.max_errors:
                    logger.warning(
                        "Connection marked as unhealthy",
                        server_id=server_id,
                        error=str(e),
                    )
                    await self._close_connection(connection)
                    del self._connections[server_id]

                return False

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        async with self._lock:
            return {
                "total_connections": len(self._connections),
                "healthy_connections": sum(
                    1 for c in self._connections.values() if c.is_healthy
                ),
                "connections": {
                    server_id: {
                        "created_at": connection.created_at,
                        "last_used": connection.last_used,
                        "is_healthy": connection.is_healthy,
                        "error_count": connection.error_count,
                    }
                    for server_id, connection in self._connections.items()
                },
            }

    async def _create_process_client(self, server_info: MCPServerInfo) -> MCPClient:
        """Create a process-based MCP client."""
        if not server_info.command:
            raise ValueError("Command is required for process server")

        # Start the process
        process = await asyncio.create_subprocess_exec(
            *server_info.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            env=server_info.env,
            cwd=server_info.working_directory,
        )

        return ProcessMCPClient(process)

    async def _create_http_client(self, server_info: MCPServerInfo) -> MCPClient:
        """Create an HTTP-based MCP client."""
        # Implementation for HTTP MCP clients
        raise NotImplementedError("HTTP MCP clients not yet implemented")

    async def _create_websocket_client(self, server_info: MCPServerInfo) -> MCPClient:
        """Create a WebSocket-based MCP client."""
        # Implementation for WebSocket MCP clients
        raise NotImplementedError("WebSocket MCP clients not yet implemented")

    async def _close_connection(self, connection: MCPConnection) -> None:
        """Close a connection."""
        try:
            # Close the client connection
            if hasattr(connection.client, "close"):
                await connection.client.close()
        except Exception as e:
            logger.warning(
                "Error closing connection", server_id=connection.server_id, error=str(e)
            )

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                async with self._lock:
                    current_time = asyncio.get_event_loop().time()
                    connections_to_remove = []

                    for server_id, connection in self._connections.items():
                        # Remove connections that haven't been used recently
                        if (
                            current_time - connection.last_used
                            > self.connection_timeout
                        ):
                            connections_to_remove.append(server_id)
                        # Remove unhealthy connections
                        elif not connection.is_healthy:
                            connections_to_remove.append(server_id)

                    for server_id in connections_to_remove:
                        connection = self._connections[server_id]
                        await self._close_connection(connection)
                        del self._connections[server_id]
                        logger.info("Connection cleaned up", server_id=server_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))


class MCPConnectionManager:
    """Manages MCP connections and provides high-level interface."""

    def __init__(self, max_connections: int = 10):
        self.pool = MCPConnectionPool(max_connections)
        self.server_manager: Any | None = None  # Will be set by dependency injection

    async def start(self) -> None:
        """Start the connection manager."""
        await self.pool.start()

    async def stop(self) -> None:
        """Stop the connection manager."""
        await self.pool.stop()

    async def get_client(self, server_id: str) -> MCPClient | None:
        """Get an MCP client for a server."""
        # Try to get existing connection
        connection = await self.pool.get_connection(server_id)
        if connection:
            return connection.client

        # Create new connection if server exists
        if self.server_manager:
            server_info = await self.server_manager.get_server(server_id)
            if server_info and server_info.status == MCPServerStatus.RUNNING:
                connection = await self.pool.create_connection(server_id, server_info)
                if connection:
                    return connection.client

        return None

    async def call_tool(
        self, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on a specific server."""
        client = await self.get_client(server_id)
        if not client:
            raise ValueError(f"No client available for server '{server_id}'")

        return await client.call_tool(tool_name, arguments)

    async def list_tools(self, server_id: str) -> list[Any]:
        """List tools from a specific server."""
        client = await self.get_client(server_id)
        if not client:
            return []

        return await client.list_tools()

    async def list_resources(self, server_id: str) -> list[Any]:
        """List resources from a specific server."""
        client = await self.get_client(server_id)
        if not client:
            return []

        return await client.list_resources()

    async def list_prompts(self, server_id: str) -> list[Any]:
        """List prompts from a specific server."""
        client = await self.get_client(server_id)
        if not client:
            return []

        return await client.list_prompts()

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all connections."""
        results = {}

        for server_id in self.pool._connections:
            results[server_id] = await self.pool.health_check_connection(server_id)

        return results

    async def get_stats(self) -> dict[str, Any]:
        """Get connection manager statistics."""
        return await self.pool.get_connection_stats()


class ProcessMCPClient(MCPClient):
    """MCP client for process-based communication."""

    def __init__(self, process: asyncio.subprocess.Process):
        super().__init__()
        self.process = process
        self._reader_task: asyncio.Task[None] | None = None
        self._start_reader()

    def _start_reader(self) -> None:
        """Start reading messages from the process."""
        self._reader_task = asyncio.create_task(self._read_messages())

    async def _send_message(self, message: MCPMessage) -> None:
        """Send message via process stdin."""
        if not self.process.stdin:
            raise RuntimeError("Process stdin not available")

        message_json = message.to_json() + "\n"
        self.process.stdin.write(message_json.encode())
        if hasattr(self.process.stdin, "drain"):
            await self.process.stdin.drain()

    async def _read_messages(self) -> None:
        """Read messages from process stdout."""
        if not self.process.stdout:
            return

        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                message_json = line.decode().strip()
                if message_json:
                    try:
                        message = MCPMessage.from_json(message_json)
                        await self.handle_message(message)
                    except Exception as e:
                        logger.warning(
                            "Failed to parse message",
                            message=message_json,
                            error=str(e),
                        )

        except Exception as e:
            logger.error("Error reading from process", error=str(e))
        finally:
            # Clean up reader task
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()

    async def close(self) -> None:
        """Close the client connection."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()


class HTTPMCPClient(MCPClient):
    """MCP client for HTTP-based communication."""

    def __init__(self, base_url: str, api_key: str | None = None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: Any | None = None  # httpx.AsyncClient

    async def _send_message(self, message: MCPMessage) -> None:
        """Send message via HTTP."""
        # Implementation for HTTP MCP communication
        raise NotImplementedError("HTTP MCP client not yet implemented")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._session:
            await self._session.aclose()


class WebSocketMCPClient(MCPClient):
    """MCP client for WebSocket-based communication."""

    def __init__(self, websocket_url: str):
        super().__init__()
        self.websocket_url = websocket_url
        self._websocket: Any | None = None  # websockets.WebSocketClientProtocol

    async def _send_message(self, message: MCPMessage) -> None:
        """Send message via WebSocket."""
        # Implementation for WebSocket MCP communication
        raise NotImplementedError("WebSocket MCP client not yet implemented")

    async def close(self) -> None:
        """Close the WebSocket client."""
        if self._websocket:
            await self._websocket.close()
