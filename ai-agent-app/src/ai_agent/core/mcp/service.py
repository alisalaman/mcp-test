"""MCP service for managing MCP servers."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ai_agent.domain.models import MCPServer
from ai_agent.infrastructure.database.base import Repository


class CreateMCPServerRequest(BaseModel):
    """Request to create a new MCP server."""

    name: str
    description: str | None = None
    endpoint: str
    authentication: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    health_check_url: str | None = None
    enabled: bool = True


class MCPService:
    """Service for managing MCP servers."""

    def __init__(self, repository: Repository, current_user: str):
        self.repository = repository
        self.current_user = current_user

    async def create_mcp_server(self, request: CreateMCPServerRequest) -> MCPServer:
        """Create a new MCP server."""
        mcp_server = MCPServer(
            name=request.name,
            description=request.description,
            endpoint=request.endpoint,
            authentication=request.authentication,
            capabilities=request.capabilities,
            health_check_url=request.health_check_url,
            enabled=request.enabled,
        )
        return await self.repository.create_mcp_server(mcp_server)

    async def get_mcp_server(self, mcp_server_id: UUID) -> MCPServer | None:
        """Get MCP server by ID."""
        return await self.repository.get_mcp_server(mcp_server_id)

    async def list_mcp_servers(
        self,
        enabled: bool | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MCPServer]:
        """List MCP servers with filtering."""
        # Get MCP servers from repository
        servers: list[MCPServer] = await self.repository.list_mcp_servers(
            enabled_only=enabled if enabled is not None else False,
            limit=limit,
            offset=offset,
        )

        # Apply additional filtering
        if enabled is not None:
            servers = [server for server in servers if server.enabled == enabled]

        if search:
            search_lower = search.lower()
            servers = [
                server
                for server in servers
                if search_lower in server.name.lower()
                or (server.description and search_lower in server.description.lower())
            ]

        return servers

    async def update_mcp_server(
        self, mcp_server_id: UUID, request: CreateMCPServerRequest
    ) -> MCPServer:
        """Update MCP server configuration."""
        mcp_server = await self.get_mcp_server(mcp_server_id)
        if not mcp_server:
            raise ValueError("MCP server not found")

        # Update fields
        mcp_server.name = request.name
        mcp_server.description = request.description
        mcp_server.endpoint = request.endpoint
        mcp_server.authentication = request.authentication
        mcp_server.capabilities = request.capabilities
        mcp_server.health_check_url = request.health_check_url
        mcp_server.enabled = request.enabled

        return await self.repository.update_mcp_server(mcp_server)

    async def delete_mcp_server(self, mcp_server_id: UUID) -> None:
        """Delete an MCP server."""
        await self.repository.delete_mcp_server(mcp_server_id)
