"""MCP Servers API endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_agent.api.dependencies import get_mcp_service, get_current_user
from ai_agent.core.mcp.service import MCPService, CreateMCPServerRequest
from ai_agent.domain.models import MCPServer

router = APIRouter(prefix="/mcp-servers", tags=["mcp-servers"])


@router.post("/", response_model=MCPServer, status_code=201)
async def create_mcp_server(
    request: CreateMCPServerRequest,
    mcp_service: Annotated[MCPService, Depends(get_mcp_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> MCPServer:
    """Create a new MCP server."""
    return await mcp_service.create_mcp_server(request)


@router.get("/", response_model=list[MCPServer])
async def list_mcp_servers(
    mcp_service: Annotated[MCPService, Depends(get_mcp_service)],
    current_user: Annotated[str, Depends(get_current_user)],
    enabled: Annotated[
        bool | None, Query(description="Filter by enabled status")
    ] = None,
    search: Annotated[
        str | None, Query(description="Search in server name/description")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[MCPServer]:
    """List MCP servers with filtering."""
    servers: list[MCPServer] = await mcp_service.list_mcp_servers(
        enabled=enabled,
        search=search,
        limit=limit,
        offset=offset,
    )
    return servers


@router.get("/{mcp_server_id}", response_model=MCPServer)
async def get_mcp_server(
    mcp_server_id: UUID,
    mcp_service: Annotated[MCPService, Depends(get_mcp_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> MCPServer:
    """Get specific MCP server by ID."""
    mcp_server = await mcp_service.get_mcp_server(mcp_server_id)
    if not mcp_server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return mcp_server


@router.put("/{mcp_server_id}", response_model=MCPServer)
async def update_mcp_server(
    mcp_server_id: UUID,
    request: CreateMCPServerRequest,
    mcp_service: Annotated[MCPService, Depends(get_mcp_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> MCPServer:
    """Update MCP server configuration."""
    return await mcp_service.update_mcp_server(mcp_server_id, request)


@router.delete("/{mcp_server_id}", status_code=204)
async def delete_mcp_server(
    mcp_server_id: UUID,
    mcp_service: Annotated[MCPService, Depends(get_mcp_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete an MCP server."""
    await mcp_service.delete_mcp_server(mcp_server_id)
