"""Tools API endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_agent.api.dependencies import get_tool_service, get_current_user
from ai_agent.core.tools.service import ToolService, CreateToolRequest
from ai_agent.domain.models import Tool

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post("/", response_model=Tool, status_code=201)
async def create_tool(
    request: CreateToolRequest,
    tool_service: Annotated[ToolService, Depends(get_tool_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Tool:
    """Create a new tool."""
    return await tool_service.create_tool(request)


@router.get("/", response_model=list[Tool])
async def list_tools(
    tool_service: Annotated[ToolService, Depends(get_tool_service)],
    current_user: Annotated[str, Depends(get_current_user)],
    enabled: Annotated[
        bool | None, Query(description="Filter by enabled status")
    ] = None,
    search: Annotated[
        str | None, Query(description="Search in tool name/description")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Tool]:
    """List tools with filtering."""
    tools: list[Tool] = await tool_service.list_tools(
        enabled=enabled,
        search=search,
        limit=limit,
        offset=offset,
    )
    return tools


@router.get("/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: UUID,
    tool_service: Annotated[ToolService, Depends(get_tool_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Tool:
    """Get specific tool by ID."""
    tool = await tool_service.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.put("/{tool_id}", response_model=Tool)
async def update_tool(
    tool_id: UUID,
    request: CreateToolRequest,
    tool_service: Annotated[ToolService, Depends(get_tool_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Tool:
    """Update tool configuration."""
    return await tool_service.update_tool(tool_id, request)


@router.delete("/{tool_id}", status_code=204)
async def delete_tool(
    tool_id: UUID,
    tool_service: Annotated[ToolService, Depends(get_tool_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete a tool."""
    await tool_service.delete_tool(tool_id)
