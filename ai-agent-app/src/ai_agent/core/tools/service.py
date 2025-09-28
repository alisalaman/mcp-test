"""Tool service for managing available tools."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ai_agent.domain.models import Tool
from ai_agent.infrastructure.database.base import Repository


class CreateToolRequest(BaseModel):
    """Request to create a new tool."""

    name: str
    description: str
    tool_schema: dict[str, Any]
    mcp_server_id: UUID | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolService:
    """Service for managing available tools."""

    def __init__(self, repository: Repository, current_user: str):
        self.repository = repository
        self.current_user = current_user

    async def create_tool(self, request: CreateToolRequest) -> Tool:
        """Create a new tool."""
        tool = Tool(
            name=request.name,
            description=request.description,
            tool_schema=request.tool_schema,
            mcp_server_id=request.mcp_server_id,
            enabled=request.enabled,
            metadata=request.metadata,
        )
        return await self.repository.create_tool(tool)

    async def get_tool(self, tool_id: UUID) -> Tool | None:
        """Get tool by ID."""
        return await self.repository.get_tool(tool_id)

    async def list_tools(
        self,
        enabled: bool | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Tool]:
        """List tools with filtering."""
        # Get tools from repository
        tools: list[Tool] = await self.repository.list_tools(
            enabled_only=enabled if enabled is not None else False,
            limit=limit,
            offset=offset,
        )

        # Apply additional filtering
        if enabled is not None:
            tools = [tool for tool in tools if tool.enabled == enabled]

        if search:
            search_lower = search.lower()
            tools = [
                tool
                for tool in tools
                if search_lower in tool.name.lower()
                or search_lower in tool.description.lower()
            ]

        return tools

    async def update_tool(self, tool_id: UUID, request: CreateToolRequest) -> Tool:
        """Update tool configuration."""
        tool = await self.get_tool(tool_id)
        if not tool:
            raise ValueError("Tool not found")

        # Update fields
        tool.name = request.name
        tool.description = request.description
        tool.tool_schema = request.tool_schema
        tool.mcp_server_id = request.mcp_server_id
        tool.enabled = request.enabled
        tool.metadata.update(request.metadata)

        return await self.repository.update_tool(tool)

    async def delete_tool(self, tool_id: UUID) -> None:
        """Delete a tool."""
        await self.repository.delete_tool(tool_id)
