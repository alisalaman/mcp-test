"""Dynamic tool discovery and registration system."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4
import structlog

from .protocol import MCPTool
from .server_manager import (
    MCPServerManager,
)

logger = structlog.get_logger()


class ToolStatus(str, Enum):
    """Tool status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ToolCategory(str, Enum):
    """Tool categories."""

    GENERAL = "general"
    FILE_OPERATIONS = "file_operations"
    WEB_SCRAPING = "web_scraping"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    AI_ML = "ai_ml"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolMetadata:
    """Tool metadata and configuration."""

    category: ToolCategory = ToolCategory.GENERAL
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str | None = None
    description: str | None = None
    documentation_url: str | None = None
    rate_limit: int | None = None  # requests per minute
    timeout: float | None = None  # seconds
    requires_auth: bool = False
    is_experimental: bool = False
    is_deprecated: bool = False
    deprecation_message: str | None = None
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredTool:
    """A registered tool with metadata."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    server_id: str = ""
    tool: MCPTool = field(
        default_factory=lambda: MCPTool(name="", description="", input_schema={})
    )
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    status: ToolStatus = ToolStatus.AVAILABLE
    last_used: float | None = None
    usage_count: int = 0
    error_count: int = 0
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    updated_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""

    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for managing MCP tools with discovery and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}  # tool_name -> RegisteredTool
        self._tools_by_server: dict[str, set[str]] = (
            {}
        )  # server_id -> set of tool_names
        self._tools_by_category: dict[ToolCategory, set[str]] = (
            {}
        )  # category -> set of tool_names
        self._tools_by_tag: dict[str, set[str]] = {}  # tag -> set of tool_names
        self._server_manager: MCPServerManager | None = None
        self._connection_manager: Any | None = None  # MCPConnectionManager
        self._lock = asyncio.Lock()
        self._discovery_task: asyncio.Task[None] | None = None
        self._discovery_interval = 30  # 30 seconds

    async def start(self) -> None:
        """Start the tool registry."""
        if self._discovery_task is None or self._discovery_task.done():
            self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info("Tool registry started")

    async def stop(self) -> None:
        """Stop the tool registry."""
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Tool registry stopped")

    def set_server_manager(self, server_manager: MCPServerManager) -> None:
        """Set the MCP server manager."""
        self._server_manager = server_manager

    def set_connection_manager(self, connection_manager: Any) -> None:
        """Set the MCP connection manager."""
        self._connection_manager = connection_manager

    async def discover_tools(
        self, server_id: str | None = None
    ) -> list[RegisteredTool]:
        """Discover tools from MCP servers."""
        if not self._server_manager:
            logger.warning("Server manager not set, cannot discover tools")
            return []

        discovered_tools = []

        try:
            # Get tools from server manager
            tool_infos = await self._server_manager.get_tools(server_id)

            async with self._lock:
                for tool_info in tool_infos:
                    # Check if tool already exists
                    existing_tool = self._tools.get(tool_info.tool.name)

                    if existing_tool and existing_tool.server_id == tool_info.server_id:
                        # Update existing tool
                        existing_tool.tool = tool_info.tool
                        existing_tool.status = ToolStatus.AVAILABLE
                        existing_tool.updated_at = asyncio.get_event_loop().time()
                        discovered_tools.append(existing_tool)
                    else:
                        # Register new tool
                        registered_tool = await self._register_tool(
                            tool_info.tool,
                            tool_info.server_id,
                            self._infer_tool_metadata(tool_info.tool),
                        )
                        discovered_tools.append(registered_tool)

            logger.info(
                "Tools discovered", count=len(discovered_tools), server_id=server_id
            )

        except Exception as e:
            logger.error("Tool discovery failed", error=str(e), server_id=server_id)

        return discovered_tools

    async def register_tool(
        self, tool: MCPTool, server_id: str, metadata: ToolMetadata | None = None
    ) -> RegisteredTool:
        """Register a tool manually."""
        async with self._lock:
            return await self._register_tool(
                tool, server_id, metadata or ToolMetadata()
            )

    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        async with self._lock:
            if tool_name not in self._tools:
                return False

            tool = self._tools[tool_name]

            # Remove from indexes
            if tool.server_id in self._tools_by_server:
                self._tools_by_server[tool.server_id].discard(tool_name)
                if not self._tools_by_server[tool.server_id]:
                    del self._tools_by_server[tool.server_id]

            if tool.metadata.category in self._tools_by_category:
                self._tools_by_category[tool.metadata.category].discard(tool_name)
                if not self._tools_by_category[tool.metadata.category]:
                    del self._tools_by_category[tool.metadata.category]

            for tag in tool.metadata.tags:
                if tag in self._tools_by_tag:
                    self._tools_by_tag[tag].discard(tool_name)
                    if not self._tools_by_tag[tag]:
                        del self._tools_by_tag[tag]

            # Remove the tool
            del self._tools[tool_name]

            logger.info("Tool unregistered", tool_name=tool_name)
            return True

    async def get_tool(self, tool_name: str) -> RegisteredTool | None:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    async def list_tools(
        self,
        category: ToolCategory | None = None,
        tag: str | None = None,
        server_id: str | None = None,
        status: ToolStatus | None = None,
        include_deprecated: bool = False,
    ) -> list[RegisteredTool]:
        """List tools with filtering."""
        tools = list(self._tools.values())

        # Apply filters
        if category:
            tools = [t for t in tools if t.metadata.category == category]

        if tag:
            tools = [t for t in tools if tag in t.metadata.tags]

        if server_id:
            tools = [t for t in tools if t.server_id == server_id]

        if status:
            tools = [t for t in tools if t.status == status]

        if not include_deprecated:
            tools = [t for t in tools if not t.metadata.is_deprecated]

        return tools

    async def search_tools(self, query: str, limit: int = 10) -> list[RegisteredTool]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        matches = []

        for tool in self._tools.values():
            score = 0

            # Name match (highest priority)
            if query_lower in tool.name.lower():
                score += 10

            # Description match
            if tool.tool.description and query_lower in tool.tool.description.lower():
                score += 5

            # Tag match
            for tag in tool.metadata.tags:
                if query_lower in tag.lower():
                    score += 3

            if score > 0:
                matches.append((score, tool))

        # Sort by score (descending) and return top results
        matches.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in matches[:limit]]

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], timeout: float | None = None
    ) -> ToolExecutionResult:
        """Execute a tool."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Get the tool
            tool = await self.get_tool(tool_name)
            if not tool:
                return ToolExecutionResult(
                    success=False, error=f"Tool '{tool_name}' not found"
                )

            # Check if tool is available
            if tool.status != ToolStatus.AVAILABLE:
                return ToolExecutionResult(
                    success=False,
                    error=f"Tool '{tool_name}' is not available (status: {tool.status})",
                )

            # Check if deprecated
            if tool.metadata.is_deprecated:
                logger.warning(
                    "Using deprecated tool",
                    tool_name=tool_name,
                    message=tool.metadata.deprecation_message,
                )

            # Execute the tool
            if not self._connection_manager:
                return ToolExecutionResult(
                    success=False, error="Connection manager not available"
                )

            # Apply timeout
            if timeout is None:
                timeout = tool.metadata.timeout or 30.0

            result = await asyncio.wait_for(
                self._connection_manager.call_tool(
                    tool.server_id, tool_name, arguments
                ),
                timeout=timeout,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            # Update tool usage statistics
            async with self._lock:
                tool.last_used = asyncio.get_event_loop().time()
                tool.usage_count += 1
                tool.error_count = 0  # Reset error count on success

            logger.info(
                "Tool executed successfully",
                tool_name=tool_name,
                execution_time=execution_time,
            )

            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    "tool_id": tool.id,
                    "server_id": tool.server_id,
                    "usage_count": tool.usage_count,
                },
            )

        except TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error("Tool execution timeout", tool_name=tool_name, timeout=timeout)

            # Update error count
            async with self._lock:
                if tool_name in self._tools:
                    self._tools[tool_name].error_count += 1

            return ToolExecutionResult(
                success=False,
                error=f"Tool execution timeout after {timeout}s",
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))

            # Update error count
            async with self._lock:
                if tool_name in self._tools:
                    self._tools[tool_name].error_count += 1
                    # Mark as error if too many failures
                    if self._tools[tool_name].error_count >= 5:
                        self._tools[tool_name].status = ToolStatus.ERROR

            return ToolExecutionResult(
                success=False, error=str(e), execution_time=execution_time
            )

    async def get_tool_stats(self) -> dict[str, Any]:
        """Get tool registry statistics."""
        async with self._lock:
            total_tools = len(self._tools)
            available_tools = sum(
                1 for t in self._tools.values() if t.status == ToolStatus.AVAILABLE
            )
            error_tools = sum(
                1 for t in self._tools.values() if t.status == ToolStatus.ERROR
            )
            deprecated_tools = sum(
                1 for t in self._tools.values() if t.metadata.is_deprecated
            )

            # Category breakdown
            category_counts: dict[str, int] = {}
            for tool in self._tools.values():
                category = tool.metadata.category.value
                category_counts[category] = category_counts.get(category, 0) + 1

            # Server breakdown
            server_counts: dict[str, int] = {}
            for tool in self._tools.values():
                server_id = tool.server_id
                server_counts[server_id] = server_counts.get(server_id, 0) + 1

            return {
                "total_tools": total_tools,
                "available_tools": available_tools,
                "error_tools": error_tools,
                "deprecated_tools": deprecated_tools,
                "category_breakdown": category_counts,
                "server_breakdown": server_counts,
                "most_used_tools": sorted(
                    [(t.name, t.usage_count) for t in self._tools.values()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
            }

    async def _register_tool(
        self, tool: MCPTool, server_id: str, metadata: ToolMetadata
    ) -> RegisteredTool:
        """Internal method to register a tool."""
        # Create registered tool
        registered_tool = RegisteredTool(
            name=tool.name, server_id=server_id, tool=tool, metadata=metadata
        )

        # Store the tool
        self._tools[tool.name] = registered_tool

        # Update indexes
        if server_id not in self._tools_by_server:
            self._tools_by_server[server_id] = set()
        self._tools_by_server[server_id].add(tool.name)

        if metadata.category not in self._tools_by_category:
            self._tools_by_category[metadata.category] = set()
        self._tools_by_category[metadata.category].add(tool.name)

        for tag in metadata.tags:
            if tag not in self._tools_by_tag:
                self._tools_by_tag[tag] = set()
            self._tools_by_tag[tag].add(tool.name)

        logger.info("Tool registered", tool_name=tool.name, server_id=server_id)
        return registered_tool

    def _infer_tool_metadata(self, tool: MCPTool) -> ToolMetadata:
        """Infer tool metadata from tool definition."""
        metadata = ToolMetadata()

        # Infer category from tool name and description
        name_lower = tool.name.lower()
        desc_lower = tool.description.lower() if tool.description else ""

        if any(
            keyword in name_lower
            for keyword in ["file", "read", "write", "delete", "copy", "move"]
        ):
            metadata.category = ToolCategory.FILE_OPERATIONS
        elif any(
            keyword in name_lower
            for keyword in ["web", "scrape", "fetch", "http", "url"]
        ):
            metadata.category = ToolCategory.WEB_SCRAPING
        elif any(
            keyword in name_lower
            for keyword in ["data", "analyze", "process", "transform"]
        ):
            metadata.category = ToolCategory.DATA_ANALYSIS
        elif any(
            keyword in name_lower for keyword in ["email", "message", "send", "notify"]
        ):
            metadata.category = ToolCategory.COMMUNICATION
        elif any(
            keyword in name_lower for keyword in ["code", "git", "build", "deploy"]
        ):
            metadata.category = ToolCategory.DEVELOPMENT
        elif any(keyword in name_lower for keyword in ["ai", "ml", "model", "predict"]):
            metadata.category = ToolCategory.AI_ML
        elif any(
            keyword in name_lower for keyword in ["system", "process", "kill", "status"]
        ):
            metadata.category = ToolCategory.SYSTEM
        else:
            metadata.category = ToolCategory.GENERAL

        # Extract tags from description
        if tool.description:
            # Simple keyword extraction
            keywords = ["api", "rest", "json", "async", "sync", "batch", "stream"]
            for keyword in keywords:
                if keyword in desc_lower:
                    metadata.tags.append(keyword)

        return metadata

    async def _discovery_loop(self) -> None:
        """Background tool discovery loop."""
        while True:
            try:
                await asyncio.sleep(self._discovery_interval)
                await self.discover_tools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Tool discovery loop error", error=str(e))
