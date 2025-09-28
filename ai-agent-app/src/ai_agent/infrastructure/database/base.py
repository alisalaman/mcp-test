"""
Repository interface and base classes for data persistence.

This module defines the abstract repository interface that all storage
implementations must follow, ensuring consistent data access patterns.
"""

from abc import ABC, abstractmethod
from typing import Protocol
from uuid import UUID

from ai_agent.domain.models import Agent, MCPServer, Message, Session, Tool


class Repository(Protocol):
    """Repository interface for data persistence operations."""

    # Connection lifecycle management
    async def connect(self) -> None:
        """Initialize connection to the storage backend."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the storage backend."""
        ...

    async def health_check(self) -> bool:
        """Check if the storage backend is healthy."""
        ...

    # Session operations
    async def create_session(self, session: Session) -> Session:
        """Create a new session."""
        ...

    async def get_session(self, session_id: UUID) -> Session | None:
        """Get a session by ID."""
        ...

    async def list_sessions(
        self, user_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Session]:
        """List sessions with optional filtering."""
        ...

    async def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        ...

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages."""
        ...

    # Message operations
    async def create_message(self, message: Message) -> Message:
        """Create a new message."""
        ...

    async def get_message(self, message_id: UUID) -> Message | None:
        """Get a message by ID."""
        ...

    async def get_messages_by_session(
        self, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """Get messages for a session."""
        ...

    async def update_message(self, message: Message) -> Message:
        """Update an existing message."""
        ...

    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message."""
        ...

    # Agent operations
    async def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent."""
        ...

    async def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get an agent by ID."""
        ...

    async def list_agents(self, limit: int = 100, offset: int = 0) -> list[Agent]:
        """List agents."""
        ...

    async def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent."""
        ...

    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent."""
        ...

    # Tool operations
    async def create_tool(self, tool: Tool) -> Tool:
        """Create a new tool."""
        ...

    async def get_tool(self, tool_id: UUID) -> Tool | None:
        """Get a tool by ID."""
        ...

    async def list_tools(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Tool]:
        """List tools."""
        ...

    async def update_tool(self, tool: Tool) -> Tool:
        """Update an existing tool."""
        ...

    async def delete_tool(self, tool_id: UUID) -> bool:
        """Delete a tool."""
        ...

    # MCP Server operations
    async def create_mcp_server(self, server: MCPServer) -> MCPServer:
        """Create a new MCP server."""
        ...

    async def get_mcp_server(self, server_id: UUID) -> MCPServer | None:
        """Get an MCP server by ID."""
        ...

    async def list_mcp_servers(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[MCPServer]:
        """List MCP servers."""
        ...

    async def update_mcp_server(self, server: MCPServer) -> MCPServer:
        """Update an existing MCP server."""
        ...

    async def delete_mcp_server(self, server_id: UUID) -> bool:
        """Delete an MCP server."""
        ...


class BaseRepository(ABC):
    """Abstract base repository with common functionality."""

    def __init__(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if repository is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the storage backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the storage backend."""
        pass

    async def health_check(self) -> bool:
        """Basic health check implementation."""
        return self.is_connected

    def _ensure_connected(self) -> None:
        """Ensure repository is connected before operations."""
        if not self.is_connected:
            raise RuntimeError("Repository not connected. Call connect() first.")


class RepositoryError(Exception):
    """Base exception for repository operations."""

    pass


class ConnectionError(RepositoryError):
    """Connection-related repository error."""

    pass


class NotFoundError(RepositoryError):
    """Entity not found error."""

    pass


class DuplicateError(RepositoryError):
    """Duplicate entity error."""

    pass


class ValidationError(RepositoryError):
    """Validation error during repository operation."""

    pass
