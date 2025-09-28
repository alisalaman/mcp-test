"""
In-memory repository implementation for development and testing.

This provides fast, non-persistent storage using Python dictionaries with
thread-safe operations and efficient indexing for session-message relationships.
"""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from uuid import UUID

from ai_agent.domain.models import Agent, MCPServer, Message, Session, Tool
from ai_agent.infrastructure.database.base import (
    BaseRepository,
    DuplicateError,
    NotFoundError,
)


class InMemoryRepository(BaseRepository):
    """In-memory storage for development and testing."""

    def __init__(self) -> None:
        super().__init__()

        # Storage dictionaries
        self._sessions: dict[UUID, Session] = {}
        self._messages: dict[UUID, Message] = {}
        self._agents: dict[UUID, Agent] = {}
        self._tools: dict[UUID, Tool] = {}
        self._mcp_servers: dict[UUID, MCPServer] = {}

        # Indexes for efficient queries
        self._messages_by_session: dict[UUID, list[UUID]] = defaultdict(list)
        self._tools_by_agent: dict[UUID, list[UUID]] = defaultdict(list)
        self._sessions_by_user: dict[str, list[UUID]] = defaultdict(list)
        self._messages_by_user: dict[str, list[UUID]] = defaultdict(list)

        # Thread safety
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize in-memory storage."""
        self._connected = True

    async def disconnect(self) -> None:
        """Clear in-memory storage."""
        async with self._lock:
            self._sessions.clear()
            self._messages.clear()
            self._agents.clear()
            self._tools.clear()
            self._mcp_servers.clear()
            self._messages_by_session.clear()
            self._tools_by_agent.clear()
            self._sessions_by_user.clear()
            self._messages_by_user.clear()
        self._connected = False

    async def health_check(self) -> bool:
        """Check health of in-memory storage."""
        return bool(self.is_connected)

    # Session operations
    async def create_session(self, session: Session) -> Session:
        """Create a new session."""
        self._ensure_connected()
        async with self._lock:
            if session.id in self._sessions:
                raise DuplicateError(f"Session with ID {session.id} already exists")

            self._sessions[session.id] = session
            self._sessions_by_user[session.user_id].append(session.id)
            return session

    async def get_session(self, session_id: UUID) -> Session | None:
        """Get session by ID."""
        self._ensure_connected()
        return self._sessions.get(session_id)

    async def list_sessions(
        self, user_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Session]:
        """List sessions with optional filtering."""
        self._ensure_connected()

        if user_id:
            # Use user index for O(1) lookup
            session_ids = self._sessions_by_user.get(user_id, [])
            sessions = [
                self._sessions[sid] for sid in session_ids if sid in self._sessions
            ]
        else:
            sessions = list(self._sessions.values())

        # Sort by last activity
        sessions.sort(key=lambda s: s.last_activity, reverse=True)

        return sessions[offset : offset + limit]

    async def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        self._ensure_connected()
        async with self._lock:
            if session.id not in self._sessions:
                raise NotFoundError(f"Session with ID {session.id} not found")

            # Update timestamp
            session.updated_at = datetime.now(UTC)
            self._sessions[session.id] = session
            return session

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages."""
        self._ensure_connected()
        async with self._lock:
            if session_id not in self._sessions:
                return False

            # Delete all messages in this session
            message_ids = self._messages_by_session.get(session_id, [])
            for message_id in message_ids:
                self._messages.pop(message_id, None)

            # Clean up indexes
            self._messages_by_session.pop(session_id, None)

            # Remove from user index
            session = self._sessions[session_id]
            if session.user_id in self._sessions_by_user:
                user_sessions = self._sessions_by_user[session.user_id]
                if session_id in user_sessions:
                    user_sessions.remove(session_id)

            # Delete session
            self._sessions.pop(session_id, None)
            return True

    # Message operations
    async def create_message(self, message: Message) -> Message:
        """Create a new message."""
        self._ensure_connected()
        async with self._lock:
            if message.id in self._messages:
                raise DuplicateError(f"Message with ID {message.id} already exists")

            # Verify session exists
            if message.session_id not in self._sessions:
                raise NotFoundError(f"Session with ID {message.session_id} not found")

            # Store message
            self._messages[message.id] = message
            self._messages_by_session[message.session_id].append(message.id)

            # Add to user index
            session = self._sessions[message.session_id]
            self._messages_by_user[session.user_id].append(message.id)

            # Update session metadata
            session = self._sessions[message.session_id]
            session.message_count += 1
            session.last_activity = datetime.now(UTC)
            session.updated_at = datetime.now(UTC)

            return message

    async def get_message(self, message_id: UUID) -> Message | None:
        """Get message by ID."""
        self._ensure_connected()
        return self._messages.get(message_id)

    async def get_messages_by_session(
        self, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """Get messages for a session."""
        self._ensure_connected()
        message_ids = self._messages_by_session.get(session_id, [])
        messages = [self._messages[mid] for mid in message_ids if mid in self._messages]

        # Sort by creation time
        messages.sort(key=lambda m: m.created_at)

        return messages[offset : offset + limit]

    async def get_messages_by_user(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """Get messages for a user using user index for O(1) lookup."""
        self._ensure_connected()
        message_ids = self._messages_by_user.get(user_id, [])
        messages = [self._messages[mid] for mid in message_ids if mid in self._messages]

        # Sort by creation time
        messages.sort(key=lambda m: m.created_at)

        return messages[offset : offset + limit]

    async def update_message(self, message: Message) -> Message:
        """Update an existing message."""
        self._ensure_connected()
        async with self._lock:
            if message.id not in self._messages:
                raise NotFoundError(f"Message with ID {message.id} not found")

            # Update timestamp
            message.updated_at = datetime.now(UTC)
            self._messages[message.id] = message
            return message

    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message."""
        self._ensure_connected()
        async with self._lock:
            if message_id not in self._messages:
                return False

            message = self._messages[message_id]

            # Remove from session index
            session_messages = self._messages_by_session.get(message.session_id, [])
            if message_id in session_messages:
                session_messages.remove(message_id)

            # Remove from user index
            if message.session_id in self._sessions:
                session = self._sessions[message.session_id]
                user_messages = self._messages_by_user.get(session.user_id, [])
                if message_id in user_messages:
                    user_messages.remove(message_id)

            # Update session message count
            if message.session_id in self._sessions:
                session = self._sessions[message.session_id]
                session.message_count = max(0, session.message_count - 1)
                session.updated_at = datetime.now(UTC)

            # Delete message
            self._messages.pop(message_id, None)
            return True

    # Agent operations
    async def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent."""
        self._ensure_connected()
        async with self._lock:
            if agent.id in self._agents:
                raise DuplicateError(f"Agent with ID {agent.id} already exists")

            # Check for duplicate names
            for existing_agent in self._agents.values():
                if existing_agent.name == agent.name:
                    raise DuplicateError(
                        f"Agent with name '{agent.name}' already exists"
                    )

            self._agents[agent.id] = agent
            return agent

    async def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        self._ensure_connected()
        return self._agents.get(agent_id)

    async def list_agents(self, limit: int = 100, offset: int = 0) -> list[Agent]:
        """List agents."""
        self._ensure_connected()
        agents = list(self._agents.values())

        # Sort by name
        agents.sort(key=lambda a: a.name)

        return agents[offset : offset + limit]

    async def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent."""
        self._ensure_connected()
        async with self._lock:
            if agent.id not in self._agents:
                raise NotFoundError(f"Agent with ID {agent.id} not found")

            # Check for duplicate names (excluding current agent)
            for existing_id, existing_agent in self._agents.items():
                if existing_id != agent.id and existing_agent.name == agent.name:
                    raise DuplicateError(
                        f"Agent with name '{agent.name}' already exists"
                    )

            # Update timestamp
            agent.updated_at = datetime.now(UTC)
            self._agents[agent.id] = agent
            return agent

    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent."""
        self._ensure_connected()
        async with self._lock:
            if agent_id not in self._agents:
                return False

            # Clean up tool associations
            self._tools_by_agent.pop(agent_id, None)

            # Delete agent
            self._agents.pop(agent_id, None)
            return True

    # Tool operations
    async def create_tool(self, tool: Tool) -> Tool:
        """Create a new tool."""
        self._ensure_connected()
        async with self._lock:
            if tool.id in self._tools:
                raise DuplicateError(f"Tool with ID {tool.id} already exists")

            # Check for duplicate names
            for existing_tool in self._tools.values():
                if existing_tool.name == tool.name:
                    raise DuplicateError(f"Tool with name '{tool.name}' already exists")

            self._tools[tool.id] = tool
            return tool

    async def get_tool(self, tool_id: UUID) -> Tool | None:
        """Get tool by ID."""
        self._ensure_connected()
        return self._tools.get(tool_id)

    async def list_tools(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Tool]:
        """List tools."""
        self._ensure_connected()
        tools = list(self._tools.values())

        if enabled_only:
            tools = [t for t in tools if t.enabled]

        # Sort by name
        tools.sort(key=lambda t: t.name)

        return tools[offset : offset + limit]

    async def update_tool(self, tool: Tool) -> Tool:
        """Update an existing tool."""
        self._ensure_connected()
        async with self._lock:
            if tool.id not in self._tools:
                raise NotFoundError(f"Tool with ID {tool.id} not found")

            # Check for duplicate names (excluding current tool)
            for existing_id, existing_tool in self._tools.items():
                if existing_id != tool.id and existing_tool.name == tool.name:
                    raise DuplicateError(f"Tool with name '{tool.name}' already exists")

            # Update timestamp
            tool.updated_at = datetime.now(UTC)
            self._tools[tool.id] = tool
            return tool

    async def delete_tool(self, tool_id: UUID) -> bool:
        """Delete a tool."""
        self._ensure_connected()
        async with self._lock:
            if tool_id not in self._tools:
                return False

            # Delete tool
            self._tools.pop(tool_id, None)
            return True

    # MCP Server operations
    async def create_mcp_server(self, server: MCPServer) -> MCPServer:
        """Create a new MCP server."""
        self._ensure_connected()
        async with self._lock:
            if server.id in self._mcp_servers:
                raise DuplicateError(f"MCP Server with ID {server.id} already exists")

            # Check for duplicate names
            for existing_server in self._mcp_servers.values():
                if existing_server.name == server.name:
                    raise DuplicateError(
                        f"MCP Server with name '{server.name}' already exists"
                    )

            self._mcp_servers[server.id] = server
            return server

    async def get_mcp_server(self, server_id: UUID) -> MCPServer | None:
        """Get MCP server by ID."""
        self._ensure_connected()
        return self._mcp_servers.get(server_id)

    async def list_mcp_servers(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[MCPServer]:
        """List MCP servers."""
        self._ensure_connected()
        servers = list(self._mcp_servers.values())

        if enabled_only:
            servers = [s for s in servers if s.enabled]

        # Sort by name
        servers.sort(key=lambda s: s.name)

        return servers[offset : offset + limit]

    async def update_mcp_server(self, server: MCPServer) -> MCPServer:
        """Update an existing MCP server."""
        self._ensure_connected()
        async with self._lock:
            if server.id not in self._mcp_servers:
                raise NotFoundError(f"MCP Server with ID {server.id} not found")

            # Check for duplicate names (excluding current server)
            for existing_id, existing_server in self._mcp_servers.items():
                if existing_id != server.id and existing_server.name == server.name:
                    raise DuplicateError(
                        f"MCP Server with name '{server.name}' already exists"
                    )

            # Update timestamp
            server.updated_at = datetime.now(UTC)
            self._mcp_servers[server.id] = server
            return server

    async def delete_mcp_server(self, server_id: UUID) -> bool:
        """Delete an MCP server."""
        self._ensure_connected()
        async with self._lock:
            if server_id not in self._mcp_servers:
                return False

            # Clean up associated tools
            tools_to_remove = []
            for tool_id, tool in self._tools.items():
                if tool.mcp_server_id == server_id:
                    tools_to_remove.append(tool_id)

            for tool_id in tools_to_remove:
                self._tools.pop(tool_id, None)

            # Delete server
            self._mcp_servers.pop(server_id, None)
            return True
