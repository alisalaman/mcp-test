"""
Redis repository implementation for session state and caching.

This provides high-performance session state management with TTL support,
efficient key-value operations, and connection pooling.
"""

import json
from datetime import UTC, datetime
from uuid import UUID

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

from ai_agent.config.settings import RedisSettings
from ai_agent.domain.models import Agent, MCPServer, Message, Session, Tool
from ai_agent.infrastructure.database.base import (
    BaseRepository,
    ConnectionError,
    DuplicateError,
    NotFoundError,
)


class RedisRepository(BaseRepository):
    """Redis-based storage for session state and caching."""

    def __init__(self, settings: RedisSettings):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not available. Install with: pip install redis[hiredis]"
            )

        super().__init__()
        self.settings = settings
        self._redis: redis.Redis | None = None

        # Key prefixes for different entity types
        self.SESSION_PREFIX = "session:"
        self.MESSAGE_PREFIX = "message:"
        self.AGENT_PREFIX = "agent:"
        self.TOOL_PREFIX = "tool:"
        self.MCP_SERVER_PREFIX = "mcp_server:"

        # Index prefixes for relationships
        self.SESSION_MESSAGES_PREFIX = "session_messages:"
        self.USER_SESSIONS_PREFIX = "user_sessions:"
        self.AGENT_TOOLS_PREFIX = "agent_tools:"

        # Default TTL (1 hour)
        self.DEFAULT_TTL = 3600

    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = redis.from_url(
                self.settings.url,
                max_connections=self.settings.max_connections,
                retry_on_timeout=self.settings.retry_on_timeout,
                health_check_interval=self.settings.health_check_interval,
            )

            # Test connection
            await self._ensure_redis_connection().ping()
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._ensure_redis_connection().close()
        self._connected = False

    def _ensure_redis_connection(self) -> redis.Redis:
        """Ensure Redis connection is available and return it."""
        if not self._redis:
            raise ConnectionError("Redis connection not available")
        return self._redis

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.is_connected or not self._redis:
            return False

        try:
            await self._ensure_redis_connection().ping()
            return True
        except Exception:
            return False

    # Session operations
    async def create_session(self, session: Session) -> Session:
        """Create a new session in Redis."""
        self._ensure_connected()
        session_key = f"{self.SESSION_PREFIX}{session.id}"
        user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{session.user_id}"

        # Check if session already exists
        redis_client = self._ensure_redis_connection()
        if await redis_client.exists(session_key):
            raise DuplicateError(f"Session with ID {session.id} already exists")

        # Store session data
        session_data = session.model_dump_json()
        await redis_client.setex(session_key, self.DEFAULT_TTL, session_data)

        # Add to user's session list
        if session.user_id:
            await redis_client.zadd(
                user_sessions_key, {str(session.id): session.created_at.timestamp()}
            )
            await redis_client.expire(user_sessions_key, self.DEFAULT_TTL)

        return session

    async def get_session(self, session_id: UUID) -> Session | None:
        """Get session from Redis."""
        self._ensure_connected()
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        session_data = await self._ensure_redis_connection().get(session_key)

        if session_data:
            data = json.loads(session_data)
            return Session.model_validate(data)

        return None

    async def list_sessions(
        self, user_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Session]:
        """List sessions with optional filtering."""
        self._ensure_connected()
        sessions = []

        if user_id:
            # Get sessions from user index
            user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            session_ids = await self._ensure_redis_connection().zrevrange(
                user_sessions_key, offset, offset + limit - 1
            )

            for session_id in session_ids:
                session = await self.get_session(UUID(session_id))
                if session:
                    sessions.append(session)
        else:
            # Get all sessions (expensive operation)
            keys = await self._ensure_redis_connection().keys(f"{self.SESSION_PREFIX}*")
            for key in keys[offset : offset + limit]:
                session_id = key.decode().replace(self.SESSION_PREFIX, "")
                session = await self.get_session(UUID(session_id))
                if session:
                    sessions.append(session)

            # Sort by last activity
            sessions.sort(key=lambda s: s.last_activity, reverse=True)

        return sessions

    async def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        self._ensure_connected()
        session_key = f"{self.SESSION_PREFIX}{session.id}"

        # Check if session exists
        if not await self._ensure_redis_connection().exists(session_key):
            raise NotFoundError(f"Session with ID {session.id} not found")

        # Update timestamp
        session.updated_at = datetime.now(UTC)

        # Store updated session data
        session_data = session.model_dump_json()
        await self._ensure_redis_connection().setex(
            session_key, self.DEFAULT_TTL, session_data
        )

        return session

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages."""
        self._ensure_connected()
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        session_messages_key = f"{self.SESSION_MESSAGES_PREFIX}{session_id}"

        # Get session to find user_id for cleanup
        session = await self.get_session(session_id)
        if not session:
            return False

        # Delete all messages in this session
        message_ids = await self._ensure_redis_connection().zrange(
            session_messages_key, 0, -1
        )
        for message_id in message_ids:
            await self._ensure_redis_connection().delete(
                f"{self.MESSAGE_PREFIX}{message_id}"
            )

        # Clean up indexes
        await self._ensure_redis_connection().delete(session_messages_key)

        if session.user_id:
            user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{session.user_id}"
            await self._ensure_redis_connection().zrem(
                user_sessions_key, str(session_id)
            )

        # Delete session
        deleted_count = await self._ensure_redis_connection().delete(session_key)
        return bool(deleted_count > 0)

    # Message operations
    async def create_message(self, message: Message) -> Message:
        """Create a new message in Redis."""
        self._ensure_connected()
        message_key = f"{self.MESSAGE_PREFIX}{message.id}"
        session_messages_key = f"{self.SESSION_MESSAGES_PREFIX}{message.session_id}"

        # Check if message already exists
        if await self._ensure_redis_connection().exists(message_key):
            raise DuplicateError(f"Message with ID {message.id} already exists")

        # Verify session exists
        session = await self.get_session(message.session_id)
        if not session:
            raise NotFoundError(f"Session with ID {message.session_id} not found")

        # Store message data
        message_data = message.model_dump_json()
        await self._ensure_redis_connection().setex(
            message_key, self.DEFAULT_TTL, message_data
        )

        # Add to session's message list (sorted by timestamp)
        await self._ensure_redis_connection().zadd(
            session_messages_key, {str(message.id): message.created_at.timestamp()}
        )
        await self._ensure_redis_connection().expire(
            session_messages_key, self.DEFAULT_TTL
        )

        # Update session metadata
        await self._update_session_activity(message.session_id)

        return message

    async def get_message(self, message_id: UUID) -> Message | None:
        """Get message from Redis."""
        self._ensure_connected()
        message_key = f"{self.MESSAGE_PREFIX}{message_id}"
        message_data = await self._ensure_redis_connection().get(message_key)

        if message_data:
            data = json.loads(message_data)
            return Message.model_validate(data)

        return None

    async def get_messages_by_session(
        self, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """Get messages for a session."""
        self._ensure_connected()
        session_messages_key = f"{self.SESSION_MESSAGES_PREFIX}{session_id}"

        # Get message IDs sorted by timestamp
        message_ids = await self._ensure_redis_connection().zrange(
            session_messages_key, offset, offset + limit - 1
        )

        messages = []
        for message_id in message_ids:
            message = await self.get_message(UUID(message_id))
            if message:
                messages.append(message)

        return messages

    async def update_message(self, message: Message) -> Message:
        """Update an existing message."""
        self._ensure_connected()
        message_key = f"{self.MESSAGE_PREFIX}{message.id}"

        # Check if message exists
        if not await self._ensure_redis_connection().exists(message_key):
            raise NotFoundError(f"Message with ID {message.id} not found")

        # Update timestamp
        message.updated_at = datetime.now(UTC)

        # Store updated message data
        message_data = message.model_dump_json()
        await self._ensure_redis_connection().setex(
            message_key, self.DEFAULT_TTL, message_data
        )

        return message

    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message."""
        self._ensure_connected()
        message_key = f"{self.MESSAGE_PREFIX}{message_id}"

        # Get message to find session for cleanup
        message = await self.get_message(message_id)
        if not message:
            return False

        # Remove from session index
        session_messages_key = f"{self.SESSION_MESSAGES_PREFIX}{message.session_id}"
        await self._ensure_redis_connection().zrem(
            session_messages_key, str(message_id)
        )

        # Update session message count
        session = await self.get_session(message.session_id)
        if session:
            session.message_count = max(0, session.message_count - 1)
            session.updated_at = datetime.now(UTC)
            await self.update_session(session)

        # Delete message
        deleted_count = await self._ensure_redis_connection().delete(message_key)
        return bool(deleted_count > 0)

    async def _update_session_activity(self, session_id: UUID) -> None:
        """Update session's last activity and message count."""
        session_key = f"{self.SESSION_PREFIX}{session_id}"

        # Get current session
        session_data = await self._ensure_redis_connection().get(session_key)
        if session_data:
            session_dict = json.loads(session_data)
            session_dict["last_activity"] = datetime.now(UTC).isoformat()
            session_dict["message_count"] = session_dict.get("message_count", 0) + 1
            session_dict["updated_at"] = datetime.now(UTC).isoformat()

            # Update session
            await self._ensure_redis_connection().setex(
                session_key, self.DEFAULT_TTL, json.dumps(session_dict)
            )

    # Agent operations (basic implementation for Redis)
    async def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent."""
        self._ensure_connected()
        agent_key = f"{self.AGENT_PREFIX}{agent.id}"

        if await self._ensure_redis_connection().exists(agent_key):
            raise DuplicateError(f"Agent with ID {agent.id} already exists")

        agent_data = agent.model_dump_json()
        await self._ensure_redis_connection().setex(
            agent_key, self.DEFAULT_TTL, agent_data
        )
        return agent

    async def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        self._ensure_connected()
        agent_key = f"{self.AGENT_PREFIX}{agent_id}"
        agent_data = await self._ensure_redis_connection().get(agent_key)

        if agent_data:
            data = json.loads(agent_data)
            return Agent.model_validate(data)

        return None

    async def list_agents(self, limit: int = 100, offset: int = 0) -> list[Agent]:
        """List agents."""
        self._ensure_connected()
        keys = await self._ensure_redis_connection().keys(f"{self.AGENT_PREFIX}*")
        agents = []

        for key in keys[offset : offset + limit]:
            agent_id = key.decode().replace(self.AGENT_PREFIX, "")
            agent = await self.get_agent(UUID(agent_id))
            if agent:
                agents.append(agent)

        return agents

    async def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent."""
        self._ensure_connected()
        agent_key = f"{self.AGENT_PREFIX}{agent.id}"

        if not await self._ensure_redis_connection().exists(agent_key):
            raise NotFoundError(f"Agent with ID {agent.id} not found")

        agent.updated_at = datetime.now(UTC)
        agent_data = agent.model_dump_json()
        await self._ensure_redis_connection().setex(
            agent_key, self.DEFAULT_TTL, agent_data
        )
        return agent

    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent."""
        self._ensure_connected()
        agent_key = f"{self.AGENT_PREFIX}{agent_id}"
        deleted_count = await self._ensure_redis_connection().delete(agent_key)
        return bool(deleted_count > 0)

    # Tool operations (basic implementation for Redis)
    async def create_tool(self, tool: Tool) -> Tool:
        """Create a new tool."""
        self._ensure_connected()
        tool_key = f"{self.TOOL_PREFIX}{tool.id}"

        if await self._ensure_redis_connection().exists(tool_key):
            raise DuplicateError(f"Tool with ID {tool.id} already exists")

        tool_data = tool.model_dump_json()
        await self._ensure_redis_connection().setex(
            tool_key, self.DEFAULT_TTL, tool_data
        )
        return tool

    async def get_tool(self, tool_id: UUID) -> Tool | None:
        """Get tool by ID."""
        self._ensure_connected()
        tool_key = f"{self.TOOL_PREFIX}{tool_id}"
        tool_data = await self._ensure_redis_connection().get(tool_key)

        if tool_data:
            data = json.loads(tool_data)
            return Tool.model_validate(data)

        return None

    async def list_tools(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Tool]:
        """List tools."""
        self._ensure_connected()
        keys = await self._ensure_redis_connection().keys(f"{self.TOOL_PREFIX}*")
        tools = []

        for key in keys[offset : offset + limit]:
            tool_id = key.decode().replace(self.TOOL_PREFIX, "")
            tool = await self.get_tool(UUID(tool_id))
            if tool and (not enabled_only or tool.enabled):
                tools.append(tool)

        return tools

    async def update_tool(self, tool: Tool) -> Tool:
        """Update an existing tool."""
        self._ensure_connected()
        tool_key = f"{self.TOOL_PREFIX}{tool.id}"

        if not await self._ensure_redis_connection().exists(tool_key):
            raise NotFoundError(f"Tool with ID {tool.id} not found")

        tool.updated_at = datetime.now(UTC)
        tool_data = tool.model_dump_json()
        await self._ensure_redis_connection().setex(
            tool_key, self.DEFAULT_TTL, tool_data
        )
        return tool

    async def delete_tool(self, tool_id: UUID) -> bool:
        """Delete a tool."""
        self._ensure_connected()
        tool_key = f"{self.TOOL_PREFIX}{tool_id}"
        deleted_count = await self._ensure_redis_connection().delete(tool_key)
        return bool(deleted_count > 0)

    # MCP Server operations (basic implementation for Redis)
    async def create_mcp_server(self, server: MCPServer) -> MCPServer:
        """Create a new MCP server."""
        self._ensure_connected()
        server_key = f"{self.MCP_SERVER_PREFIX}{server.id}"

        if await self._ensure_redis_connection().exists(server_key):
            raise DuplicateError(f"MCP Server with ID {server.id} already exists")

        server_data = server.model_dump_json()
        await self._ensure_redis_connection().setex(
            server_key, self.DEFAULT_TTL, server_data
        )
        return server

    async def get_mcp_server(self, server_id: UUID) -> MCPServer | None:
        """Get MCP server by ID."""
        self._ensure_connected()
        server_key = f"{self.MCP_SERVER_PREFIX}{server_id}"
        server_data = await self._ensure_redis_connection().get(server_key)

        if server_data:
            data = json.loads(server_data)
            return MCPServer.model_validate(data)

        return None

    async def list_mcp_servers(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[MCPServer]:
        """List MCP servers."""
        self._ensure_connected()
        keys = await self._ensure_redis_connection().keys(f"{self.MCP_SERVER_PREFIX}*")
        servers = []

        for key in keys[offset : offset + limit]:
            server_id = key.decode().replace(self.MCP_SERVER_PREFIX, "")
            server = await self.get_mcp_server(UUID(server_id))
            if server and (not enabled_only or server.enabled):
                servers.append(server)

        return servers

    async def update_mcp_server(self, server: MCPServer) -> MCPServer:
        """Update an existing MCP server."""
        self._ensure_connected()
        server_key = f"{self.MCP_SERVER_PREFIX}{server.id}"

        if not await self._ensure_redis_connection().exists(server_key):
            raise NotFoundError(f"MCP Server with ID {server.id} not found")

        server.updated_at = datetime.now(UTC)
        server_data = server.model_dump_json()
        await self._ensure_redis_connection().setex(
            server_key, self.DEFAULT_TTL, server_data
        )
        return server

    async def delete_mcp_server(self, server_id: UUID) -> bool:
        """Delete an MCP server."""
        self._ensure_connected()
        server_key = f"{self.MCP_SERVER_PREFIX}{server_id}"
        deleted_count = await self._ensure_redis_connection().delete(server_key)
        return bool(deleted_count > 0)
