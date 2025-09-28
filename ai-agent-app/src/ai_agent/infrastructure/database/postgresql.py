"""
PostgreSQL repository implementation for production persistence.

This provides full ACID compliance, efficient querying with proper indexes,
connection pooling, and transaction support for data integrity.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False

from ai_agent.config.settings import DatabaseSettings
from ai_agent.domain.models import (
    Agent,
    AgentStatus,
    MCPServer,
    Message,
    MessageRole,
    Session,
    Tool,
)
from ai_agent.infrastructure.database.base import (
    BaseRepository,
    ConnectionError,
    DuplicateError,
    NotFoundError,
)
from ai_agent.observability.logging import get_logger


class PostgreSQLRepository(BaseRepository):
    """PostgreSQL-based storage for full persistence."""

    def __init__(self, settings: DatabaseSettings):
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is not available. Install with: pip install asyncpg"
            ) from None

        super().__init__()
        self.settings = settings
        self._pool: asyncpg.Pool | None = None
        self.logger = get_logger(__name__)

    async def connect(self) -> None:
        """Initialize database connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.settings.host,
                port=self.settings.port,
                database=self.settings.name,
                user=self.settings.user,
                password=self.settings.password,
                min_size=self.settings.min_pool_size,
                max_size=self.settings.max_pool_size,
                command_timeout=self.settings.pool_timeout,
                # Add connection health checks and proper cleanup
                max_queries=50000,  # Close connections after 50k queries
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                setup=self._setup_connection,  # Setup function for each connection
            )
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    async def _setup_connection(self, conn: Any) -> None:
        """Setup function for each new connection."""
        # Set connection-specific settings
        await conn.execute("SET statement_timeout = '30s'")
        await conn.execute("SET idle_in_transaction_session_timeout = '60s'")

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            try:
                # Gracefully close all connections
                await self._pool.close()
                self.logger.info("PostgreSQL connection pool closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing PostgreSQL pool: {e}")
            finally:
                self._pool = None
        self._connected = False

    async def health_check(self) -> bool:
        """Check database connection health."""
        if not self.is_connected or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Any]:
        """Get database connection from pool."""
        self._ensure_connected()
        if not self._pool:
            raise ConnectionError("Database pool not available")
        async with self._pool.acquire() as connection:
            yield connection

    # Session operations
    async def create_session(self, session: Session) -> Session:
        """Create a new session in PostgreSQL."""
        async with self.get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO sessions (
                        id, user_id, title, metadata, message_count,
                        last_activity, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    session.id,
                    session.user_id,
                    session.title,
                    session.metadata,
                    session.message_count,
                    session.last_activity,
                    session.created_at,
                    session.updated_at,
                )
                return session
            except asyncpg.UniqueViolationError:
                raise DuplicateError(f"Session with ID {session.id} already exists")

    async def get_session(self, session_id: UUID) -> Session | None:
        """Get session from PostgreSQL."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1", session_id
            )

            if row:
                return Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row["title"],
                    metadata=row["metadata"],
                    message_count=row["message_count"],
                    last_activity=row["last_activity"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

        return None

    async def list_sessions(
        self, user_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Session]:
        """List sessions with optional filtering."""
        async with self.get_connection() as conn:
            if user_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM sessions
                    WHERE user_id = $1
                    ORDER BY last_activity DESC
                    LIMIT $2 OFFSET $3
                """,
                    user_id,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM sessions
                    ORDER BY last_activity DESC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            return [
                Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row["title"],
                    metadata=row["metadata"],
                    message_count=row["message_count"],
                    last_activity=row["last_activity"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        async with self.get_connection() as conn:
            session.updated_at = datetime.now(UTC)

            result = await conn.execute(
                """
                UPDATE sessions SET
                    user_id = $2,
                    title = $3,
                    metadata = $4,
                    message_count = $5,
                    last_activity = $6,
                    updated_at = $7
                WHERE id = $1
            """,
                session.id,
                session.user_id,
                session.title,
                session.metadata,
                session.message_count,
                session.last_activity,
                session.updated_at,
            )

            if result == "UPDATE 0":
                raise NotFoundError(f"Session with ID {session.id} not found")

            return session

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Delete all messages in the session (CASCADE should handle this)
                await conn.execute(
                    "DELETE FROM messages WHERE session_id = $1", session_id
                )

                # Delete session
                result = await conn.execute(
                    "DELETE FROM sessions WHERE id = $1", session_id
                )

                return bool(result == "DELETE 1")

    # Message operations
    async def create_message(self, message: Message) -> Message:
        """Create a new message in PostgreSQL."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                try:
                    # Insert message
                    await conn.execute(
                        """
                        INSERT INTO messages (
                            id, session_id, role, content, metadata,
                            tool_calls, parent_message_id, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        message.id,
                        message.session_id,
                        message.role.value,
                        message.content,
                        message.metadata,
                        message.tool_calls,
                        message.parent_message_id,
                        message.created_at,
                        message.updated_at,
                    )

                    # Update session message count and last activity
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET message_count = message_count + 1,
                            last_activity = $2,
                            updated_at = $2
                        WHERE id = $1
                    """,
                        message.session_id,
                        datetime.now(UTC),
                    )

                    return message
                except asyncpg.UniqueViolationError:
                    raise DuplicateError(f"Message with ID {message.id} already exists")
                except asyncpg.ForeignKeyViolationError:
                    raise NotFoundError(
                        f"Session with ID {message.session_id} not found"
                    )

    async def get_message(self, message_id: UUID) -> Message | None:
        """Get message from PostgreSQL."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM messages WHERE id = $1", message_id
            )

            if row:
                return Message(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    metadata=row["metadata"],
                    tool_calls=row["tool_calls"],
                    parent_message_id=row["parent_message_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

        return None

    async def get_messages_by_session(
        self, session_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """Get messages for a session."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM messages
                WHERE session_id = $1
                ORDER BY created_at ASC
                LIMIT $2 OFFSET $3
            """,
                session_id,
                limit,
                offset,
            )

            return [
                Message(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    metadata=row["metadata"],
                    tool_calls=row["tool_calls"],
                    parent_message_id=row["parent_message_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def update_message(self, message: Message) -> Message:
        """Update an existing message."""
        async with self.get_connection() as conn:
            message.updated_at = datetime.now(UTC)

            result = await conn.execute(
                """
                UPDATE messages SET
                    role = $2,
                    content = $3,
                    metadata = $4,
                    tool_calls = $5,
                    parent_message_id = $6,
                    updated_at = $7
                WHERE id = $1
            """,
                message.id,
                message.role.value,
                message.content,
                message.metadata,
                message.tool_calls,
                message.parent_message_id,
                message.updated_at,
            )

            if result == "UPDATE 0":
                raise NotFoundError(f"Message with ID {message.id} not found")

            return message

    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Get message to find session
                message = await self.get_message(message_id)
                if not message:
                    return False

                # Delete message
                result = await conn.execute(
                    "DELETE FROM messages WHERE id = $1", message_id
                )

                # Update session message count
                if result == "DELETE 1":
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET message_count = message_count - 1,
                            updated_at = $2
                        WHERE id = $1
                    """,
                        message.session_id,
                        datetime.now(UTC),
                    )
                    return True

                return False

    # Agent operations
    async def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent."""
        async with self.get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO agents (
                        id, name, description, system_prompt, model_config,
                        tools, status, metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    agent.id,
                    agent.name,
                    agent.description,
                    agent.system_prompt,
                    agent.model_config,
                    agent.tools,
                    agent.status.value,
                    agent.metadata,
                    agent.created_at,
                    agent.updated_at,
                )
                return agent
            except asyncpg.UniqueViolationError:
                raise DuplicateError(f"Agent with name '{agent.name}' already exists")

    async def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow("SELECT * FROM agents WHERE id = $1", agent_id)

            if row:
                return Agent(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    system_prompt=row["system_prompt"],
                    model_config=row["model_config"],
                    tools=row["tools"],
                    status=AgentStatus(row["status"]),
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

        return None

    async def list_agents(self, limit: int = 100, offset: int = 0) -> list[Agent]:
        """List agents."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM agents
                ORDER BY name ASC
                LIMIT $1 OFFSET $2
            """,
                limit,
                offset,
            )

            return [
                Agent(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    system_prompt=row["system_prompt"],
                    model_config=row["model_config"],
                    tools=row["tools"],
                    status=AgentStatus(row["status"]),
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent."""
        async with self.get_connection() as conn:
            agent.updated_at = datetime.now(UTC)

            try:
                result = await conn.execute(
                    """
                    UPDATE agents SET
                        name = $2,
                        description = $3,
                        system_prompt = $4,
                        model_config = $5,
                        tools = $6,
                        status = $7,
                        metadata = $8,
                        updated_at = $9
                    WHERE id = $1
                """,
                    agent.id,
                    agent.name,
                    agent.description,
                    agent.system_prompt,
                    agent.model_config,
                    agent.tools,
                    agent.status.value,
                    agent.metadata,
                    agent.updated_at,
                )

                if result == "UPDATE 0":
                    raise NotFoundError(f"Agent with ID {agent.id} not found")

                return agent
            except asyncpg.UniqueViolationError:
                raise DuplicateError(f"Agent with name '{agent.name}' already exists")

    async def delete_agent(self, agent_id: UUID) -> bool:
        """Delete an agent."""
        async with self.get_connection() as conn:
            result = await conn.execute("DELETE FROM agents WHERE id = $1", agent_id)
            return bool(result == "DELETE 1")

    # Tool operations
    async def create_tool(self, tool: Tool) -> Tool:
        """Create a new tool."""
        async with self.get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO tools (
                        id, name, description, schema, mcp_server_id,
                        enabled, metadata, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    tool.id,
                    tool.name,
                    tool.description,
                    tool.schema,
                    tool.mcp_server_id,
                    tool.enabled,
                    tool.metadata,
                    tool.created_at,
                    tool.updated_at,
                )
                return tool
            except asyncpg.UniqueViolationError:
                raise DuplicateError(f"Tool with name '{tool.name}' already exists")

    async def get_tool(self, tool_id: UUID) -> Tool | None:
        """Get tool by ID."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow("SELECT * FROM tools WHERE id = $1", tool_id)

            if row:
                return Tool(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    schema=row["schema"],
                    mcp_server_id=row["mcp_server_id"],
                    enabled=row["enabled"],
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

        return None

    async def list_tools(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[Tool]:
        """List tools."""
        async with self.get_connection() as conn:
            if enabled_only:
                rows = await conn.fetch(
                    """
                    SELECT * FROM tools
                    WHERE enabled = true
                    ORDER BY name ASC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM tools
                    ORDER BY name ASC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            return [
                Tool(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    schema=row["schema"],
                    mcp_server_id=row["mcp_server_id"],
                    enabled=row["enabled"],
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def update_tool(self, tool: Tool) -> Tool:
        """Update an existing tool."""
        async with self.get_connection() as conn:
            tool.updated_at = datetime.now(UTC)

            try:
                result = await conn.execute(
                    """
                    UPDATE tools SET
                        name = $2,
                        description = $3,
                        schema = $4,
                        mcp_server_id = $5,
                        enabled = $6,
                        metadata = $7,
                        updated_at = $8
                    WHERE id = $1
                """,
                    tool.id,
                    tool.name,
                    tool.description,
                    tool.schema,
                    tool.mcp_server_id,
                    tool.enabled,
                    tool.metadata,
                    tool.updated_at,
                )

                if result == "UPDATE 0":
                    raise NotFoundError(f"Tool with ID {tool.id} not found")

                return tool
            except asyncpg.UniqueViolationError:
                raise DuplicateError(f"Tool with name '{tool.name}' already exists")

    async def delete_tool(self, tool_id: UUID) -> bool:
        """Delete a tool."""
        async with self.get_connection() as conn:
            result = await conn.execute("DELETE FROM tools WHERE id = $1", tool_id)
            return bool(result == "DELETE 1")

    # MCP Server operations
    async def create_mcp_server(self, server: MCPServer) -> MCPServer:
        """Create a new MCP server."""
        async with self.get_connection() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO mcp_servers (
                        id, name, description, endpoint, authentication,
                        capabilities, health_check_url, enabled, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    server.id,
                    server.name,
                    server.description,
                    server.endpoint,
                    server.authentication,
                    server.capabilities,
                    server.health_check_url,
                    server.enabled,
                    server.created_at,
                    server.updated_at,
                )
                return server
            except asyncpg.UniqueViolationError:
                raise DuplicateError(
                    f"MCP Server with name '{server.name}' already exists"
                )

    async def get_mcp_server(self, server_id: UUID) -> MCPServer | None:
        """Get MCP server by ID."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_servers WHERE id = $1", server_id
            )

            if row:
                return MCPServer(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    endpoint=row["endpoint"],
                    authentication=row["authentication"],
                    capabilities=row["capabilities"],
                    health_check_url=row["health_check_url"],
                    enabled=row["enabled"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

        return None

    async def list_mcp_servers(
        self, enabled_only: bool = True, limit: int = 100, offset: int = 0
    ) -> list[MCPServer]:
        """List MCP servers."""
        async with self.get_connection() as conn:
            if enabled_only:
                rows = await conn.fetch(
                    """
                    SELECT * FROM mcp_servers
                    WHERE enabled = true
                    ORDER BY name ASC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM mcp_servers
                    ORDER BY name ASC
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            return [
                MCPServer(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    endpoint=row["endpoint"],
                    authentication=row["authentication"],
                    capabilities=row["capabilities"],
                    health_check_url=row["health_check_url"],
                    enabled=row["enabled"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def update_mcp_server(self, server: MCPServer) -> MCPServer:
        """Update an existing MCP server."""
        async with self.get_connection() as conn:
            server.updated_at = datetime.now(UTC)

            try:
                result = await conn.execute(
                    """
                    UPDATE mcp_servers SET
                        name = $2,
                        description = $3,
                        endpoint = $4,
                        authentication = $5,
                        capabilities = $6,
                        health_check_url = $7,
                        enabled = $8,
                        updated_at = $9
                    WHERE id = $1
                """,
                    server.id,
                    server.name,
                    server.description,
                    server.endpoint,
                    server.authentication,
                    server.capabilities,
                    server.health_check_url,
                    server.enabled,
                    server.updated_at,
                )

                if result == "UPDATE 0":
                    raise NotFoundError(f"MCP Server with ID {server.id} not found")

                return server
            except asyncpg.UniqueViolationError:
                raise DuplicateError(
                    f"MCP Server with name '{server.name}' already exists"
                )

    async def delete_mcp_server(self, server_id: UUID) -> bool:
        """Delete an MCP server."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                # First, update tools to remove server reference
                await conn.execute(
                    "UPDATE tools SET mcp_server_id = NULL WHERE mcp_server_id = $1",
                    server_id,
                )

                # Then delete the server
                result = await conn.execute(
                    "DELETE FROM mcp_servers WHERE id = $1", server_id
                )

                return bool(result == "DELETE 1")
