"""FastAPI dependencies for dependency injection and service management."""

import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from typing import TYPE_CHECKING

from ai_agent.config.settings import ApplicationSettings, get_settings
from ai_agent.infrastructure.database.factory import RepositoryFactory
from ai_agent.infrastructure.database.base import Repository

# Security scheme
security = HTTPBearer(auto_error=False)

if TYPE_CHECKING:
    from ai_agent.core.sessions.service import SessionService
    from ai_agent.core.messages.service import MessageService
    from ai_agent.core.agents.service import AgentService
    from ai_agent.core.tools.service import ToolService
    from ai_agent.core.mcp.service import MCPService


async def get_settings_dependency() -> ApplicationSettings:
    """Get application settings."""
    return get_settings()


async def get_repository() -> Repository:
    """Get repository instance for dependency injection."""
    settings = get_settings()
    repository = RepositoryFactory.create_repository(settings)
    await repository.connect()
    return repository


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """
    Get current user from request.

    For now, this is a simplified implementation that extracts user from headers.
    In production, this would validate JWT tokens or API keys.
    """
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # In production, validate API key against database
        return f"api_user_{api_key[:8]}"

    # Check for Authorization header
    if credentials:
        # In production, validate JWT token
        return f"jwt_user_{credentials.credentials[:8]}"

    # For development, allow anonymous access
    if get_settings().is_development:
        return "anonymous_user"

    # In production, require authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_session_service(
    repository: Repository = Depends(get_repository),
    current_user: str = Depends(get_current_user),
) -> "SessionService":
    """Get session service with dependencies."""
    from ai_agent.core.sessions.service import SessionService

    return SessionService(repository, current_user)


async def get_message_service(
    repository: Repository = Depends(get_repository),
    current_user: str = Depends(get_current_user),
) -> "MessageService":
    """Get message service with dependencies."""
    from ai_agent.core.messages.service import MessageService

    return MessageService(repository, current_user)


async def get_agent_service(
    repository: Repository = Depends(get_repository),
    current_user: str = Depends(get_current_user),
) -> "AgentService":
    """Get agent service with dependencies."""
    from ai_agent.core.agents.service import AgentService

    return AgentService(repository, current_user)


async def get_tool_service(
    repository: Repository = Depends(get_repository),
    current_user: str = Depends(get_current_user),
) -> "ToolService":
    """Get tool service with dependencies."""
    from ai_agent.core.tools.service import ToolService

    return ToolService(repository, current_user)


async def get_mcp_service(
    repository: Repository = Depends(get_repository),
    current_user: str = Depends(get_current_user),
) -> "MCPService":
    """Get MCP service with dependencies."""
    from ai_agent.core.mcp.service import MCPService

    return MCPService(repository, current_user)


def get_correlation_id(request: Request) -> str:
    """Extract correlation ID from request headers."""
    return request.headers.get("X-Correlation-ID", str(uuid.uuid4()))


def get_user_tier(current_user: str = Depends(get_current_user)) -> str:
    """Get user tier for rate limiting."""
    # Simplified implementation - in production, this would check user subscription
    if current_user.startswith("premium_"):
        return "premium"
    elif current_user.startswith("api_") or current_user.startswith("jwt_"):
        return "authenticated"
    else:
        return "default"
