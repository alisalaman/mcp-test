"""FastAPI dependencies for dependency injection and service management."""

import uuid
from typing import TYPE_CHECKING

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ai_agent.config.settings import ApplicationSettings, get_settings
from ai_agent.infrastructure.database.factory import RepositoryFactory
from ai_agent.infrastructure.database.base import Repository
from ai_agent.security.auth import get_auth_service
from ai_agent.domain.exceptions import AuthenticationException

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
    Get current user from request with proper JWT validation.

    Validates JWT tokens or API keys and returns the authenticated user ID.
    """
    auth_service = get_auth_service()
    settings = get_settings()

    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        try:
            user = auth_service.verify_api_key(api_key)
            if user:
                return str(user.id)
        except Exception as e:
            raise AuthenticationException(f"Invalid API key: {str(e)}")

    # Check for Authorization header with JWT token
    if credentials:
        try:
            user = auth_service.verify_access_token(credentials.credentials)
            return str(user.id)
        except Exception as e:
            raise AuthenticationException(f"Invalid JWT token: {str(e)}")

    # For development, allow anonymous access
    if settings.is_development:
        return "anonymous_user"

    # In production, require authentication
    raise AuthenticationException("Authentication required")


async def get_current_user_with_roles(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> tuple[str, list[str]]:
    """
    Get current user with roles from request with proper JWT validation.

    Returns tuple of (user_id, roles) for authorization purposes.
    """
    auth_service = get_auth_service()
    settings = get_settings()

    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        try:
            user = auth_service.verify_api_key(api_key)
            if user:
                roles = [role.value for role in user.roles]
                return user.id, roles
        except Exception as e:
            raise AuthenticationException(f"Invalid API key: {str(e)}")

    # Check for Authorization header with JWT token
    if credentials:
        try:
            user = auth_service.verify_access_token(credentials.credentials)
            roles = [role.value for role in user.roles]
            return user.id, roles
        except Exception as e:
            raise AuthenticationException(f"Invalid JWT token: {str(e)}")

    # For development, allow anonymous access with default role
    if settings.is_development:
        return "anonymous_user", ["readonly"]

    # In production, require authentication
    raise AuthenticationException("Authentication required")


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
