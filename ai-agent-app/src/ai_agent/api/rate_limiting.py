"""Rate limiting implementation for the API."""

from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on user tier and IP."""
    # Get user tier from request state (set by middleware)
    user_tier = getattr(request.state, "user_tier", "default")
    ip_address = get_remote_address(request)

    # Create key based on user tier and IP
    return f"{user_tier}:{ip_address}"


# Initialize rate limiter
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=["100/minute"],  # Default rate limit
)

# Rate limit configurations based on user tier
RATE_LIMITS = {
    "default": "100/minute",
    "authenticated": "1000/minute",
    "premium": "5000/minute",
}

# Endpoint-specific rate limits
ENDPOINT_LIMITS = {
    "create_session": "10/minute",
    "create_message": "50/minute",
    "execute_agent": "5/minute",
    "bulk_create": "5/minute",
    "bulk_delete": "5/minute",
}


def get_user_rate_limit(user_tier: str) -> str:
    """Get rate limit string for user tier."""
    return RATE_LIMITS.get(user_tier, RATE_LIMITS["default"])


def rate_limit_by_tier(user_tier: str) -> Callable[..., Any]:
    """Create rate limit decorator based on user tier."""
    limit = get_user_rate_limit(user_tier)
    return limiter.limit(limit)  # type: ignore[no-any-return]


def rate_limit_endpoint(endpoint_name: str) -> Callable[..., Any]:
    """Create rate limit decorator for specific endpoint."""
    limit = ENDPOINT_LIMITS.get(endpoint_name, "100/minute")
    return limiter.limit(limit)  # type: ignore[no-any-return]


# Rate limit exceeded handler
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Handle rate limit exceeded errors."""
    return _rate_limit_exceeded_handler(request, exc)  # type: ignore[no-any-return]
