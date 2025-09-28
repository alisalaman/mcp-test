"""Circuit breaker decorators for easy integration."""

import functools
from collections.abc import Callable
from typing import Any

import structlog

from .config import CircuitBreakerSettings
from .manager import CircuitBreakerManager

logger = structlog.get_logger()


def circuit_protected(
    service_name: str, settings: CircuitBreakerSettings | None = None
) -> Callable[..., Any]:
    """Decorator to protect functions with circuit breaker.

    Args:
        service_name: Name of the service to protect
        settings: Optional circuit breaker settings

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get circuit breaker manager (this would typically be injected)
            # For now, we'll create a default one
            if settings is None:
                settings_instance = CircuitBreakerSettings()
            else:
                settings_instance = settings

            manager = CircuitBreakerManager(settings_instance)
            breaker = manager.get_breaker(service_name)

            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def circuit_protected_with_fallback(
    service_name: str,
    fallback_func: Callable[..., Any] | None = None,
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """Decorator to protect functions with circuit breaker and fallback.

    Args:
        service_name: Name of the service to protect
        fallback_func: Optional fallback function to call when circuit is open
        settings: Optional circuit breaker settings

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get circuit breaker manager
            if settings is None:
                settings_instance = CircuitBreakerSettings()
            else:
                settings_instance = settings

            manager = CircuitBreakerManager(settings_instance)
            breaker = manager.get_breaker(service_name)

            try:
                return await breaker.call(func, *args, **kwargs)
            except Exception as e:
                # If we have a fallback function, use it
                if fallback_func is not None:
                    logger.warning(
                        "Circuit breaker protection triggered, using fallback",
                        service_name=service_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    return await fallback_func(*args, **kwargs)
                else:
                    # Re-raise the exception if no fallback
                    raise

        return wrapper

    return decorator


def circuit_protected_with_retry(
    service_name: str,
    retry_decorator: Callable[..., Any] | None = None,
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """Decorator to protect functions with both circuit breaker and retry logic.

    Args:
        service_name: Name of the service to protect
        retry_decorator: Optional retry decorator to apply
        settings: Optional circuit breaker settings

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Apply retry decorator if provided
        if retry_decorator is not None:
            func = retry_decorator(func)

        # Apply circuit breaker protection
        return circuit_protected(service_name, settings)(func)  # type: ignore[no-any-return]  # type: ignore[no-any-return]

    return decorator


# Service-specific decorators
def llm_circuit_protected(
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """LLM-specific circuit breaker decorator."""
    return circuit_protected("llm", settings)


def database_circuit_protected(
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """Database-specific circuit breaker decorator."""
    return circuit_protected("database", settings)


def mcp_circuit_protected(
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """MCP-specific circuit breaker decorator."""
    return circuit_protected("mcp", settings)


def secret_circuit_protected(
    settings: CircuitBreakerSettings | None = None,
) -> Callable[..., Any]:
    """Secret manager-specific circuit breaker decorator."""
    return circuit_protected("secret", settings)
