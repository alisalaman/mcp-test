"""Integration decorators combining all resilience patterns."""

import functools
from collections.abc import Callable
from typing import Any

import structlog

from .circuit_breaker import CircuitBreakerManager
from .exceptions import (
    CircuitBreakerOpenException,
    FallbackFailedException,
    RateLimitExceededException,
)
from .fallback import FallbackManager
from .rate_limiting import RateLimitManager
from .retry import RetryManager

logger = structlog.get_logger()


def resilient_service(  # type: ignore[no-untyped-def]
    service_name: str,
    retry_manager: RetryManager | None = None,
    circuit_breaker_manager: CircuitBreakerManager | None = None,
    fallback_manager: FallbackManager | None = None,
    rate_limit_manager: RateLimitManager | None = None,
    rate_limit_key: str | None = None,
    correlation_id: str | None = None,
):
    """Decorator that applies all resilience patterns to a service call.

    Args:
        service_name: Name of the service
        retry_manager: Retry manager instance
        circuit_breaker_manager: Circuit breaker manager instance
        fallback_manager: Fallback manager instance
        rate_limit_manager: Rate limit manager instance
        rate_limit_key: Key for rate limiting (defaults to service_name)
        correlation_id: Optional correlation ID for logging

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Rate limiting check
            if rate_limit_manager:
                key = rate_limit_key or service_name
                rate_result = await rate_limit_manager.is_allowed(service_name, key)

                if not rate_result.allowed:
                    logger.warning(
                        "Rate limit exceeded",
                        service_name=service_name,
                        key=key,
                        retry_after=rate_result.retry_after,
                        correlation_id=correlation_id,
                    )
                    raise RateLimitExceededException(
                        service_name, rate_result.retry_after, correlation_id
                    )

                # Consume tokens
                await rate_limit_manager.consume(service_name, key)

            # Circuit breaker check
            if circuit_breaker_manager:
                breaker = circuit_breaker_manager.get_breaker(service_name)
                if breaker.state.value == "open":
                    logger.warning(
                        "Circuit breaker is open",
                        service_name=service_name,
                        correlation_id=correlation_id,
                    )
                    raise CircuitBreakerOpenException(service_name, correlation_id)

            # Try original function with retry logic
            try:
                if retry_manager:
                    retry_decorator = retry_manager.get_retry_decorator(
                        service_name, correlation_id
                    )
                    return await retry_decorator(func)(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)

            except Exception as e:
                # Try fallback if available
                if fallback_manager:
                    try:
                        logger.info(
                            "Attempting fallback",
                            service_name=service_name,
                            error=str(e),
                            correlation_id=correlation_id,
                        )
                        return await fallback_manager.handle_service_call(
                            service_name, func, *args, **kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(
                            "Fallback also failed",
                            service_name=service_name,
                            original_error=str(e),
                            fallback_error=str(fallback_error),
                            correlation_id=correlation_id,
                        )
                        raise FallbackFailedException(
                            service_name, "all_strategies", correlation_id
                        ) from fallback_error
                else:
                    # Re-raise original exception if no fallback
                    raise

        return wrapper

    return decorator


def llm_resilient(  # type: ignore[no-untyped-def]
    retry_manager: RetryManager | None = None,
    circuit_breaker_manager: CircuitBreakerManager | None = None,
    fallback_manager: FallbackManager | None = None,
    rate_limit_manager: RateLimitManager | None = None,
    correlation_id: str | None = None,
):
    """LLM-specific resilient decorator."""
    return resilient_service(
        service_name="llm",
        retry_manager=retry_manager,
        circuit_breaker_manager=circuit_breaker_manager,
        fallback_manager=fallback_manager,
        rate_limit_manager=rate_limit_manager,
        correlation_id=correlation_id,
    )


def database_resilient(  # type: ignore[no-untyped-def]
    retry_manager: RetryManager | None = None,
    circuit_breaker_manager: CircuitBreakerManager | None = None,
    fallback_manager: FallbackManager | None = None,
    rate_limit_manager: RateLimitManager | None = None,
    correlation_id: str | None = None,
):
    """Database-specific resilient decorator."""
    return resilient_service(
        service_name="database",
        retry_manager=retry_manager,
        circuit_breaker_manager=circuit_breaker_manager,
        fallback_manager=fallback_manager,
        rate_limit_manager=rate_limit_manager,
        correlation_id=correlation_id,
    )


def mcp_resilient(  # type: ignore[no-untyped-def]
    retry_manager: RetryManager | None = None,
    circuit_breaker_manager: CircuitBreakerManager | None = None,
    fallback_manager: FallbackManager | None = None,
    rate_limit_manager: RateLimitManager | None = None,
    correlation_id: str | None = None,
):
    """MCP-specific resilient decorator."""
    return resilient_service(
        service_name="mcp",
        retry_manager=retry_manager,
        circuit_breaker_manager=circuit_breaker_manager,
        fallback_manager=fallback_manager,
        rate_limit_manager=rate_limit_manager,
        correlation_id=correlation_id,
    )


def secret_resilient(  # type: ignore[no-untyped-def]
    retry_manager: RetryManager | None = None,
    circuit_breaker_manager: CircuitBreakerManager | None = None,
    fallback_manager: FallbackManager | None = None,
    rate_limit_manager: RateLimitManager | None = None,
    correlation_id: str | None = None,
):
    """Secret manager-specific resilient decorator."""
    return resilient_service(
        service_name="secret",
        retry_manager=retry_manager,
        circuit_breaker_manager=circuit_breaker_manager,
        fallback_manager=fallback_manager,
        rate_limit_manager=rate_limit_manager,
        correlation_id=correlation_id,
    )


class ResilienceManager:
    """Centralized manager for all resilience patterns."""

    def __init__(
        self,
        retry_manager: RetryManager | None = None,
        circuit_breaker_manager: CircuitBreakerManager | None = None,
        fallback_manager: FallbackManager | None = None,
        rate_limit_manager: RateLimitManager | None = None,
    ):
        """Initialize resilience manager.

        Args:
            retry_manager: Retry manager instance
            circuit_breaker_manager: Circuit breaker manager instance
            fallback_manager: Fallback manager instance
            rate_limit_manager: Rate limit manager instance
        """
        self.retry_manager = retry_manager
        self.circuit_breaker_manager = circuit_breaker_manager
        self.fallback_manager = fallback_manager
        self.rate_limit_manager = rate_limit_manager

    def get_resilient_decorator(  # type: ignore[no-untyped-def]
        self, service_name: str, correlation_id: str | None = None
    ):
        """Get resilient decorator for a service.

        Args:
            service_name: Name of the service
            correlation_id: Optional correlation ID for logging

        Returns:
            Resilient decorator
        """
        return resilient_service(
            service_name=service_name,
            retry_manager=self.retry_manager,
            circuit_breaker_manager=self.circuit_breaker_manager,
            fallback_manager=self.fallback_manager,
            rate_limit_manager=self.rate_limit_manager,
            correlation_id=correlation_id,
        )

    def get_llm_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get LLM resilient decorator."""
        return llm_resilient(  # type: ignore[no-any-return]
            retry_manager=self.retry_manager,
            circuit_breaker_manager=self.circuit_breaker_manager,
            fallback_manager=self.fallback_manager,
            rate_limit_manager=self.rate_limit_manager,
            correlation_id=correlation_id,
        )

    def get_database_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get database resilient decorator."""
        return database_resilient(  # type: ignore[no-any-return]
            retry_manager=self.retry_manager,
            circuit_breaker_manager=self.circuit_breaker_manager,
            fallback_manager=self.fallback_manager,
            rate_limit_manager=self.rate_limit_manager,
            correlation_id=correlation_id,
        )

    def get_mcp_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get MCP resilient decorator."""
        return mcp_resilient(  # type: ignore[no-any-return]
            retry_manager=self.retry_manager,
            circuit_breaker_manager=self.circuit_breaker_manager,
            fallback_manager=self.fallback_manager,
            rate_limit_manager=self.rate_limit_manager,
            correlation_id=correlation_id,
        )

    def get_secret_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get secret manager resilient decorator."""
        return secret_resilient(  # type: ignore[no-any-return]
            retry_manager=self.retry_manager,
            circuit_breaker_manager=self.circuit_breaker_manager,
            fallback_manager=self.fallback_manager,
            rate_limit_manager=self.rate_limit_manager,
            correlation_id=correlation_id,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all resilience components.

        Returns:
            Statistics dictionary
        """
        stats = {}

        if self.retry_manager:
            stats["retry"] = self.retry_manager.get_stats()

        if self.circuit_breaker_manager:
            stats["circuit_breaker"] = self.circuit_breaker_manager.get_global_stats()

        if self.fallback_manager:
            stats["fallback"] = self.fallback_manager.get_stats()

        if self.rate_limit_manager:
            stats["rate_limiting"] = self.rate_limit_manager.get_stats()

        return stats
