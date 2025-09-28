"""Retry decorators for external service calls."""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any

import structlog
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..exceptions import ExternalServiceException
from .config import RetryConfig, RetrySettings
from .strategies import ExponentialBackoffStrategy, RetryContext

logger = structlog.get_logger()


def retry_decorator(
    config: RetryConfig,
    strategy: Any | None = None,
    correlation_id: str | None = None,
) -> Callable[..., Any]:
    """Create a retry decorator with the given configuration.

    Args:
        config: Retry configuration
        strategy: Retry strategy (defaults to ExponentialBackoffStrategy)
        correlation_id: Optional correlation ID for logging

    Returns:
        Decorator function
    """
    if strategy is None:
        strategy = ExponentialBackoffStrategy()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = RetryContext(config, strategy)

            while True:
                try:
                    result = await func(*args, **kwargs)

                    if context.attempt > 0:
                        logger.info(
                            "Operation succeeded after retries",
                            function=func.__name__,
                            correlation_id=correlation_id,
                            **context.get_metrics(),
                        )

                    return result

                except Exception as error:
                    if not await context.should_retry(error):
                        logger.error(
                            "Operation failed after all retries",
                            function=func.__name__,
                            correlation_id=correlation_id,
                            **context.get_metrics(),
                        )
                        raise

                    delay = await context.get_delay()

                    logger.warning(
                        "Operation failed, retrying",
                        function=func.__name__,
                        attempt=context.attempt,
                        delay=delay,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        correlation_id=correlation_id,
                    )

                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def get_retry_decorator(
    service_type: str, settings: RetrySettings
) -> Callable[..., Any]:
    """Get a retry decorator for a specific service type.

    Args:
        service_type: Type of service (llm, database, mcp, secret)
        settings: Retry settings

    Returns:
        Retry decorator function
    """
    if service_type == "llm":
        config = settings.get_llm_config()
    elif service_type == "database":
        config = settings.get_database_config()
    elif service_type == "mcp":
        config = settings.get_mcp_config()
    elif service_type == "secret":
        config = settings.get_secret_config()
    else:
        raise ValueError(f"Unknown service type: {service_type}")

    return retry_decorator(config)


# Tenacity-based decorators for compatibility
def tenacity_retry_decorator(
    config: RetryConfig, correlation_id: str | None = None
) -> Callable[..., Any]:
    """Create a tenacity-based retry decorator.

    Args:
        config: Retry configuration
        correlation_id: Optional correlation ID for logging

    Returns:
        Tenacity retry decorator
    """
    # Convert retryable errors to exception types
    retry_exceptions = []
    for error_name in config.retryable_errors:
        try:
            # Try to import and get the exception class
            if "." in error_name:
                module_name, class_name = error_name.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                exception_class = getattr(module, class_name)
                retry_exceptions.append(exception_class)
        except (ImportError, AttributeError):
            # If we can't import the exception, skip it
            logger.warning(f"Could not import exception type: {error_name}")

    # Add common exception types
    retry_exceptions.extend(
        [
            ExternalServiceException,
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
        ]
    )

    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential_jitter(
            initial=config.base_delay,
            max=config.max_delay,
            jitter=2.0 if config.jitter else 0.0,
        ),
        retry=retry_if_exception_type(tuple(retry_exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )


# Service-specific decorators
def llm_retry_decorator(
    settings: RetrySettings, correlation_id: str | None = None
) -> Callable[..., Any]:
    """Get LLM-specific retry decorator."""
    config = settings.get_llm_config()
    return tenacity_retry_decorator(config, correlation_id)


def database_retry_decorator(
    settings: RetrySettings, correlation_id: str | None = None
) -> Callable[..., Any]:
    """Get database-specific retry decorator."""
    config = settings.get_database_config()
    return tenacity_retry_decorator(config, correlation_id)


def mcp_retry_decorator(
    settings: RetrySettings, correlation_id: str | None = None
) -> Callable[..., Any]:
    """Get MCP-specific retry decorator."""
    config = settings.get_mcp_config()
    return tenacity_retry_decorator(config, correlation_id)


def secret_retry_decorator(
    settings: RetrySettings, correlation_id: str | None = None
) -> Callable[..., Any]:
    """Get secret manager-specific retry decorator."""
    config = settings.get_secret_config()
    return tenacity_retry_decorator(config, correlation_id)
