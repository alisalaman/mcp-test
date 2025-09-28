"""Retry manager for centralized retry configuration and management."""

from collections.abc import Callable
from typing import Any

import structlog

from .config import RetryConfig, RetrySettings
from .decorators import (
    database_retry_decorator,
    llm_retry_decorator,
    mcp_retry_decorator,
    secret_retry_decorator,
    tenacity_retry_decorator,
)

logger = structlog.get_logger()


class RetryManager:
    """Centralized retry management for external services."""

    def __init__(self, settings: RetrySettings):
        """Initialize retry manager with settings."""
        self.settings = settings
        self._decorators: dict[str, Callable[..., Any]] = {}

    def get_llm_retry_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get retry decorator for LLM API calls.

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        cache_key = f"llm_{correlation_id or 'default'}"
        if cache_key not in self._decorators:
            self._decorators[cache_key] = llm_retry_decorator(
                self.settings, correlation_id
            )
        return self._decorators[cache_key]

    def get_database_retry_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get retry decorator for database operations.

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        cache_key = f"database_{correlation_id or 'default'}"
        if cache_key not in self._decorators:
            self._decorators[cache_key] = database_retry_decorator(
                self.settings, correlation_id
            )
        return self._decorators[cache_key]

    def get_mcp_retry_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get retry decorator for MCP server calls.

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        cache_key = f"mcp_{correlation_id or 'default'}"
        if cache_key not in self._decorators:
            self._decorators[cache_key] = mcp_retry_decorator(
                self.settings, correlation_id
            )
        return self._decorators[cache_key]

    def get_secret_retry_decorator(
        self, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get retry decorator for secret manager calls.

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        cache_key = f"secret_{correlation_id or 'default'}"
        if cache_key not in self._decorators:
            self._decorators[cache_key] = secret_retry_decorator(
                self.settings, correlation_id
            )
        return self._decorators[cache_key]

    def get_retry_decorator(
        self, service_type: str, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Get retry decorator for a specific service type.

        Args:
            service_type: Type of service (llm, database, mcp, secret)
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        if service_type == "llm":
            return self.get_llm_retry_decorator(correlation_id)
        elif service_type == "database":
            return self.get_database_retry_decorator(correlation_id)
        elif service_type == "mcp":
            return self.get_mcp_retry_decorator(correlation_id)
        elif service_type == "secret":
            return self.get_secret_retry_decorator(correlation_id)
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    def get_config(self, service_type: str) -> RetryConfig:
        """Get retry configuration for a service type.

        Args:
            service_type: Type of service (llm, database, mcp, secret)

        Returns:
            Retry configuration
        """
        if service_type == "llm":
            return self.settings.get_llm_config()
        elif service_type == "database":
            return self.settings.get_database_config()
        elif service_type == "mcp":
            return self.settings.get_mcp_config()
        elif service_type == "secret":
            return self.settings.get_secret_config()
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    def create_custom_decorator(
        self, config: RetryConfig, correlation_id: str | None = None
    ) -> Callable[..., Any]:
        """Create a custom retry decorator with specific configuration.

        Args:
            config: Custom retry configuration
            correlation_id: Optional correlation ID for logging

        Returns:
            Retry decorator function
        """
        return tenacity_retry_decorator(config, correlation_id)

    def clear_cache(self) -> None:
        """Clear the decorator cache."""
        self._decorators.clear()
        logger.info("Retry decorator cache cleared")

    def get_stats(self) -> dict[str, int]:
        """Get retry manager statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_decorators": len(self._decorators),
            "service_types": len(
                {key.split("_")[0] for key in self._decorators.keys()}
            ),
        }
