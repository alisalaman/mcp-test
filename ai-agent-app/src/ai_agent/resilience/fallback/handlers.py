"""Fallback handlers for service integration."""

from collections.abc import Callable
from typing import Any

import structlog

from .strategies import DefaultValueFallbackStrategy, FallbackConfig, FallbackStrategy

logger = structlog.get_logger()


class FallbackHandler:
    """Base fallback handler."""

    def __init__(self, config: FallbackConfig):
        """Initialize fallback handler.

        Args:
            config: Fallback configuration
        """
        self.config = config
        self.strategies: list[FallbackStrategy] = []

    def add_strategy(self, strategy: FallbackStrategy) -> None:
        """Add a fallback strategy.

        Args:
            strategy: Fallback strategy to add
        """
        self.strategies.append(strategy)
        logger.info("Added fallback strategy", strategy_type=type(strategy).__name__)

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Execute fallback strategies in order.

        Args:
            service_name: Name of the service
            original_func: Original function that failed
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Fallback result
        """
        if not self.strategies:
            logger.warning(
                "No fallback strategies available", service_name=service_name
            )
            return None

        last_error = None

        for strategy in self.strategies:
            try:
                logger.info(
                    "Trying fallback strategy",
                    service_name=service_name,
                    strategy_type=type(strategy).__name__,
                )

                result = await strategy.execute_fallback(
                    service_name, original_func, *args, **kwargs
                )

                if result is not None:
                    logger.info(
                        "Fallback strategy succeeded",
                        service_name=service_name,
                        strategy_type=type(strategy).__name__,
                    )
                    return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Fallback strategy failed",
                    service_name=service_name,
                    strategy_type=type(strategy).__name__,
                    error=str(e),
                )
                continue

        # All strategies failed
        logger.error(
            "All fallback strategies failed",
            service_name=service_name,
            last_error=str(last_error) if last_error else "Unknown",
        )

        if last_error:
            raise last_error
        else:
            raise RuntimeError(
                f"All fallback strategies failed for service {service_name}"
            )


class ServiceFallbackHandler(FallbackHandler):
    """Service-specific fallback handler."""

    def __init__(
        self, service_name: str, config: FallbackConfig, default_value: Any = None
    ):
        """Initialize service fallback handler.

        Args:
            service_name: Name of the service
            config: Fallback configuration
            default_value: Default value for this service
        """
        super().__init__(config)
        self.service_name = service_name

        # Add default value strategy
        if default_value is not None:
            default_strategy = DefaultValueFallbackStrategy(config, default_value)
            self.add_strategy(default_strategy)

    async def handle_service_call(  # type: ignore[no-untyped-def]
        self, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Handle a service call with fallback.

        Args:
            original_func: Original function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result
        """
        try:
            # Try original function first
            return await original_func(*args, **kwargs)

        except Exception as e:
            logger.warning(
                "Service call failed, attempting fallback",
                service_name=self.service_name,
                error=str(e),
            )

            # Execute fallback strategies
            return await self.execute_fallback(
                self.service_name, original_func, *args, **kwargs
            )


class LLMFallbackHandler(ServiceFallbackHandler):
    """LLM-specific fallback handler."""

    def __init__(self, config: FallbackConfig):
        """Initialize LLM fallback handler.

        Args:
            config: Fallback configuration
        """
        super().__init__(
            "llm",
            config,
            "I'm sorry, I'm currently unable to process your request. Please try again later.",
        )

        # Add LLM-specific strategies
        self._add_llm_strategies()

    def _add_llm_strategies(self) -> None:
        """Add LLM-specific fallback strategies."""
        # Add cached response strategy
        from .strategies import CachedFallbackStrategy

        cache_strategy = CachedFallbackStrategy(self.config)
        self.add_strategy(cache_strategy)

        # Add retry with backoff strategy
        from .strategies import RetryWithBackoffFallbackStrategy

        retry_strategy = RetryWithBackoffFallbackStrategy(
            self.config, max_retries=2, base_delay=1.0
        )
        self.add_strategy(retry_strategy)


class DatabaseFallbackHandler(ServiceFallbackHandler):
    """Database-specific fallback handler."""

    def __init__(self, config: FallbackConfig):
        """Initialize database fallback handler.

        Args:
            config: Fallback configuration
        """
        super().__init__("database", config, None)

        # Add database-specific strategies
        self._add_database_strategies()

    def _add_database_strategies(self) -> None:
        """Add database-specific fallback strategies."""
        # Add cached data strategy
        from .strategies import CachedFallbackStrategy

        cache_strategy = CachedFallbackStrategy(self.config)
        self.add_strategy(cache_strategy)

        # Add retry with backoff strategy
        from .strategies import RetryWithBackoffFallbackStrategy

        retry_strategy = RetryWithBackoffFallbackStrategy(
            self.config, max_retries=3, base_delay=0.5
        )
        self.add_strategy(retry_strategy)


class MCPFallbackHandler(ServiceFallbackHandler):
    """MCP-specific fallback handler."""

    def __init__(self, config: FallbackConfig):
        """Initialize MCP fallback handler.

        Args:
            config: Fallback configuration
        """
        super().__init__("mcp", config, None)

        # Add MCP-specific strategies
        self._add_mcp_strategies()

    def _add_mcp_strategies(self) -> None:
        """Add MCP-specific fallback strategies."""
        # Add cached tool result strategy
        from .strategies import CachedFallbackStrategy

        cache_strategy = CachedFallbackStrategy(self.config)
        self.add_strategy(cache_strategy)

        # Add retry with backoff strategy
        from .strategies import RetryWithBackoffFallbackStrategy

        retry_strategy = RetryWithBackoffFallbackStrategy(
            self.config, max_retries=2, base_delay=2.0
        )
        self.add_strategy(retry_strategy)


class SecretFallbackHandler(ServiceFallbackHandler):
    """Secret manager-specific fallback handler."""

    def __init__(self, config: FallbackConfig):
        """Initialize secret fallback handler.

        Args:
            config: Fallback configuration
        """
        super().__init__("secret", config, None)

        # Add secret-specific strategies
        self._add_secret_strategies()

    def _add_secret_strategies(self) -> None:
        """Add secret-specific fallback strategies."""
        # Add cached secret strategy
        from .strategies import CachedFallbackStrategy

        cache_strategy = CachedFallbackStrategy(self.config)
        self.add_strategy(cache_strategy)

        # Add retry with backoff strategy
        from .strategies import RetryWithBackoffFallbackStrategy

        retry_strategy = RetryWithBackoffFallbackStrategy(
            self.config, max_retries=1, base_delay=1.0
        )
        self.add_strategy(retry_strategy)
