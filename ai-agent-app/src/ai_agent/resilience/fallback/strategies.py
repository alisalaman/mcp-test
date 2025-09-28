"""Fallback strategies for graceful degradation."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies."""

    enabled: bool = True
    cache_ttl: float = 300.0  # 5 minutes
    max_cache_size: int = 1000
    retry_after: float = 60.0  # seconds
    fallback_timeout: float = 5.0  # seconds


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""

    def __init__(self, config: FallbackConfig):
        """Initialize fallback strategy.

        Args:
            config: Fallback configuration
        """
        self.config = config

    @abstractmethod
    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Execute fallback strategy.

        Args:
            service_name: Name of the service
            original_func: Original function that failed
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Fallback result
        """
        pass

    def should_use_fallback(self, error: Exception) -> bool:
        """Determine if fallback should be used for this error.

        Args:
            error: Exception that occurred

        Returns:
            True if fallback should be used
        """
        return True


class DefaultValueFallbackStrategy(FallbackStrategy):
    """Fallback strategy that returns a default value."""

    def __init__(self, config: FallbackConfig, default_value: Any = None):
        """Initialize default value fallback.

        Args:
            config: Fallback configuration
            default_value: Default value to return
        """
        super().__init__(config)
        self.default_value = default_value

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Return default value."""
        logger.warning(
            "Using default value fallback",
            service_name=service_name,
            default_value=self.default_value,
        )
        return self.default_value


class CachedFallbackStrategy(FallbackStrategy):
    """Fallback strategy that returns cached values."""

    def __init__(self, config: FallbackConfig):
        """Initialize cached fallback.

        Args:
            config: Fallback configuration
        """
        super().__init__(config)
        self._cache: dict[str, dict[str, Any]] = {}

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Return cached value if available."""
        cache_key = self._get_cache_key(service_name, original_func, *args, **kwargs)

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]

            # Check if cache is still valid
            if time.time() - cached_data["timestamp"] < self.config.cache_ttl:
                logger.info(
                    "Using cached fallback value",
                    service_name=service_name,
                    cache_age=time.time() - cached_data["timestamp"],
                )
                return cached_data["value"]
            else:
                # Remove expired cache entry
                del self._cache[cache_key]

        logger.warning(
            "No cached value available for fallback", service_name=service_name
        )
        return None

    def cache_result(  # type: ignore[no-untyped-def]
        self,
        service_name: str,
        original_func: Callable[..., Any],
        result: Any,
        *args,
        **kwargs,
    ) -> None:
        """Cache a successful result.

        Args:
            service_name: Name of the service
            original_func: Original function
            result: Result to cache
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        cache_key = self._get_cache_key(service_name, original_func, *args, **kwargs)

        # Clean up old entries if cache is full
        if len(self._cache) >= self.config.max_cache_size:
            self._cleanup_cache()

        self._cache[cache_key] = {"value": result, "timestamp": time.time()}

    def _get_cache_key(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> str:
        """Generate cache key for function call.

        Args:
            service_name: Name of the service
            original_func: Original function
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Cache key string
        """
        # Create a hash of the function and arguments
        import hashlib

        key_data = f"{service_name}:{original_func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cleanup_cache(self) -> None:
        """Remove oldest cache entries."""
        if not self._cache:
            return

        # Sort by timestamp and remove oldest 25%
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1]["timestamp"])

        entries_to_remove = len(sorted_entries) // 4
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[key]


class AlternativeServiceFallbackStrategy(FallbackStrategy):
    """Fallback strategy that calls an alternative service."""

    def __init__(
        self,
        config: FallbackConfig,
        alternative_func: Callable[..., Any],
        service_name: str = "alternative",
    ):
        """Initialize alternative service fallback.

        Args:
            config: Fallback configuration
            alternative_func: Alternative function to call
            service_name: Name of the alternative service
        """
        super().__init__(config)
        self.alternative_func = alternative_func
        self.service_name = service_name

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Call alternative service."""
        try:
            logger.info(
                "Using alternative service fallback",
                original_service=service_name,
                alternative_service=self.service_name,
            )

            result = await asyncio.wait_for(
                self.alternative_func(*args, **kwargs),
                timeout=self.config.fallback_timeout,
            )

            return result

        except Exception as e:
            logger.error(
                "Alternative service also failed",
                original_service=service_name,
                alternative_service=self.service_name,
                error=str(e),
            )
            raise


class RetryWithBackoffFallbackStrategy(FallbackStrategy):
    """Fallback strategy that retries with exponential backoff."""

    def __init__(
        self, config: FallbackConfig, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Initialize retry with backoff fallback.

        Args:
            config: Fallback configuration
            max_retries: Maximum number of retries
            base_delay: Base delay between retries
        """
        super().__init__(config)
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Retry with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Retrying with backoff fallback",
                    service_name=service_name,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

                result = await asyncio.wait_for(
                    original_func(*args, **kwargs), timeout=self.config.fallback_timeout
                )

                return result

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        "All retry attempts failed",
                        service_name=service_name,
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = self.base_delay * (2**attempt)
                await asyncio.sleep(delay)


class CircuitBreakerFallbackStrategy(FallbackStrategy):
    """Fallback strategy that checks circuit breaker state."""

    def __init__(
        self,
        config: FallbackConfig,
        circuit_breaker_manager: Any,
        fallback_strategy: FallbackStrategy,
    ) -> None:
        """Initialize circuit breaker fallback.

        Args:
            config: Fallback configuration
            circuit_breaker_manager: Circuit breaker manager instance
            fallback_strategy: Fallback strategy to use when circuit is open
        """
        super().__init__(config)
        self.circuit_breaker_manager = circuit_breaker_manager
        self.fallback_strategy = fallback_strategy

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Execute fallback based on circuit breaker state."""
        try:
            breaker = self.circuit_breaker_manager.get_breaker(service_name)

            if breaker.state.value == "open":
                logger.warning(
                    "Circuit breaker is open, using fallback strategy",
                    service_name=service_name,
                )
                return await self.fallback_strategy.execute_fallback(
                    service_name, original_func, *args, **kwargs
                )
            else:
                # Circuit is closed or half-open, try original function
                return await original_func(*args, **kwargs)

        except Exception as e:
            logger.error(
                "Circuit breaker fallback failed",
                service_name=service_name,
                error=str(e),
            )
            raise
