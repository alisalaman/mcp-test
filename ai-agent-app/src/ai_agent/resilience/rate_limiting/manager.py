"""Rate limit manager for multiple services."""

from typing import Any

import structlog

from .limiter import RateLimitConfig, RateLimiter, RateLimitResult

logger = structlog.get_logger()


class RateLimitManager:
    """Manages rate limiting for multiple services."""

    def __init__(self) -> None:
        """Initialize rate limit manager."""
        self.limiters: dict[str, RateLimiter] = {}

    def add_limiter(self, service_name: str, config: RateLimitConfig) -> RateLimiter:
        """Add rate limiter for a service.

        Args:
            service_name: Name of the service
            config: Rate limiting configuration

        Returns:
            Rate limiter instance
        """
        limiter = self._create_limiter(config)
        self.limiters[service_name] = limiter

        logger.info(
            "Added rate limiter for service",
            service_name=service_name,
            strategy=config.strategy,
            requests_per_minute=config.requests_per_minute,
        )

        return limiter

    def get_limiter(self, service_name: str) -> RateLimiter | None:
        """Get rate limiter for a service.

        Args:
            service_name: Name of the service

        Returns:
            Rate limiter or None if not found
        """
        return self.limiters.get(service_name)

    def _create_limiter(self, config: RateLimitConfig) -> RateLimiter:
        """Create rate limiter based on configuration.

        Args:
            config: Rate limiting configuration

        Returns:
            Rate limiter instance
        """
        if config.strategy == "token_bucket":
            from .limiter import TokenBucketRateLimiter

            return TokenBucketRateLimiter(config)
        elif config.strategy == "sliding_window":
            from .limiter import SlidingWindowRateLimiter

            return SlidingWindowRateLimiter(config)
        elif config.strategy == "fixed_window":
            from .limiter import FixedWindowRateLimiter

            return FixedWindowRateLimiter(config)
        else:
            logger.warning(
                "Unknown rate limiting strategy, using token bucket",
                strategy=config.strategy,
            )
            from .limiter import TokenBucketRateLimiter

            return TokenBucketRateLimiter(config)

    async def is_allowed(self, service_name: str, key: str) -> RateLimitResult:
        """Check if request is allowed for a service.

        Args:
            service_name: Name of the service
            key: Unique key for rate limiting

        Returns:
            Rate limiting result
        """
        limiter = self.get_limiter(service_name)

        if not limiter:
            logger.warning(
                "No rate limiter found for service", service_name=service_name
            )
            # Allow request if no limiter configured
            return RateLimitResult(
                allowed=True, remaining=999999, reset_time=0, limit=999999, used=0
            )

        return await limiter.is_allowed(key)

    async def consume(
        self, service_name: str, key: str, tokens: int = 1
    ) -> RateLimitResult:
        """Consume tokens for a service request.

        Args:
            service_name: Name of the service
            key: Unique key for rate limiting
            tokens: Number of tokens to consume

        Returns:
            Rate limiting result
        """
        limiter = self.get_limiter(service_name)

        if not limiter:
            logger.warning(
                "No rate limiter found for service", service_name=service_name
            )
            # Allow request if no limiter configured
            return RateLimitResult(
                allowed=True, remaining=999999, reset_time=0, limit=999999, used=tokens
            )

        return await limiter.consume(key, tokens)

    async def get_usage(self, service_name: str, key: str) -> dict[str, Any]:
        """Get current usage for a service and key.

        Args:
            service_name: Name of the service
            key: Unique key for rate limiting

        Returns:
            Usage information
        """
        limiter = self.get_limiter(service_name)

        if not limiter:
            return {
                "service_name": service_name,
                "key": key,
                "error": "No rate limiter configured",
            }

        usage = await limiter.get_usage(key)
        usage["service_name"] = service_name
        usage["key"] = key

        return usage

    async def reset(self, service_name: str, key: str) -> None:
        """Reset rate limit for a service and key.

        Args:
            service_name: Name of the service
            key: Unique key for rate limiting
        """
        limiter = self.get_limiter(service_name)

        if not limiter:
            logger.warning(
                "No rate limiter found for service", service_name=service_name
            )
            return

        await limiter.reset(key)
        logger.info("Reset rate limit", service_name=service_name, key=key)

    def remove_limiter(self, service_name: str) -> None:
        """Remove rate limiter for a service.

        Args:
            service_name: Name of the service
        """
        if service_name in self.limiters:
            del self.limiters[service_name]
            logger.info("Removed rate limiter", service_name=service_name)
        else:
            logger.warning("Rate limiter not found", service_name=service_name)

    def get_all_limiters(self) -> dict[str, RateLimiter]:
        """Get all rate limiters.

        Returns:
            Dictionary of service names to rate limiters
        """
        return self.limiters.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get rate limit manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_limiters": len(self.limiters),
            "services": list(self.limiters.keys()),
            "limiter_types": {
                service: type(limiter).__name__
                for service, limiter in self.limiters.items()
            },
        }

    def clear_all_limiters(self) -> None:
        """Clear all rate limiters."""
        self.limiters.clear()
        logger.info("Cleared all rate limiters")
