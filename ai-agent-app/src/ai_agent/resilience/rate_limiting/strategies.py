"""Rate limiting strategies implementation."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size_seconds: int = 60
    max_requests: int = 100


class RateLimitStrategy(ABC):
    """Abstract base class for rate limiting strategies."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limit strategy.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.logger = logger.bind(strategy=type(self).__name__)

    @abstractmethod
    async def is_allowed(self, key: str) -> tuple[bool, dict[str, Any] | None]:
        """Check if request is allowed.

        Args:
            key: Unique identifier for the rate limit

        Returns:
            Tuple of (is_allowed, metadata)
        """
        pass

    @abstractmethod
    async def consume(
        self, key: str, tokens: int = 1
    ) -> tuple[bool, dict[str, Any] | None]:
        """Consume tokens from the rate limit.

        Args:
            key: Unique identifier for the rate limit
            tokens: Number of tokens to consume

        Returns:
            Tuple of (success, metadata)
        """
        pass


class TokenBucketStrategy(RateLimitStrategy):
    """Token bucket rate limiting strategy."""

    def __init__(self, config: RateLimitConfig):
        """Initialize token bucket strategy."""
        super().__init__(config)
        self.buckets: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, dict[str, Any] | None]:
        """Check if request is allowed using token bucket."""
        async with self._lock:
            bucket = self._get_bucket(key)
            current_time = time.time()

            # Refill tokens based on elapsed time
            elapsed = current_time - bucket["last_refill"]
            tokens_to_add = elapsed * self.config.requests_per_second
            bucket["tokens"] = min(
                self.config.burst_size, bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = current_time

            if bucket["tokens"] >= 1:
                return True, {
                    "tokens_remaining": bucket["tokens"],
                    "burst_size": self.config.burst_size,
                }
            else:
                return False, {
                    "tokens_remaining": bucket["tokens"],
                    "retry_after": 1.0 / self.config.requests_per_second,
                }

    async def consume(
        self, key: str, tokens: int = 1
    ) -> tuple[bool, dict[str, Any] | None]:
        """Consume tokens from the bucket."""
        async with self._lock:
            bucket = self._get_bucket(key)
            current_time = time.time()

            # Refill tokens
            elapsed = current_time - bucket["last_refill"]
            tokens_to_add = elapsed * self.config.requests_per_second
            bucket["tokens"] = min(
                self.config.burst_size, bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = current_time

            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True, {
                    "tokens_remaining": bucket["tokens"],
                    "tokens_consumed": tokens,
                }
            else:
                return False, {
                    "tokens_remaining": bucket["tokens"],
                    "retry_after": tokens / self.config.requests_per_second,
                }

    def _get_bucket(self, key: str) -> dict[str, Any]:
        """Get or create bucket for key."""
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": self.config.burst_size,
                "last_refill": time.time(),
            }
        return self.buckets[key]


class SlidingWindowStrategy(RateLimitStrategy):
    """Sliding window rate limiting strategy."""

    def __init__(self, config: RateLimitConfig):
        """Initialize sliding window strategy."""
        super().__init__(config)
        self.windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, dict[str, Any] | None]:
        """Check if request is allowed using sliding window."""
        async with self._lock:
            current_time = time.time()
            window = self._get_window(key)

            # Remove old requests outside the window
            cutoff_time = current_time - self.config.window_size_seconds
            window[:] = [req_time for req_time in window if req_time > cutoff_time]

            if len(window) < self.config.max_requests:
                return True, {
                    "requests_in_window": len(window),
                    "max_requests": self.config.max_requests,
                }
            else:
                oldest_request = min(window)
                retry_after = (
                    oldest_request + self.config.window_size_seconds - current_time
                )
                return False, {
                    "requests_in_window": len(window),
                    "retry_after": max(0, retry_after),
                }

    async def consume(
        self, key: str, tokens: int = 1
    ) -> tuple[bool, dict[str, Any] | None]:
        """Consume tokens from the sliding window."""
        async with self._lock:
            current_time = time.time()
            window = self._get_window(key)

            # Remove old requests
            cutoff_time = current_time - self.config.window_size_seconds
            window[:] = [req_time for req_time in window if req_time > cutoff_time]

            if len(window) + tokens <= self.config.max_requests:
                # Add new requests
                for _ in range(tokens):
                    window.append(current_time)
                return True, {
                    "requests_in_window": len(window),
                    "tokens_consumed": tokens,
                }
            else:
                oldest_request = min(window) if window else current_time
                retry_after = (
                    oldest_request + self.config.window_size_seconds - current_time
                )
                return False, {
                    "requests_in_window": len(window),
                    "retry_after": max(0, retry_after),
                }

    def _get_window(self, key: str) -> list[float]:
        """Get or create window for key."""
        if key not in self.windows:
            self.windows[key] = []
        return self.windows[key]


class FixedWindowStrategy(RateLimitStrategy):
    """Fixed window rate limiting strategy."""

    def __init__(self, config: RateLimitConfig):
        """Initialize fixed window strategy."""
        super().__init__(config)
        self.windows: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> tuple[bool, dict[str, Any] | None]:
        """Check if request is allowed using fixed window."""
        async with self._lock:
            current_time = time.time()
            window = self._get_window(key)

            # Check if we need to reset the window
            if current_time - window["start_time"] >= self.config.window_size_seconds:
                window["count"] = 0
                window["start_time"] = current_time

            if window["count"] < self.config.max_requests:
                return True, {
                    "requests_in_window": window["count"],
                    "max_requests": self.config.max_requests,
                    "window_reset_in": self.config.window_size_seconds
                    - (current_time - window["start_time"]),
                }
            else:
                window_reset_in = self.config.window_size_seconds - (
                    current_time - window["start_time"]
                )
                return False, {
                    "requests_in_window": window["count"],
                    "retry_after": window_reset_in,
                }

    async def consume(
        self, key: str, tokens: int = 1
    ) -> tuple[bool, dict[str, Any] | None]:
        """Consume tokens from the fixed window."""
        async with self._lock:
            current_time = time.time()
            window = self._get_window(key)

            # Check if we need to reset the window
            if current_time - window["start_time"] >= self.config.window_size_seconds:
                window["count"] = 0
                window["start_time"] = current_time

            if window["count"] + tokens <= self.config.max_requests:
                window["count"] += tokens
                return True, {
                    "requests_in_window": window["count"],
                    "tokens_consumed": tokens,
                }
            else:
                window_reset_in = self.config.window_size_seconds - (
                    current_time - window["start_time"]
                )
                return False, {
                    "requests_in_window": window["count"],
                    "retry_after": window_reset_in,
                }

    def _get_window(self, key: str) -> dict[str, Any]:
        """Get or create window for key."""
        if key not in self.windows:
            self.windows[key] = {"count": 0, "start_time": time.time()}
        return self.windows[key]
