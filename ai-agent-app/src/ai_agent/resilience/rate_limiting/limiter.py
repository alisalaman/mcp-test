"""Rate limiter implementation for external service protection."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    strategy: str = "token_bucket"  # token_bucket, sliding_window, fixed_window


@dataclass
class RateLimitResult:
    """Rate limiting result."""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: float | None = None
    limit: int = 0
    used: int = 0


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config

    @abstractmethod
    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            key: Unique key for rate limiting (e.g., user ID, IP address)

        Returns:
            Rate limiting result
        """
        pass

    @abstractmethod
    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens for a request.

        Args:
            key: Unique key for rate limiting
            tokens: Number of tokens to consume

        Returns:
            Rate limiting result
        """
        pass

    @abstractmethod
    async def get_usage(self, key: str) -> dict[str, Any]:
        """Get current usage for a key.

        Args:
            key: Unique key for rate limiting

        Returns:
            Usage information
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Unique key for rate limiting
        """
        pass


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter implementation."""

    def __init__(self, config: RateLimitConfig):
        """Initialize token bucket rate limiter.

        Args:
            config: Rate limiting configuration
        """
        super().__init__(config)
        self.buckets: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed using token bucket."""
        async with self._lock:
            bucket = await self._get_bucket(key)
            current_time = time.time()

            # Refill tokens based on time elapsed
            await self._refill_tokens(bucket, current_time)

            if bucket["tokens"] >= 1:
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"] - 1),
                    reset_time=bucket["last_refill"] + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=int(self.config.requests_per_minute - bucket["tokens"] + 1),
                )
            else:
                # Calculate retry after time
                retry_after = (1.0 / self.config.requests_per_minute) * 60

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=bucket["last_refill"] + self.config.window_size,
                    retry_after=retry_after,
                    limit=self.config.requests_per_minute,
                    used=self.config.requests_per_minute,
                )

    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens for a request."""
        async with self._lock:
            bucket = await self._get_bucket(key)
            current_time = time.time()

            # Refill tokens based on time elapsed
            await self._refill_tokens(bucket, current_time)

            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                bucket["last_refill"] = current_time

                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_time=bucket["last_refill"] + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=int(self.config.requests_per_minute - bucket["tokens"]),
                )
            else:
                # Calculate retry after time
                retry_after = (tokens / self.config.requests_per_minute) * 60

                return RateLimitResult(
                    allowed=False,
                    remaining=int(bucket["tokens"]),
                    reset_time=bucket["last_refill"] + self.config.window_size,
                    retry_after=retry_after,
                    limit=self.config.requests_per_minute,
                    used=int(self.config.requests_per_minute - bucket["tokens"]),
                )

    async def get_usage(self, key: str) -> dict[str, Any]:
        """Get current usage for a key."""
        async with self._lock:
            bucket = await self._get_bucket(key)
            current_time = time.time()

            # Refill tokens based on time elapsed
            await self._refill_tokens(bucket, current_time)

            return {
                "tokens": int(bucket["tokens"]),
                "capacity": self.config.requests_per_minute,
                "last_refill": bucket["last_refill"],
                "refill_rate": self.config.requests_per_minute / 60.0,
            }

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            if key in self.buckets:
                del self.buckets[key]

    async def _get_bucket(self, key: str) -> dict[str, Any]:
        """Get or create token bucket for key."""
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": self.config.requests_per_minute,
                "last_refill": time.time(),
            }
        return self.buckets[key]

    async def _refill_tokens(self, bucket: dict[str, Any], current_time: float) -> None:
        """Refill tokens based on time elapsed."""
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = (time_elapsed / 60.0) * self.config.requests_per_minute

        bucket["tokens"] = min(
            self.config.requests_per_minute, bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = current_time


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter implementation."""

    def __init__(self, config: RateLimitConfig):
        """Initialize sliding window rate limiter.

        Args:
            config: Rate limiting configuration
        """
        super().__init__(config)
        self.windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed using sliding window."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key)

            # Remove old requests outside the window
            cutoff_time = current_time - self.config.window_size
            window[:] = [req_time for req_time in window if req_time > cutoff_time]

            if len(window) < self.config.requests_per_minute:
                return RateLimitResult(
                    allowed=True,
                    remaining=self.config.requests_per_minute - len(window) - 1,
                    reset_time=current_time + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=len(window) + 1,
                )
            else:
                # Calculate retry after time
                oldest_request = min(window)
                retry_after = oldest_request + self.config.window_size - current_time

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=oldest_request + self.config.window_size,
                    retry_after=max(0, retry_after),
                    limit=self.config.requests_per_minute,
                    used=len(window),
                )

    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens for a request."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key)

            # Remove old requests outside the window
            cutoff_time = current_time - self.config.window_size
            window[:] = [req_time for req_time in window if req_time > cutoff_time]

            if len(window) + tokens <= self.config.requests_per_minute:
                # Add new requests
                for _ in range(tokens):
                    window.append(current_time)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.config.requests_per_minute - len(window),
                    reset_time=current_time + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=len(window),
                )
            else:
                # Calculate retry after time
                oldest_request = min(window) if window else current_time
                retry_after = oldest_request + self.config.window_size - current_time

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=oldest_request + self.config.window_size,
                    retry_after=max(0, retry_after),
                    limit=self.config.requests_per_minute,
                    used=len(window),
                )

    async def get_usage(self, key: str) -> dict[str, Any]:
        """Get current usage for a key."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key)

            # Remove old requests outside the window
            cutoff_time = current_time - self.config.window_size
            window[:] = [req_time for req_time in window if req_time > cutoff_time]

            return {
                "requests": len(window),
                "limit": self.config.requests_per_minute,
                "window_size": self.config.window_size,
                "oldest_request": min(window) if window else None,
            }

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            if key in self.windows:
                del self.windows[key]

    async def _get_window(self, key: str) -> list[float]:
        """Get or create sliding window for key."""
        if key not in self.windows:
            self.windows[key] = []
        return self.windows[key]


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter implementation."""

    def __init__(self, config: RateLimitConfig):
        """Initialize fixed window rate limiter.

        Args:
            config: Rate limiting configuration
        """
        super().__init__(config)
        self.windows: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed using fixed window."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key, current_time)

            if window["count"] < self.config.requests_per_minute:
                return RateLimitResult(
                    allowed=True,
                    remaining=self.config.requests_per_minute - window["count"] - 1,
                    reset_time=window["start_time"] + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=window["count"] + 1,
                )
            else:
                # Calculate retry after time
                retry_after = (
                    window["start_time"] + self.config.window_size - current_time
                )

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=window["start_time"] + self.config.window_size,
                    retry_after=max(0, retry_after),
                    limit=self.config.requests_per_minute,
                    used=window["count"],
                )

    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens for a request."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key, current_time)

            if window["count"] + tokens <= self.config.requests_per_minute:
                window["count"] += tokens

                return RateLimitResult(
                    allowed=True,
                    remaining=self.config.requests_per_minute - window["count"],
                    reset_time=window["start_time"] + self.config.window_size,
                    limit=self.config.requests_per_minute,
                    used=window["count"],
                )
            else:
                # Calculate retry after time
                retry_after = (
                    window["start_time"] + self.config.window_size - current_time
                )

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=window["start_time"] + self.config.window_size,
                    retry_after=max(0, retry_after),
                    limit=self.config.requests_per_minute,
                    used=window["count"],
                )

    async def get_usage(self, key: str) -> dict[str, Any]:
        """Get current usage for a key."""
        async with self._lock:
            current_time = time.time()
            window = await self._get_window(key, current_time)

            return {
                "count": window["count"],
                "limit": self.config.requests_per_minute,
                "window_start": window["start_time"],
                "window_end": window["start_time"] + self.config.window_size,
            }

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            if key in self.windows:
                del self.windows[key]

    async def _get_window(self, key: str, current_time: float) -> dict[str, Any]:
        """Get or create fixed window for key."""
        window_start = (
            int(current_time // self.config.window_size) * self.config.window_size
        )

        if key not in self.windows or self.windows[key]["start_time"] != window_start:
            self.windows[key] = {"count": 0, "start_time": window_start}

        return self.windows[key]
