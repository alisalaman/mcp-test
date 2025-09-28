"""Rate limiting storage implementations."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class RateLimitStorage(ABC):
    """Abstract base class for rate limit storage."""

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Get rate limit data for key.

        Args:
            key: Storage key

        Returns:
            Rate limit data or None
        """
        pass

    @abstractmethod
    async def set(self, key: str, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Set rate limit data for key.

        Args:
            key: Storage key
            data: Rate limit data
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete rate limit data for key.

        Args:
            key: Storage key

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        pass


class InMemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._data: dict[str, dict[str, Any]] = {}
        self._ttl: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.logger = logger.bind(storage="in_memory")

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get rate limit data for key."""
        async with self._lock:
            # Check TTL
            if key in self._ttl and time.time() > self._ttl[key]:
                del self._data[key]
                del self._ttl[key]
                return None

            return self._data.get(key)

    async def set(self, key: str, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Set rate limit data for key."""
        async with self._lock:
            try:
                self._data[key] = data.copy()
                if ttl:
                    self._ttl[key] = time.time() + ttl
                return True
            except Exception as e:
                self.logger.error(
                    "Failed to set rate limit data", key=key, error=str(e)
                )
                return False

    async def delete(self, key: str) -> bool:
        """Delete rate limit data for key."""
        async with self._lock:
            try:
                self._data.pop(key, None)
                self._ttl.pop(key, None)
                return True
            except Exception as e:
                self.logger.error(
                    "Failed to delete rate limit data", key=key, error=str(e)
                )
                return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            # Check TTL
            if key in self._ttl and time.time() > self._ttl[key]:
                del self._data[key]
                del self._ttl[key]
                return False

            return key in self._data

    async def cleanup_expired(self) -> int:
        """Clean up expired entries.

        Returns:
            Number of entries cleaned up
        """
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._ttl.items() if current_time > expiry
            ]

            for key in expired_keys:
                self._data.pop(key, None)
                self._ttl.pop(key, None)

            if expired_keys:
                self.logger.debug("Cleaned up expired entries", count=len(expired_keys))

            return len(expired_keys)


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage."""

    def __init__(self, redis_client: Any):
        """Initialize Redis storage.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.logger = logger.bind(storage="redis")

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get rate limit data for key."""
        try:
            data = await self.redis.get(key)
            if data:
                return dict(json.loads(data))
            return None
        except Exception as e:
            self.logger.error("Failed to get rate limit data", key=key, error=str(e))
            return None

    async def set(self, key: str, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Set rate limit data for key."""
        try:
            serialized = json.dumps(data)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            return True
        except Exception as e:
            self.logger.error("Failed to set rate limit data", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete rate limit data for key."""
        try:
            result = await self.redis.delete(key)
            return bool(result > 0)
        except Exception as e:
            self.logger.error("Failed to delete rate limit data", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            result = await self.redis.exists(key)
            return bool(result > 0)
        except Exception as e:
            self.logger.error("Failed to check key existence", key=key, error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Increment a counter for key.

        Args:
            key: Storage key
            amount: Amount to increment by
            ttl: Time to live in seconds

        Returns:
            New counter value
        """
        try:
            if ttl:
                # Use pipeline for atomic operation
                pipe = self.redis.pipeline()
                pipe.incrby(key, amount)
                pipe.expire(key, ttl)
                results = await pipe.execute()
                return int(results[0])
            else:
                return int(await self.redis.incrby(key, amount))
        except Exception as e:
            self.logger.error("Failed to increment counter", key=key, error=str(e))
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get TTL for key.

        Args:
            key: Storage key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        try:
            return int(await self.redis.ttl(key))
        except Exception as e:
            self.logger.error("Failed to get TTL", key=key, error=str(e))
            return -2
