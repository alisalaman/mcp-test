"""Health checker implementation for external services."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    service_name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
            "error": self.error,
        }


class HealthChecker(ABC):
    """Abstract base class for health checkers."""

    def __init__(self, service_name: str, timeout: float = 5.0):
        """Initialize health checker.

        Args:
            service_name: Name of the service
            timeout: Timeout for health checks in seconds
        """
        self.service_name = service_name
        self.timeout = timeout

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            Health check result
        """
        pass

    async def run_check(self) -> HealthCheckResult:
        """Run health check with timeout.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(self.check_health(), timeout=self.timeout)
            return result
        except TimeoutError:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                timestamp=datetime.now(UTC),
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                error="Timeout",
            )
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(UTC),
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                error=str(e),
            )


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database services."""

    def __init__(
        self,
        service_name: str,
        connection_func: Callable[[], Any],
        timeout: float = 5.0,
    ):
        """Initialize database health checker.

        Args:
            service_name: Name of the database service
            connection_func: Function to get database connection
            timeout: Timeout for health checks
        """
        super().__init__(service_name, timeout)
        self.connection_func = connection_func

    async def check_health(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.time()

        try:
            # Get connection and run a simple query
            connection = await self.connection_func()

            # Run a simple health check query
            if hasattr(connection, "execute"):
                await connection.execute("SELECT 1")
            elif hasattr(connection, "fetchval"):
                await connection.fetchval("SELECT 1")
            else:
                # For connection pools, try to get a connection
                async with connection.acquire() as conn:
                    await conn.execute("SELECT 1")

            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.HEALTHY,
                message="Database is healthy",
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
                details={
                    "query": "SELECT 1",
                    "connection_type": type(connection).__name__,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
                details={},
                error=str(e),
            )


class HTTPHealthChecker(HealthChecker):
    """Health checker for HTTP services."""

    def __init__(
        self,
        service_name: str,
        url: str,
        timeout: float = 5.0,
        expected_status: int = 200,
    ):
        """Initialize HTTP health checker.

        Args:
            service_name: Name of the service
            url: Health check URL
            timeout: Timeout for health checks
            expected_status: Expected HTTP status code
        """
        super().__init__(service_name, timeout)
        self.url = url
        self.expected_status = expected_status

    async def check_health(self) -> HealthCheckResult:
        """Check HTTP service health."""
        start_time = time.time()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.url)
                response_time = (time.time() - start_time) * 1000

                if response.status_code == self.expected_status:
                    return HealthCheckResult(
                        service_name=self.service_name,
                        status=HealthStatus.HEALTHY,
                        message=f"HTTP service is healthy (status: {response.status_code})",
                        timestamp=datetime.now(UTC),
                        response_time_ms=response_time,
                        details={
                            "url": self.url,
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                        },
                    )
                else:
                    return HealthCheckResult(
                        service_name=self.service_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"HTTP service returned unexpected status: {response.status_code}",
                        timestamp=datetime.now(UTC),
                        response_time_ms=response_time,
                        details={
                            "url": self.url,
                            "status_code": response.status_code,
                            "expected_status": self.expected_status,
                        },
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP health check failed: {str(e)}",
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
                details={"url": self.url},
                error=str(e),
            )


class RedisHealthChecker(HealthChecker):
    """Health checker for Redis services."""

    def __init__(self, service_name: str, redis_client: Any, timeout: float = 5.0):
        """Initialize Redis health checker.

        Args:
            service_name: Name of the Redis service
            redis_client: Redis client instance
            timeout: Timeout for health checks
        """
        super().__init__(service_name, timeout)
        self.redis_client = redis_client

    async def check_health(self) -> HealthCheckResult:
        """Check Redis health."""
        start_time = time.time()

        try:
            # Test Redis connection with ping
            result = await self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000

            if result:
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.HEALTHY,
                    message="Redis is healthy",
                    timestamp=datetime.now(UTC),
                    response_time_ms=response_time,
                    details={
                        "ping_result": result,
                        "redis_version": await self._get_redis_info(),
                    },
                )
            else:
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    timestamp=datetime.now(UTC),
                    response_time_ms=response_time,
                    details={},
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis health check failed: {str(e)}",
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
                details={},
                error=str(e),
            )

    async def _get_redis_info(self) -> dict[str, Any]:
        """Get Redis server information."""
        try:
            info = await self.redis_client.info()
            return {
                "version": info.get("redis_version", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
            }
        except Exception:
            return {}


class CustomHealthChecker(HealthChecker):
    """Custom health checker for user-defined checks."""

    def __init__(
        self, service_name: str, check_func: Callable[[], Any], timeout: float = 5.0
    ):
        """Initialize custom health checker.

        Args:
            service_name: Name of the service
            check_func: Custom health check function
            timeout: Timeout for health checks
        """
        super().__init__(service_name, timeout)
        self.check_func = check_func

    async def check_health(self) -> HealthCheckResult:
        """Run custom health check."""
        start_time = time.time()

        try:
            result = await self.check_func()
            response_time = (time.time() - start_time) * 1000

            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, dict):
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=(
                        HealthStatus.HEALTHY
                        if result.get("healthy", False)
                        else HealthStatus.UNHEALTHY
                    ),
                    message=result.get("message", "Custom health check completed"),
                    timestamp=datetime.now(UTC),
                    response_time_ms=response_time,
                    details=result.get("details", {}),
                )
            else:
                # Assume boolean result
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Custom health check completed",
                    timestamp=datetime.now(UTC),
                    response_time_ms=response_time,
                    details={},
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom health check failed: {str(e)}",
                timestamp=datetime.now(UTC),
                response_time_ms=response_time,
                details={},
                error=str(e),
            )
