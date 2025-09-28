"""Health monitors for various system components."""

import asyncio
import time
from typing import Any
from collections.abc import Callable, Awaitable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    is_healthy: bool
    message: str
    details: dict[str, Any] | None = None
    response_time_ms: float | None = None
    timestamp: float | None = None


class HealthMonitor(ABC):
    """Base class for health monitors."""

    def __init__(self, name: str, timeout: float = 5.0):
        """Initialize health monitor.

        Args:
            name: Monitor name
            timeout: Timeout in seconds
        """
        self.name = name
        self.timeout = timeout
        self._last_check: HealthCheckResult | None = None
        self._check_count = 0
        self._failure_count = 0

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Check the health of the monitored component.

        Returns:
            Health check result
        """
        pass

    async def run_check(self) -> HealthCheckResult:
        """Run health check with timeout and error handling.

        Returns:
            Health check result
        """
        start_time = time.time()
        self._check_count += 1

        try:
            result = await asyncio.wait_for(self.check_health(), timeout=self.timeout)
            result.response_time_ms = (time.time() - start_time) * 1000
            result.timestamp = time.time()

            if not result.is_healthy:
                self._failure_count += 1

            self._last_check = result
            return result

        except TimeoutError:
            self._failure_count += 1
            result = HealthCheckResult(
                is_healthy=False,
                message=f"Health check timed out after {self.timeout}s",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
            )
            self._last_check = result
            logger.warning(f"Health check timeout for {self.name}")
            return result

        except Exception as e:
            self._failure_count += 1
            result = HealthCheckResult(
                is_healthy=False,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
            )
            self._last_check = result
            logger.error(f"Health check error for {self.name}: {e}")
            return result

    def get_last_check(self) -> HealthCheckResult | None:
        """Get the last health check result.

        Returns:
            Last health check result or None
        """
        return self._last_check

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics.

        Returns:
            Monitor statistics
        """
        return {
            "name": self.name,
            "check_count": self._check_count,
            "failure_count": self._failure_count,
            "success_rate": (self._check_count - self._failure_count)
            / max(self._check_count, 1),
            "last_check": self._last_check.timestamp if self._last_check else None,
        }


class DatabaseHealthMonitor(HealthMonitor):
    """Health monitor for database connections."""

    def __init__(
        self,
        connection_func: Callable[[], Awaitable[Any]],
        timeout: float = 5.0,
        query: str = "SELECT 1",
    ):
        """Initialize database health monitor.

        Args:
            connection_func: Function to get database connection
            timeout: Timeout in seconds
            query: Test query to run
        """
        super().__init__("database", timeout)
        self.connection_func = connection_func
        self.query = query

    async def check_health(self) -> HealthCheckResult:
        """Check database health.

        Returns:
            Health check result
        """
        try:
            connection = await self.connection_func()

            # Execute test query
            if hasattr(connection, "execute"):
                # SQLAlchemy-style connection
                await connection.execute(self.query)
            elif hasattr(connection, "fetchone"):
                # AsyncPG-style connection
                await connection.fetchone(self.query)
            else:
                # Generic connection
                await connection.execute(self.query)

            return HealthCheckResult(
                is_healthy=True,
                message="Database connection is healthy",
                details={"query": self.query},
            )

        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)},
            )


class RedisHealthMonitor(HealthMonitor):
    """Health monitor for Redis connections."""

    def __init__(
        self,
        redis_client: Any,
        timeout: float = 5.0,
        test_key: str = "health_check",
    ):
        """Initialize Redis health monitor.

        Args:
            redis_client: Redis client instance
            timeout: Timeout in seconds
            test_key: Test key to use for health check
        """
        super().__init__("redis", timeout)
        self.redis_client = redis_client
        self.test_key = test_key

    async def check_health(self) -> HealthCheckResult:
        """Check Redis health.

        Returns:
            Health check result
        """
        try:
            # Test ping
            await self.redis_client.ping()

            # Test set/get
            test_value = f"health_check_{int(time.time())}"
            await self.redis_client.set(self.test_key, test_value, ex=60)
            retrieved_value = await self.redis_client.get(self.test_key)

            if retrieved_value != test_value:
                return HealthCheckResult(
                    is_healthy=False,
                    message="Redis set/get test failed",
                    details={"expected": test_value, "actual": retrieved_value},
                )

            # Clean up test key
            await self.redis_client.delete(self.test_key)

            return HealthCheckResult(
                is_healthy=True,
                message="Redis connection is healthy",
                details={"test_key": self.test_key},
            )

        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                message=f"Redis connection failed: {str(e)}",
                details={"error": str(e)},
            )


class ExternalServiceHealthMonitor(HealthMonitor):
    """Health monitor for external services."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[bool]],
        timeout: float = 10.0,
    ):
        """Initialize external service health monitor.

        Args:
            name: Service name
            check_func: Function to check service health
            timeout: Timeout in seconds
        """
        super().__init__(name, timeout)
        self.check_func = check_func

    async def check_health(self) -> HealthCheckResult:
        """Check external service health.

        Returns:
            Health check result
        """
        try:
            is_healthy = await self.check_func()

            return HealthCheckResult(
                is_healthy=is_healthy,
                message=f"External service {self.name} is {'healthy' if is_healthy else 'unhealthy'}",
                details={"service": self.name},
            )

        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                message=f"External service {self.name} check failed: {str(e)}",
                details={"service": self.name, "error": str(e)},
            )


class CompositeHealthMonitor(HealthMonitor):
    """Health monitor that combines multiple monitors."""

    def __init__(self, name: str, monitors: list[HealthMonitor], timeout: float = 10.0):
        """Initialize composite health monitor.

        Args:
            name: Monitor name
            monitors: List of health monitors
            timeout: Timeout in seconds
        """
        super().__init__(name, timeout)
        self.monitors = monitors

    async def check_health(self) -> HealthCheckResult:
        """Check health of all monitors.

        Returns:
            Health check result
        """
        results = []
        all_healthy = True

        for monitor in self.monitors:
            result = await monitor.run_check()
            results.append(
                {
                    "monitor": monitor.name,
                    "healthy": result.is_healthy,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                }
            )

            if not result.is_healthy:
                all_healthy = False

        return HealthCheckResult(
            is_healthy=all_healthy,
            message=f"Composite health check {'passed' if all_healthy else 'failed'}",
            details={"monitors": results},
        )

    def get_monitor_stats(self) -> dict[str, Any]:
        """Get statistics for all monitors.

        Returns:
            Monitor statistics
        """
        return {
            "composite": self.get_stats(),
            "monitors": [monitor.get_stats() for monitor in self.monitors],
        }
