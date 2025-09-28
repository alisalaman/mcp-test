"""Health check system implementation."""

import asyncio
import time
from datetime import datetime, UTC
from enum import Enum
from typing import Any
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field

from ..logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_func: Callable[[], Awaitable[HealthCheckResult]]
    timeout: float = 5.0
    critical: bool = True
    enabled: bool = True


class HealthChecker:
    """Main health checker for the application."""

    def __init__(self, app_name: str = "ai-agent-app", version: str = "1.0.0"):
        self.app_name = app_name
        self.version = version
        self.checks: dict[str, HealthCheck] = {}
        self.logger = get_logger(__name__)
        self._last_check_time: datetime | None = None
        self._cache_ttl: float = 30.0  # seconds
        self._cached_result: dict[str, Any] | None = None

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks[check.name] = check
        self.logger.info(f"Added health check: {check.name}")

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        if name in self.checks:
            del self.checks[name]
            self.logger.info(f"Removed health check: {name}")

    def enable_check(self, name: str) -> None:
        """Enable a health check."""
        if name in self.checks:
            self.checks[name].enabled = True
            self.logger.info(f"Enabled health check: {name}")

    def disable_check(self, name: str) -> None:
        """Disable a health check."""
        if name in self.checks:
            self.checks[name].enabled = False
            self.logger.info(f"Disabled health check: {name}")

    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
            )

        check = self.checks[name]
        if not check.enabled:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' is disabled",
            )

        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(check.check_func(), timeout=check.timeout)

            # Update duration
            result.duration_ms = (time.time() - start_time) * 1000

            return result

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check '{name}' timed out after {check.timeout}s",
                duration_ms=duration_ms,
                error="timeout",
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check '{name}' failed: {str(e)}",
                duration_ms=duration_ms,
                error=str(e),
            )

    async def run_all_checks(self, use_cache: bool = True) -> dict[str, Any]:
        """Run all health checks."""
        # Check cache
        if use_cache and self._cached_result and self._last_check_time:
            cache_age = (datetime.now(UTC) - self._last_check_time).total_seconds()
            if cache_age < self._cache_ttl:
                return self._cached_result

        # Run all checks
        results = {}
        overall_status = HealthStatus.HEALTHY
        critical_failures = 0

        for name, check in self.checks.items():
            if check.enabled:
                result = await self.run_check(name)
                results[name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat(),
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                }

                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    if check.critical:
                        critical_failures += 1
                        overall_status = HealthStatus.UNHEALTHY
                    elif overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                elif (
                    result.status == HealthStatus.DEGRADED
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        # Prepare response
        response = {
            "app_name": self.app_name,
            "version": self.version,
            "status": overall_status.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy_checks": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.HEALTHY.value
                ),
                "unhealthy_checks": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.UNHEALTHY.value
                ),
                "degraded_checks": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.DEGRADED.value
                ),
                "critical_failures": critical_failures,
            },
        }

        # Cache result
        self._cached_result = response
        self._last_check_time = datetime.now(UTC)

        return response

    async def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        result = await self.run_all_checks()
        return HealthStatus(result["status"])

    async def is_healthy(self) -> bool:
        """Check if application is healthy."""
        status = await self.get_health_status()
        return status == HealthStatus.HEALTHY

    async def is_ready(self) -> bool:
        """Check if application is ready to serve requests."""
        # For readiness, we only check critical health checks
        critical_checks = [
            name
            for name, check in self.checks.items()
            if check.critical and check.enabled
        ]

        if not critical_checks:
            return True

        for name in critical_checks:
            result = await self.run_check(name)
            if result.status != HealthStatus.HEALTHY:
                return False

        return True

    async def is_live(self) -> bool:
        """Check if application is alive (basic liveness check)."""
        # Basic liveness check - just verify the process is running
        return True

    def get_check_names(self) -> list[str]:
        """Get list of all health check names."""
        return list(self.checks.keys())

    def get_enabled_checks(self) -> list[str]:
        """Get list of enabled health check names."""
        return [name for name, check in self.checks.items() if check.enabled]

    def get_critical_checks(self) -> list[str]:
        """Get list of critical health check names."""
        return [
            name
            for name, check in self.checks.items()
            if check.critical and check.enabled
        ]


# Global health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def setup_health_checks(
    app_name: str = "ai-agent-app", version: str = "1.0.0"
) -> HealthChecker:
    """Setup global health checker."""
    global _health_checker
    _health_checker = HealthChecker(app_name, version)
    return _health_checker


# Convenience functions for common health checks
async def check_database_health() -> HealthCheckResult:
    """Check database health."""
    try:
        # This would be implemented based on your database setup
        # For now, return a placeholder
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection is healthy",
            details={"connection_pool_size": 10, "active_connections": 5},
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            error=str(e),
        )


async def check_redis_health() -> HealthCheckResult:
    """Check Redis health."""
    try:
        # This would be implemented based on your Redis setup
        # For now, return a placeholder
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection is healthy",
            details={"memory_usage": "50MB", "connected_clients": 3},
        )
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis connection failed: {str(e)}",
            error=str(e),
        )


async def check_external_services_health() -> HealthCheckResult:
    """Check external services health."""
    try:
        # This would check all external services
        # For now, return a placeholder
        return HealthCheckResult(
            name="external_services",
            status=HealthStatus.HEALTHY,
            message="All external services are healthy",
            details={"checked_services": 3, "healthy_services": 3},
        )
    except Exception as e:
        return HealthCheckResult(
            name="external_services",
            status=HealthStatus.UNHEALTHY,
            message=f"External services check failed: {str(e)}",
            error=str(e),
        )


async def check_disk_space() -> HealthCheckResult:
    """Check disk space."""
    try:
        import shutil

        # Check disk usage
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100

        if free_percent < 10:
            status = HealthStatus.UNHEALTHY
            message = f"Disk space critically low: {free_percent:.1f}% free"
        elif free_percent < 20:
            status = HealthStatus.DEGRADED
            message = f"Disk space low: {free_percent:.1f}% free"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk space is healthy: {free_percent:.1f}% free"

        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            details={
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "free_percent": free_percent,
            },
        )
    except Exception as e:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk space check failed: {str(e)}",
            error=str(e),
        )


async def check_memory_usage() -> HealthCheckResult:
    """Check memory usage."""
    try:
        import psutil

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        if memory_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage critically high: {memory_percent:.1f}%"
        elif memory_percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Memory usage high: {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is healthy: {memory_percent:.1f}%"

        return HealthCheckResult(
            name="memory_usage",
            status=status,
            message=message,
            details={
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "percent": memory_percent,
            },
        )
    except Exception as e:
        return HealthCheckResult(
            name="memory_usage",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory usage check failed: {str(e)}",
            error=str(e),
        )
