"""Health monitoring system for external services."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from .checker import HealthChecker, HealthCheckResult, HealthStatus

logger = structlog.get_logger()


@dataclass
class HealthMonitorConfig:
    """Configuration for health monitoring."""

    check_interval: float = 30.0  # seconds
    timeout: float = 5.0  # seconds
    max_consecutive_failures: int = 3
    degraded_threshold: float = 0.8  # 80% success rate
    unhealthy_threshold: float = 0.5  # 50% success rate
    enable_notifications: bool = True
    notification_callback: Callable[[str, HealthCheckResult], None] | None = None


@dataclass
class ServiceHealthMetrics:
    """Health metrics for a service."""

    service_name: str
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    consecutive_failures: int = 0
    last_check_time: datetime | None = None
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    average_response_time: float = 0.0
    health_history: list[HealthCheckResult] = field(default_factory=list)
    max_history_size: int = 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_checks == 0:
            return 0.0
        return self.successful_checks / self.total_checks

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.consecutive_failures < 3 and self.success_rate > 0.5

    @property
    def is_degraded(self) -> bool:
        """Check if service is degraded."""
        return 0.5 < self.success_rate < 0.8

    def add_check_result(self, result: HealthCheckResult) -> None:
        """Add a health check result."""
        self.total_checks += 1
        self.last_check_time = result.timestamp

        if result.status == HealthStatus.HEALTHY:
            self.successful_checks += 1
            self.consecutive_failures = 0
            self.last_success_time = result.timestamp
        else:
            self.failed_checks += 1
            self.consecutive_failures += 1
            self.last_failure_time = result.timestamp

        # Update average response time
        if self.average_response_time == 0.0:
            self.average_response_time = result.response_time_ms
        else:
            self.average_response_time = (
                self.average_response_time * (self.total_checks - 1)
                + result.response_time_ms
            ) / self.total_checks

        # Add to history
        self.health_history.append(result)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        if self.total_checks == 0:
            return HealthStatus.UNKNOWN
        elif self.consecutive_failures >= 3:
            return HealthStatus.UNHEALTHY
        elif self.success_rate <= 0.5:
            return HealthStatus.UNHEALTHY
        elif self.success_rate < 0.8:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class ServiceHealthMonitor:
    """Monitors health of multiple services."""

    def __init__(self, config: HealthMonitorConfig):
        """Initialize health monitor.

        Args:
            config: Health monitoring configuration
        """
        self.config = config
        self.checkers: dict[str, HealthChecker] = {}
        self.metrics: dict[str, ServiceHealthMetrics] = {}
        self._monitoring_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def add_checker(self, service_name: str, checker: HealthChecker) -> None:
        """Add a health checker for a service.

        Args:
            service_name: Name of the service
            checker: Health checker instance
        """
        self.checkers[service_name] = checker
        self.metrics[service_name] = ServiceHealthMetrics(service_name)
        logger.info("Added health checker", service_name=service_name)

    def remove_checker(self, service_name: str) -> None:
        """Remove a health checker.

        Args:
            service_name: Name of the service
        """
        if service_name in self.checkers:
            del self.checkers[service_name]
            del self.metrics[service_name]
            logger.info("Removed health checker", service_name=service_name)

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._monitoring_task is not None:
            logger.warning("Health monitoring already started")
            return

        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring", interval=self.config.check_interval)

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._monitoring_task is None:
            logger.warning("Health monitoring not started")
            return

        self._stop_event.set()
        self._monitoring_task.cancel()

        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass

        self._monitoring_task = None
        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _run_health_checks(self) -> None:
        """Run health checks for all services."""
        if not self.checkers:
            return

        # Run health checks concurrently
        tasks = []
        for service_name, checker in self.checkers.items():
            task = asyncio.create_task(
                self._check_service_health(service_name, checker)
            )
            tasks.append(task)

        # Wait for all checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service_health(
        self, service_name: str, checker: HealthChecker
    ) -> None:
        """Check health of a specific service."""
        try:
            result = await checker.run_check()
            self.metrics[service_name].add_check_result(result)

            # Check if we need to send notifications
            if self.config.enable_notifications and self.config.notification_callback:
                await self._check_notifications(service_name, result)

        except Exception as e:
            logger.error(
                "Error checking service health", service_name=service_name, error=str(e)
            )

    async def _check_notifications(
        self, service_name: str, result: HealthCheckResult
    ) -> None:
        """Check if notifications should be sent."""
        metrics = self.metrics[service_name]

        # Send notification for status changes
        if result.status != HealthStatus.HEALTHY:
            if metrics.consecutive_failures == 1:  # First failure
                await self._send_notification(
                    f"Service {service_name} is now unhealthy", result
                )
        elif metrics.consecutive_failures > 0:  # Recovered
            await self._send_notification(
                f"Service {service_name} has recovered", result
            )

    async def _send_notification(self, message: str, result: HealthCheckResult) -> None:
        """Send health notification."""
        if self.config.notification_callback:
            try:
                self.config.notification_callback(message, result)
            except Exception as e:
                logger.error("Error sending health notification", error=str(e))

    async def check_service_now(self, service_name: str) -> HealthCheckResult | None:
        """Check a specific service immediately.

        Args:
            service_name: Name of the service to check

        Returns:
            Health check result or None if service not found
        """
        if service_name not in self.checkers:
            logger.warning("Service not found", service_name=service_name)
            return None

        checker = self.checkers[service_name]
        result = await checker.run_check()
        self.metrics[service_name].add_check_result(result)

        return result

    def get_service_health(self, service_name: str) -> dict[str, Any] | None:
        """Get health information for a service.

        Args:
            service_name: Name of the service

        Returns:
            Health information dictionary or None if service not found
        """
        if service_name not in self.metrics:
            return None

        metrics = self.metrics[service_name]
        return {
            "service_name": service_name,
            "status": metrics.get_status().value,
            "is_healthy": metrics.is_healthy,
            "is_degraded": metrics.is_degraded,
            "success_rate": metrics.success_rate,
            "total_checks": metrics.total_checks,
            "successful_checks": metrics.successful_checks,
            "failed_checks": metrics.failed_checks,
            "consecutive_failures": metrics.consecutive_failures,
            "average_response_time_ms": metrics.average_response_time,
            "last_check_time": (
                metrics.last_check_time.isoformat() if metrics.last_check_time else None
            ),
            "last_success_time": (
                metrics.last_success_time.isoformat()
                if metrics.last_success_time
                else None
            ),
            "last_failure_time": (
                metrics.last_failure_time.isoformat()
                if metrics.last_failure_time
                else None
            ),
        }

    def get_all_health(self) -> dict[str, dict[str, Any]]:
        """Get health information for all services.

        Returns:
            Dictionary of service names to health information
        """
        return {
            service_name: health_info
            for service_name in self.metrics.keys()
            if (health_info := self.get_service_health(service_name)) is not None
        }

    def get_global_health(self) -> dict[str, Any]:
        """Get global health status.

        Returns:
            Global health information
        """
        if not self.metrics:
            return {
                "overall_status": "unknown",
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0,
                "degraded_services": 0,
            }

        healthy_count = sum(
            1 for m in self.metrics.values() if m.get_status() == HealthStatus.HEALTHY
        )
        unhealthy_count = sum(
            1 for m in self.metrics.values() if m.get_status() == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for m in self.metrics.values() if m.get_status() == HealthStatus.DEGRADED
        )

        total_services = len(self.metrics)

        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        elif healthy_count == total_services:
            overall_status = "healthy"
        else:
            overall_status = "unknown"

        return {
            "overall_status": overall_status,
            "total_services": total_services,
            "healthy_services": healthy_count,
            "unhealthy_services": unhealthy_count,
            "degraded_services": degraded_count,
        }
