"""Health check endpoints for FastAPI integration."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .monitor import ServiceHealthMonitor


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Timestamp of the health check")
    services: dict[str, dict[str, Any]] = Field(description="Individual service health")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    version: str = Field(description="Application version")


class ServiceHealthResponse(BaseModel):
    """Individual service health response model."""

    service_name: str
    status: str
    is_healthy: bool
    is_degraded: bool
    success_rate: float
    total_checks: int
    successful_checks: int
    failed_checks: int
    consecutive_failures: int
    average_response_time_ms: float
    last_check_time: str | None = None
    last_success_time: str | None = None
    last_failure_time: str | None = None


class HealthEndpoint:
    """Health check endpoint handler."""

    def __init__(
        self,
        monitor: ServiceHealthMonitor,
        app_version: str = "1.0.0",
        start_time: datetime | None = None,
    ):
        """Initialize health endpoint.

        Args:
            monitor: Health monitor instance
            app_version: Application version
            start_time: Application start time
        """
        self.monitor = monitor
        self.app_version = app_version
        self.start_time = start_time or datetime.now(UTC)

    async def get_health(self) -> HealthResponse:
        """Get overall health status.

        Returns:
            Health response
        """
        # Get all service health information
        services_health = self.monitor.get_all_health()
        global_health = self.monitor.get_global_health()

        # Calculate uptime
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        return HealthResponse(
            status=global_health["overall_status"],
            timestamp=datetime.now(UTC).isoformat(),
            services=services_health,
            uptime_seconds=uptime,
            version=self.app_version,
        )

    async def get_service_health(
        self, service_name: str
    ) -> ServiceHealthResponse | None:
        """Get health for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            Service health response or None if not found
        """
        health_data = self.monitor.get_service_health(service_name)
        if not health_data:
            return None

        return ServiceHealthResponse(**health_data)

    async def get_health_summary(self) -> dict[str, Any]:
        """Get health summary.

        Returns:
            Health summary dictionary
        """
        global_health = self.monitor.get_global_health()
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        return {
            "status": global_health["overall_status"],
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": uptime,
            "version": self.app_version,
            "summary": global_health,
        }

    async def check_service_now(self, service_name: str) -> dict[str, Any] | None:
        """Check a specific service immediately.

        Args:
            service_name: Name of the service to check

        Returns:
            Health check result or None if service not found
        """
        result = await self.monitor.check_service_now(service_name)
        if result:
            return result.to_dict()
        return None

    def get_health_metrics(self) -> dict[str, Any]:
        """Get health metrics for monitoring.

        Returns:
            Health metrics dictionary
        """
        global_health = self.monitor.get_global_health()
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        return {
            "health_status": global_health["overall_status"],
            "total_services": global_health["total_services"],
            "healthy_services": global_health["healthy_services"],
            "unhealthy_services": global_health["unhealthy_services"],
            "degraded_services": global_health["degraded_services"],
            "uptime_seconds": uptime,
            "version": self.app_version,
        }
