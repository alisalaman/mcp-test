"""Health check API endpoints."""

from datetime import datetime, UTC
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ai_agent.api.dependencies import get_settings_dependency
from ai_agent.config.settings import ApplicationSettings
from ai_agent.domain.models import ServiceHealth, ExternalServiceType

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=dict[str, str])
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
    }


@router.get("/ready", response_model=dict[str, Any])
async def readiness_check(
    settings: Annotated[ApplicationSettings, Depends(get_settings_dependency)],
) -> dict[str, Any]:
    """Readiness check for Kubernetes."""
    # Check if all required services are available
    checks = {
        "database": "unknown",
        "redis": "unknown",
        "external_services": "unknown",
    }

    # In a real implementation, you would check actual service health
    # For now, we'll simulate based on configuration
    if settings.use_database:
        checks["database"] = "healthy"  # Would check actual DB connection
    else:
        checks["database"] = "not_configured"

    if settings.use_redis:
        checks["redis"] = "healthy"  # Would check actual Redis connection
    else:
        checks["redis"] = "not_configured"

    checks["external_services"] = "healthy"  # Would check LLM providers, etc.

    # Determine overall readiness
    all_healthy = all(
        status in ["healthy", "not_configured"] for status in checks.values()
    )

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    return {
        "status": "ready",
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
    }


@router.get("/live", response_model=dict[str, str])
async def liveness_check() -> dict[str, str]:
    """Liveness check for Kubernetes."""
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/services", response_model=list[ServiceHealth])
async def service_health() -> list[ServiceHealth]:
    """Detailed health check for all services."""
    # In a real implementation, this would check actual service health
    services = [
        ServiceHealth(
            service_name="database",
            service_type=ExternalServiceType.DATABASE,
            status="healthy",
            last_check=datetime.now(UTC),
            error_count=0,
            success_count=100,
        ),
        ServiceHealth(
            service_name="redis",
            service_type=ExternalServiceType.CACHE,
            status="healthy",
            last_check=datetime.now(UTC),
            error_count=0,
            success_count=100,
        ),
        ServiceHealth(
            service_name="openai",
            service_type=ExternalServiceType.LLM_PROVIDER,
            status="healthy",
            last_check=datetime.now(UTC),
            error_count=0,
            success_count=50,
        ),
    ]

    return services
