"""Health check endpoints for FastAPI."""

from typing import Any
from datetime import datetime, UTC
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from .checker import HealthChecker
from ..logging import get_logger

logger = get_logger(__name__)


def create_health_endpoints(
    health_checker: HealthChecker | None = None,
    include_details: bool = True,
    include_dependencies: bool = True,
) -> APIRouter:
    """Create health check endpoints for FastAPI.

    Args:
        health_checker: Health checker instance (optional)
        include_details: Whether to include detailed health information
        include_dependencies: Whether to include dependency health checks

    Returns:
        FastAPI router with health endpoints
    """
    router = APIRouter(prefix="/health", tags=["health"])

    if health_checker is None:
        from .checker import get_health_checker

        health_checker = get_health_checker()

    @router.get("/")
    async def health_check() -> dict[str, Any] | JSONResponse:
        """Basic health check endpoint.

        Returns:
            Health status
        """
        try:
            results = await health_checker.run_all_checks()

            # Determine overall health status
            all_healthy = all(
                check_result.get("status") == "healthy"
                for check_result in results.get("checks", {}).values()
            )

            response_data = {
                "status": "healthy" if all_healthy else "unhealthy",
                "timestamp": results.get("timestamp"),
                "checks": results.get("checks", {}),
            }

            if include_details:
                response_data.update(
                    {
                        "summary": results.get("summary", {}),
                        "duration_ms": results.get("duration_ms", 0),
                    }
                )

            if not all_healthy:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=response_data,
                )

            return response_data

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "message": f"Health check failed: {str(e)}",
                    "timestamp": None,
                },
            )

    @router.get("/ready")
    async def readiness_check() -> dict[str, Any] | JSONResponse:
        """Readiness check endpoint.

        Returns:
            Readiness status
        """
        try:
            results = await health_checker.run_all_checks()
            # For readiness, we consider the system ready if all critical checks pass
            critical_checks = health_checker.get_critical_checks()
            all_ready = all(
                results.get("checks", {}).get(check_name, {}).get("status") == "healthy"
                for check_name in critical_checks
            )
            result = type(
                "Result",
                (),
                {
                    "is_healthy": all_ready,
                    "timestamp": results.get("timestamp"),
                    "message": (
                        "System is ready" if all_ready else "System is not ready"
                    ),
                    "response_time_ms": results.get("duration_ms", 0),
                },
            )()

            response_data = {
                "status": "ready" if result.is_healthy else "not_ready",
                "timestamp": result.timestamp,
            }

            if include_details:
                response_data.update(
                    {
                        "message": result.message,
                        "response_time_ms": result.response_time_ms,
                    }
                )

            if not result.is_healthy:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=response_data,
                )

            return response_data

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "message": f"Readiness check failed: {str(e)}",
                    "timestamp": None,
                },
            )

    @router.get("/live")
    async def liveness_check() -> dict[str, Any] | JSONResponse:
        """Liveness check endpoint.

        Returns:
            Liveness status
        """
        try:
            # For liveness, we just check if the service is responding
            # This is a simple check that doesn't require external dependencies
            result = type(
                "Result",
                (),
                {
                    "is_healthy": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": "Service is alive",
                    "response_time_ms": 0,
                },
            )()

            response_data = {
                "status": "alive" if result.is_healthy else "dead",
                "timestamp": result.timestamp,
            }

            if include_details:
                response_data.update(
                    {
                        "message": result.message,
                        "response_time_ms": result.response_time_ms,
                    }
                )

            if not result.is_healthy:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=response_data,
                )

            return response_data

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "dead",
                    "message": f"Liveness check failed: {str(e)}",
                    "timestamp": None,
                },
            )

    if include_dependencies:

        @router.get("/dependencies")
        async def dependencies_check() -> dict[str, Any] | JSONResponse:
            """Dependencies health check endpoint.

            Returns:
                Dependencies health status
            """
            try:
                results = await health_checker.run_all_checks()

                checks = results.get("checks", {})
                all_healthy = all(
                    check_result.get("status") == "healthy"
                    for check_result in checks.values()
                )

                response_data: dict[str, Any] = {
                    "status": "healthy" if all_healthy else "unhealthy",
                    "timestamp": results.get("timestamp", ""),
                    "dependencies": {},
                }

                for name, result in checks.items():
                    dep_data = {
                        "status": result.get("status", "unknown"),
                        "message": result.get("message", ""),
                    }

                    if include_details:
                        dep_data.update(
                            {
                                "response_time_ms": result.get("duration_ms", 0),
                                "timestamp": result.get("timestamp"),
                            }
                        )

                    response_data["dependencies"][name] = dep_data

                # Set overall timestamp
                if checks:
                    timestamps = [
                        result.get("timestamp")
                        for result in checks.values()
                        if result.get("timestamp")
                    ]
                    if timestamps:
                        response_data["timestamp"] = max(timestamps)

                if not all_healthy:
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content=response_data,
                    )

                return response_data

            except Exception as e:
                logger.error(f"Dependencies check failed: {e}")
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": "unhealthy",
                        "message": f"Dependencies check failed: {str(e)}",
                        "timestamp": None,
                        "dependencies": {},
                    },
                )

    @router.get("/metrics")
    async def health_metrics() -> dict[str, Any] | JSONResponse:
        """Health metrics endpoint.

        Returns:
            Health metrics
        """
        try:
            # Get health check results as metrics
            results = await health_checker.run_all_checks()

            response_data = {
                "metrics": results.get("summary", {}),
                "timestamp": results.get("timestamp"),
            }

            return response_data

        except Exception as e:
            logger.error(f"Health metrics failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": f"Health metrics failed: {str(e)}",
                    "timestamp": None,
                },
            )

    return router


def create_simple_health_endpoint() -> APIRouter:
    """Create a simple health endpoint without dependencies.

    Returns:
        FastAPI router with basic health endpoint
    """
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("/")
    async def health_check() -> dict[str, Any] | JSONResponse:
        """Simple health check endpoint.

        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "message": "Service is running",
            "timestamp": None,
        }

    return router
