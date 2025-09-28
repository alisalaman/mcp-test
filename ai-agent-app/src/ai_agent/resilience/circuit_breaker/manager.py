"""Circuit breaker manager for multiple services."""

from typing import Any

import structlog

from .breaker import CircuitBreaker
from .config import CircuitBreakerConfig, CircuitBreakerSettings

logger = structlog.get_logger()


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""

    def __init__(self, settings: CircuitBreakerSettings):
        """Initialize circuit breaker manager.

        Args:
            settings: Circuit breaker settings
        """
        self.settings = settings
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker instance
        """
        if service_name not in self._breakers:
            config = self._get_config_for_service(service_name)
            self._breakers[service_name] = CircuitBreaker(service_name, config)
            logger.info(
                "Created circuit breaker for service",
                service_name=service_name,
                config=config.model_dump(),
            )

        return self._breakers[service_name]

    def _get_config_for_service(self, service_name: str) -> CircuitBreakerConfig:
        """Get configuration for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker configuration
        """
        if service_name == "llm":
            return self.settings.get_llm_config()
        elif service_name == "database":
            return self.settings.get_database_config()
        elif service_name == "mcp":
            return self.settings.get_mcp_config()
        elif service_name == "secret":
            return self.settings.get_secret_config()
        else:
            # Default configuration for unknown services
            logger.warning(
                "Unknown service type, using default circuit breaker config",
                service_name=service_name,
            )
            return CircuitBreakerConfig()

    def get_all_breakers(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers.

        Returns:
            Dictionary of service names to circuit breakers
        """
        return self._breakers.copy()

    def get_breaker_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary of service names to breaker statistics
        """
        return {
            name: breaker.get_state_info() for name, breaker in self._breakers.items()
        }

    def reset_all_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")

    def reset_breaker(self, service_name: str) -> None:
        """Reset a specific circuit breaker.

        Args:
            service_name: Name of the service
        """
        if service_name in self._breakers:
            self._breakers[service_name].reset()
            logger.info("Reset circuit breaker", service_name=service_name)
        else:
            logger.warning("Circuit breaker not found", service_name=service_name)

    def force_open_breaker(self, service_name: str) -> None:
        """Force a circuit breaker to open state.

        Args:
            service_name: Name of the service
        """
        if service_name in self._breakers:
            self._breakers[service_name].force_open()
            logger.warning("Forced circuit breaker open", service_name=service_name)
        else:
            logger.warning("Circuit breaker not found", service_name=service_name)

    def force_close_breaker(self, service_name: str) -> None:
        """Force a circuit breaker to closed state.

        Args:
            service_name: Name of the service
        """
        if service_name in self._breakers:
            self._breakers[service_name].force_close()
            logger.info("Forced circuit breaker closed", service_name=service_name)
        else:
            logger.warning("Circuit breaker not found", service_name=service_name)

    def get_global_stats(self) -> dict[str, Any]:
        """Get global statistics across all circuit breakers.

        Returns:
            Dictionary with global statistics
        """
        if not self._breakers:
            return {
                "total_breakers": 0,
                "open_breakers": 0,
                "half_open_breakers": 0,
                "closed_breakers": 0,
                "total_requests": 0,
                "total_failures": 0,
                "total_successes": 0,
            }

        total_breakers = len(self._breakers)
        open_breakers = sum(
            1 for b in self._breakers.values() if b.state.value == "open"
        )
        half_open_breakers = sum(
            1 for b in self._breakers.values() if b.state.value == "half_open"
        )
        closed_breakers = sum(
            1 for b in self._breakers.values() if b.state.value == "closed"
        )

        total_requests = sum(b.metrics.total_requests for b in self._breakers.values())
        total_failures = sum(b.metrics.failure_count for b in self._breakers.values())
        total_successes = sum(b.metrics.success_count for b in self._breakers.values())

        return {
            "total_breakers": total_breakers,
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "closed_breakers": closed_breakers,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "total_successes": total_successes,
        }

    def remove_breaker(self, service_name: str) -> None:
        """Remove a circuit breaker.

        Args:
            service_name: Name of the service
        """
        if service_name in self._breakers:
            del self._breakers[service_name]
            logger.info("Removed circuit breaker", service_name=service_name)
        else:
            logger.warning("Circuit breaker not found", service_name=service_name)

    def clear_all_breakers(self) -> None:
        """Clear all circuit breakers."""
        self._breakers.clear()
        logger.info("Cleared all circuit breakers")
