"""Circuit breaker implementation for external service protection.

This module provides circuit breaker patterns to prevent cascading failures
and improve system resilience when external services are unavailable.
"""

from .breaker import CircuitBreaker, CircuitBreakerMetrics, CircuitState
from .config import CircuitBreakerConfig, CircuitBreakerSettings
from .decorators import circuit_protected
from .manager import CircuitBreakerManager

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerMetrics",
    "CircuitBreakerManager",
    "circuit_protected",
    "CircuitBreakerConfig",
    "CircuitBreakerSettings",
]
