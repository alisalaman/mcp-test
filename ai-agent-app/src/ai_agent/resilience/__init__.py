"""Resilience patterns for external service integration.

This module provides comprehensive resilience patterns including retry logic,
circuit breakers, health checking, fallback strategies, and rate limiting
for production-ready external service integration.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerManager, circuit_protected
from .exceptions import (
    CircuitBreakerOpenException,
    ExternalServiceException,
    FallbackFailedException,
    HealthCheckFailedException,
    RateLimitExceededException,
    ResilienceException,
    RetryExhaustedException,
)
from .fallback import FallbackManager, FallbackStrategy
from .health import HealthChecker, HealthEndpoint, ServiceHealthMonitor
from .integration import (
    ResilienceManager,
    database_resilient,
    llm_resilient,
    mcp_resilient,
    resilient_service,
    secret_resilient,
)
from .rate_limiting import RateLimiter, RateLimitManager
from .retry import RetryManager

__all__ = [
    "RetryManager",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "circuit_protected",
    "HealthChecker",
    "ServiceHealthMonitor",
    "HealthEndpoint",
    "FallbackManager",
    "FallbackStrategy",
    "RateLimiter",
    "RateLimitManager",
    "resilient_service",
    "llm_resilient",
    "database_resilient",
    "mcp_resilient",
    "secret_resilient",
    "ResilienceManager",
    "ResilienceException",
    "ExternalServiceException",
    "CircuitBreakerOpenException",
    "RateLimitExceededException",
    "HealthCheckFailedException",
    "FallbackFailedException",
    "RetryExhaustedException",
]
