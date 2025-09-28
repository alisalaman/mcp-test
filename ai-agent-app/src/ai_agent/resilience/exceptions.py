"""Resilience-specific exceptions."""

from typing import Any

from ai_agent.domain.exceptions import AIAgentException, ErrorCode


class ResilienceException(AIAgentException):
    """Base exception for resilience patterns."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        super().__init__(message, error_code, details, correlation_id)


class ExternalServiceException(ResilienceException):
    """External service error exception."""

    def __init__(
        self,
        message: str,
        service_name: str,
        service_type: str = "unknown",
        is_retryable: bool = True,
        correlation_id: str | None = None,
    ):
        super().__init__(
            message,
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            {
                "service_name": service_name,
                "service_type": service_type,
                "is_retryable": is_retryable,
            },
            correlation_id,
        )
        self.service_name = service_name
        self.service_type = service_type
        self.is_retryable = is_retryable


class CircuitBreakerOpenException(ResilienceException):
    """Circuit breaker is open exception."""

    def __init__(self, service_name: str, correlation_id: str | None = None):
        super().__init__(
            f"Circuit breaker open for service: {service_name}",
            ErrorCode.CIRCUIT_BREAKER_OPEN,
            {"service_name": service_name},
            correlation_id,
        )
        self.service_name = service_name


class RateLimitExceededException(ResilienceException):
    """Rate limit exceeded exception."""

    def __init__(
        self,
        service_name: str,
        retry_after: float | None = None,
        correlation_id: str | None = None,
    ):
        super().__init__(
            f"Rate limit exceeded for service: {service_name}",
            ErrorCode.RATE_LIMIT_ERROR,
            {"service_name": service_name, "retry_after": retry_after},
            correlation_id,
        )
        self.service_name = service_name
        self.retry_after = retry_after


class HealthCheckFailedException(ResilienceException):
    """Health check failed exception."""

    def __init__(
        self,
        service_name: str,
        error_message: str,
        correlation_id: str | None = None,
    ):
        super().__init__(
            f"Health check failed for service: {service_name}",
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            {"service_name": service_name, "error_message": error_message},
            correlation_id,
        )
        self.service_name = service_name
        self.error_message = error_message


class FallbackFailedException(ResilienceException):
    """Fallback strategy failed exception."""

    def __init__(
        self,
        service_name: str,
        fallback_strategy: str,
        correlation_id: str | None = None,
    ):
        super().__init__(
            f"Fallback strategy failed for service: {service_name}",
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            {"service_name": service_name, "fallback_strategy": fallback_strategy},
            correlation_id,
        )
        self.service_name = service_name
        self.fallback_strategy = fallback_strategy


class RetryExhaustedException(ResilienceException):
    """Retry attempts exhausted exception."""

    def __init__(
        self,
        service_name: str,
        max_attempts: int,
        last_error: str,
        correlation_id: str | None = None,
    ):
        super().__init__(
            f"Retry attempts exhausted for service: {service_name}",
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            {
                "service_name": service_name,
                "max_attempts": max_attempts,
                "last_error": last_error,
            },
            correlation_id,
        )
        self.service_name = service_name
        self.max_attempts = max_attempts
        self.last_error = last_error
