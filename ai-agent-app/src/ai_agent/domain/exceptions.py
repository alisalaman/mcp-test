"""Exception hierarchy for AI Agent application."""

from typing import Any

from .models import ErrorCode, ExternalServiceType


class AIAgentException(Exception):
    """Base exception for AI agent application."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id


class ValidationException(AIAgentException):
    """Validation error exception."""

    def __init__(self, message: str, field: str, value: Any):
        super().__init__(
            message, ErrorCode.VALIDATION_ERROR, {"field": field, "value": str(value)}
        )


class AuthenticationException(AIAgentException):
    """Authentication error exception."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR)


class AuthorizationException(AIAgentException):
    """Authorization error exception."""

    def __init__(self, message: str = "Authorization failed"):
        super().__init__(message, ErrorCode.AUTHORIZATION_ERROR)


class RateLimitException(AIAgentException):
    """Rate limit exceeded exception."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message, ErrorCode.RATE_LIMIT_ERROR)
        self.retry_after = retry_after


class ExternalServiceException(AIAgentException):
    """External service error exception."""

    def __init__(
        self,
        message: str,
        service_name: str,
        service_type: ExternalServiceType,
        is_retryable: bool = True,
    ):
        super().__init__(
            message,
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            {
                "service_name": service_name,
                "service_type": service_type,
                "is_retryable": is_retryable,
            },
        )
        self.is_retryable = is_retryable


class CircuitBreakerOpenException(AIAgentException):
    """Circuit breaker open exception."""

    def __init__(self, service_name: str):
        super().__init__(
            f"Circuit breaker open for service: {service_name}",
            ErrorCode.CIRCUIT_BREAKER_OPEN,
            {"service_name": service_name},
        )


class TimeoutException(AIAgentException):
    """Timeout error exception."""

    def __init__(self, message: str, timeout_duration: float):
        super().__init__(
            message, ErrorCode.TIMEOUT_ERROR, {"timeout_duration": timeout_duration}
        )
