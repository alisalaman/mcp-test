"""Error handling and response standardization for the API."""

from datetime import datetime, UTC
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ai_agent.domain.exceptions import (
    AIAgentException,
    AuthenticationException,
    AuthorizationException,
    CircuitBreakerOpenException,
    ExternalServiceException,
    RateLimitException,
    TimeoutException,
    ValidationException,
)


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error: bool = True
    code: str
    message: str
    details: dict[str, Any] | None = None
    correlation_id: str | None = None
    timestamp: str
    path: str


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""

    field: str
    message: str
    value: Any


class ValidationErrorResponse(ErrorResponse):
    """Validation error response."""

    code: str = "validation_error"
    validation_errors: list[ValidationErrorDetail]


def get_correlation_id(request: Request) -> str:
    """Extract correlation ID from request."""
    return getattr(request.state, "correlation_id", "unknown")


def create_error_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=code,
            message=message,
            details=details,
            correlation_id=get_correlation_id(request),
            timestamp=datetime.now(UTC).isoformat(),
            path=str(request.url.path),
        ).model_dump(),
    )


def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation errors."""
    if not isinstance(exc, RequestValidationError):
        # If it's not a RequestValidationError, re-raise it
        raise exc

    validation_errors = []

    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        validation_errors.append(
            ValidationErrorDetail(
                field=field,
                message=error["msg"],
                value=error.get("input"),
            )
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            message="Validation failed",
            validation_errors=validation_errors,
            correlation_id=get_correlation_id(request),
            timestamp=datetime.now(UTC).isoformat(),
            path=str(request.url.path),
        ).model_dump(),
    )


async def validation_exception_handler_custom(
    request: Request, exc: ValidationException
) -> JSONResponse:
    """Handle custom validation exceptions."""
    return create_error_response(
        request=request,
        code="validation_error",
        message=str(exc),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=exc.details,
    )


async def authentication_exception_handler(
    request: Request, exc: AuthenticationException
) -> JSONResponse:
    """Handle authentication errors."""
    return create_error_response(
        request=request,
        code="authentication_error",
        message=str(exc),
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


async def authorization_exception_handler(
    request: Request, exc: AuthorizationException
) -> JSONResponse:
    """Handle authorization errors."""
    return create_error_response(
        request=request,
        code="authorization_error",
        message=str(exc),
        status_code=status.HTTP_403_FORBIDDEN,
    )


async def rate_limit_exception_handler(
    request: Request, exc: RateLimitException
) -> JSONResponse:
    """Handle rate limit errors."""
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers=headers,
        content=ErrorResponse(
            code="rate_limit_error",
            message=str(exc),
            details={"retry_after": exc.retry_after},
            correlation_id=get_correlation_id(request),
            timestamp=datetime.now(UTC).isoformat(),
            path=str(request.url.path),
        ).model_dump(),
    )


async def external_service_exception_handler(
    request: Request, exc: ExternalServiceException
) -> JSONResponse:
    """Handle external service errors."""
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if exc.is_retryable
        else status.HTTP_502_BAD_GATEWAY
    )

    return create_error_response(
        request=request,
        code="external_service_error",
        message=str(exc),
        status_code=status_code,
        details=exc.details,
    )


async def circuit_breaker_exception_handler(
    request: Request, exc: CircuitBreakerOpenException
) -> JSONResponse:
    """Handle circuit breaker open errors."""
    return create_error_response(
        request=request,
        code="circuit_breaker_open",
        message=str(exc),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        details=exc.details,
    )


async def timeout_exception_handler(
    request: Request, exc: TimeoutException
) -> JSONResponse:
    """Handle timeout errors."""
    return create_error_response(
        request=request,
        code="timeout_error",
        message=str(exc),
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        details=exc.details,
    )


async def general_exception_handler(
    request: Request, exc: AIAgentException
) -> JSONResponse:
    """Handle general AI agent exceptions."""
    return create_error_response(
        request=request,
        code=exc.error_code.value,
        message=str(exc),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details=exc.details,
    )


async def unexpected_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    return create_error_response(
        request=request,
        code="internal_error",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"exception_type": type(exc).__name__},
    )
