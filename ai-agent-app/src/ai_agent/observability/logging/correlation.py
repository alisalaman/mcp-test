"""Correlation ID management for distributed tracing."""

import asyncio
import contextvars
import uuid
from typing import Any
from collections.abc import Callable

import structlog
from fastapi import Request

# Context variable for correlation ID
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIDProcessor:
    """Processor to add correlation ID to log events."""

    def __init__(self, correlation_id_key: str = "correlation_id"):
        self.correlation_id_key = correlation_id_key

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Add correlation ID to log event."""
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict[self.correlation_id_key] = correlation_id
        return event_dict


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def clear_correlation_id() -> None:
    """Clear correlation ID from context."""
    correlation_id_var.set(None)


class CorrelationIDMiddleware:
    """Middleware to manage correlation IDs for HTTP requests."""

    def __init__(self, header_name: str = "X-Correlation-ID"):
        self.header_name = header_name

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        """Process request and add correlation ID."""
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = generate_correlation_id()

        # Set in context
        set_correlation_id(correlation_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id

        return response


class CorrelationContext:
    """Context manager for correlation ID."""

    def __init__(self, correlation_id: str | None = None):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "CorrelationContext":
        """Enter context and set correlation ID."""
        self.token = correlation_id_var.set(self.correlation_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore previous correlation ID."""
        if self.token:
            correlation_id_var.reset(self.token)


def with_correlation_id(correlation_id: str | None = None) -> Callable[[Any], Any]:
    """Decorator to add correlation ID to function execution."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with CorrelationContext(correlation_id):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with CorrelationContext(correlation_id):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


class CorrelationIDLogger:
    """Logger wrapper that automatically includes correlation ID."""

    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger

    def _log_with_correlation(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Log with correlation ID included."""
        correlation_id = get_correlation_id()
        if correlation_id:
            kwargs.setdefault("correlation_id", correlation_id)

        return getattr(self.logger, method_name)(*args, **kwargs)

    def debug(self, *args: Any, **kwargs: Any) -> Any:
        """Log debug message with correlation ID."""
        return self._log_with_correlation("debug", *args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> Any:
        """Log info message with correlation ID."""
        return self._log_with_correlation("info", *args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> Any:
        """Log warning message with correlation ID."""
        return self._log_with_correlation("warning", *args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> Any:
        """Log error message with correlation ID."""
        return self._log_with_correlation("error", *args, **kwargs)

    def critical(self, *args: Any, **kwargs: Any) -> Any:
        """Log critical message with correlation ID."""
        return self._log_with_correlation("critical", *args, **kwargs)

    def exception(self, *args: Any, **kwargs: Any) -> Any:
        """Log exception with correlation ID."""
        return self._log_with_correlation("exception", *args, **kwargs)


def get_correlation_logger(name: str) -> CorrelationIDLogger:
    """Get a logger that automatically includes correlation ID."""
    logger = structlog.get_logger(name)
    return CorrelationIDLogger(logger)


class CorrelationIDTracker:
    """Track correlation IDs across async operations."""

    def __init__(self) -> None:
        self.active_correlations: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def start_tracking(
        self, correlation_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Start tracking a correlation ID."""
        async with self._lock:
            self.active_correlations[correlation_id] = {
                "start_time": asyncio.get_event_loop().time(),
                "metadata": metadata or {},
                "operations": [],
            }

    async def add_operation(
        self,
        correlation_id: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an operation to a correlation ID."""
        async with self._lock:
            if correlation_id in self.active_correlations:
                self.active_correlations[correlation_id]["operations"].append(
                    {
                        "operation": operation,
                        "timestamp": asyncio.get_event_loop().time(),
                        "metadata": metadata or {},
                    }
                )

    async def stop_tracking(self, correlation_id: str) -> dict[str, Any] | None:
        """Stop tracking a correlation ID and return summary."""
        async with self._lock:
            if correlation_id in self.active_correlations:
                data = self.active_correlations[correlation_id]
                data["end_time"] = asyncio.get_event_loop().time()
                data["duration"] = data["end_time"] - data["start_time"]

                # Remove from active tracking
                del self.active_correlations[correlation_id]

                return data
            return None

    async def get_active_correlations(self) -> dict[str, dict[str, Any]]:
        """Get all active correlation IDs."""
        async with self._lock:
            return self.active_correlations.copy()

    async def cleanup_stale_correlations(self, max_age_seconds: int = 3600) -> int:
        """Clean up stale correlation IDs."""
        current_time = asyncio.get_event_loop().time()
        stale_ids = []

        async with self._lock:
            for correlation_id, data in self.active_correlations.items():
                if current_time - data["start_time"] > max_age_seconds:
                    stale_ids.append(correlation_id)

            for correlation_id in stale_ids:
                del self.active_correlations[correlation_id]

        return len(stale_ids)


# Global correlation ID tracker
_correlation_tracker: CorrelationIDTracker | None = None


def get_correlation_tracker() -> CorrelationIDTracker:
    """Get global correlation ID tracker."""
    global _correlation_tracker
    if _correlation_tracker is None:
        _correlation_tracker = CorrelationIDTracker()
    return _correlation_tracker


async def track_correlation_operation(
    operation: str, metadata: dict[str, Any] | None = None
) -> None:
    """Track an operation for the current correlation ID."""
    correlation_id = get_correlation_id()
    if correlation_id:
        tracker = get_correlation_tracker()
        await tracker.add_operation(correlation_id, operation, metadata)
