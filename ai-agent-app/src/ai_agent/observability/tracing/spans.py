"""Span management and decorators for tracing."""

import asyncio
import functools
from typing import Any
from collections.abc import Callable, Generator, AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from opentelemetry.trace import Tracer, Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ..logging import get_logger

logger = get_logger(__name__)


class SpanManager:
    """Manager for creating and managing spans."""

    def __init__(self, tracer: Tracer):
        """Initialize span manager.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self.tracer = tracer
        self._active_spans: dict[str, Span] = {}

    def create_span(
        self,
        name: str,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span:
        """Create a new span.

        Args:
            name: Span name
            parent_span: Parent span (optional)
            attributes: Span attributes
            kind: Span kind (internal, server, client, producer, consumer)

        Returns:
            Created span
        """
        from opentelemetry.trace import SpanKind

        span_kind_map = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        span_kind = span_kind_map.get(kind or "internal", SpanKind.INTERNAL)

        span = self.tracer.start_span(
            name=name,
            kind=span_kind,
        )

        if parent_span:
            # Set parent context
            from opentelemetry.trace import set_span_in_context

            set_span_in_context(parent_span)
            # Note: OpenTelemetry spans don't have set_parent method
            # Parent relationship is established through context

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span

    def start_span(
        self,
        name: str,
        span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> str:
        """Start a span and return its ID.

        Args:
            name: Span name
            span_id: Custom span ID (optional)
            attributes: Span attributes
            kind: Span kind

        Returns:
            Span ID for tracking
        """
        if span_id is None:
            span_id = f"{name}_{id(asyncio.current_task())}"

        span = self.create_span(name, attributes=attributes, kind=kind)
        self._active_spans[span_id] = span

        logger.debug(f"Started span: {name} (ID: {span_id})")
        return span_id

    def end_span(
        self,
        span_id: str,
        status: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End a span.

        Args:
            span_id: Span ID
            status: Span status (ok, error)
            attributes: Additional attributes to set before ending
        """
        if span_id not in self._active_spans:
            logger.warning(f"Span {span_id} not found in active spans")
            return

        span = self._active_spans[span_id]

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        if status:
            if status == "ok":
                span.set_status(Status(StatusCode.OK))
            elif status == "error":
                span.set_status(Status(StatusCode.ERROR))

        span.end()
        del self._active_spans[span_id]

        logger.debug(f"Ended span: {span_id}")

    def get_active_span(self, span_id: str) -> Span | None:
        """Get active span by ID.

        Args:
            span_id: Span ID

        Returns:
            Active span or None if not found
        """
        return self._active_spans.get(span_id)

    def get_all_active_spans(self) -> dict[str, Span]:
        """Get all active spans.

        Returns:
            Dictionary of active spans
        """
        return self._active_spans.copy()

    def clear_all_spans(self) -> None:
        """Clear all active spans."""
        for span in self._active_spans.values():
            span.end()
        self._active_spans.clear()
        logger.debug("Cleared all active spans")


# Global span manager instance
_span_manager: SpanManager | None = None


def get_span_manager() -> SpanManager:
    """Get global span manager instance."""
    global _span_manager
    if _span_manager is None:
        from .tracer import get_tracer

        tracer = get_tracer()
        _span_manager = SpanManager(tracer)
    return _span_manager


def create_span(
    name: str,
    parent_span: Span | None = None,
    attributes: dict[str, Any] | None = None,
    kind: str | None = None,
) -> Span:
    """Create a new span using the global span manager.

    Args:
        name: Span name
        parent_span: Parent span (optional)
        attributes: Span attributes
        kind: Span kind

    Returns:
        Created span
    """
    manager = get_span_manager()
    return manager.create_span(name, parent_span, attributes, kind)


def trace_function(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    kind: str | None = None,
    capture_exceptions: bool = True,
) -> Callable[..., Any]:
    """Decorator to trace synchronous functions.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Span attributes
        kind: Span kind
        capture_exceptions: Whether to capture exceptions in span
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or f"{func.__module__}.{func.__name__}"
            manager = get_span_manager()
            span_id = manager.start_span(span_name, attributes=attributes, kind=kind)

            try:
                result = func(*args, **kwargs)
                manager.end_span(span_id, status="ok")
                return result
            except Exception as e:
                if capture_exceptions:
                    manager.end_span(
                        span_id, status="error", attributes={"error": str(e)}
                    )
                raise
            finally:
                # Ensure span is ended even if not explicitly ended
                if span_id in manager.get_all_active_spans():
                    manager.end_span(span_id)

        return wrapper

    return decorator


def trace_async_function(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    kind: str | None = None,
    capture_exceptions: bool = True,
) -> Callable[..., Any]:
    """Decorator to trace asynchronous functions.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Span attributes
        kind: Span kind
        capture_exceptions: Whether to capture exceptions in span
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or f"{func.__module__}.{func.__name__}"
            manager = get_span_manager()
            span_id = manager.start_span(span_name, attributes=attributes, kind=kind)

            try:
                result = await func(*args, **kwargs)
                manager.end_span(span_id, status="ok")
                return result
            except Exception as e:
                if capture_exceptions:
                    manager.end_span(
                        span_id, status="error", attributes={"error": str(e)}
                    )
                raise
            finally:
                # Ensure span is ended even if not explicitly ended
                if span_id in manager.get_all_active_spans():
                    manager.end_span(span_id)

        return wrapper

    return decorator


@contextmanager
def trace_context(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: str | None = None,
) -> Generator[Span]:
    """Context manager for tracing.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind
    """
    manager = get_span_manager()
    span_id = manager.start_span(name, attributes=attributes, kind=kind)

    try:
        span = manager.get_active_span(span_id)
        if span is None:
            raise RuntimeError(f"Failed to create span: {span_id}")
        yield span
        manager.end_span(span_id, status="ok")
    except Exception as e:
        manager.end_span(span_id, status="error", attributes={"error": str(e)})
        raise
    finally:
        # Ensure span is ended
        if span_id in manager.get_all_active_spans():
            manager.end_span(span_id)


@asynccontextmanager
async def trace_async_context(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: str | None = None,
) -> AsyncGenerator[Span]:
    """Async context manager for tracing.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind
    """
    manager = get_span_manager()
    span_id = manager.start_span(name, attributes=attributes, kind=kind)

    try:
        span = manager.get_active_span(span_id)
        if span is None:
            raise RuntimeError(f"Failed to create span: {span_id}")
        yield span
        manager.end_span(span_id, status="ok")
    except Exception as e:
        manager.end_span(span_id, status="error", attributes={"error": str(e)})
        raise
    finally:
        # Ensure span is ended
        if span_id in manager.get_all_active_spans():
            manager.end_span(span_id)


def inject_trace_context(carrier: dict[str, str]) -> None:
    """Inject trace context into carrier.

    Args:
        carrier: Dictionary to inject trace context into
    """
    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)


def extract_trace_context(carrier: dict[str, str]) -> Any:
    """Extract trace context from carrier.

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Extracted trace context
    """
    propagator = TraceContextTextMapPropagator()
    return propagator.extract(carrier)
