"""Distributed tracing configuration and management."""

from typing import Any
from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer, Span, Status, StatusCode

from ..logging import get_logger

logger = get_logger(__name__)


class TraceContext:
    """Context manager for trace operations."""

    def __init__(
        self, tracer: Tracer, name: str, attributes: dict[str, Any] | None = None
    ):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes or {}
        self.span: Span | None = None

    def __enter__(self) -> Span:
        """Enter trace context."""
        self.span = self.tracer.start_span(self.name, attributes=self.attributes)
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit trace context."""
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.end()

    async def __aenter__(self) -> Span:
        """Async enter trace context."""
        self.span = self.tracer.start_span(self.name, attributes=self.attributes)
        return self.span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async exit trace context."""
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.end()


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "ai-agent-app",
        service_version: str = "1.0.0",
        environment: str = "development",
        sample_rate: float = 1.0,
        exporter_type: str = "console",  # console, jaeger, otlp, zipkin
        exporter_config: dict[str, Any] | None = None,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.sample_rate = sample_rate
        self.exporter_type = exporter_type
        self.exporter_config = exporter_config or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "sample_rate": self.sample_rate,
            "exporter_type": self.exporter_type,
            "exporter_config": self.exporter_config,
        }


def setup_tracing(config: TracingConfig) -> Tracer:
    """Setup distributed tracing."""
    try:
        # Create resource
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
                "deployment.environment": config.environment,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Create exporter based on type
        exporter = _create_exporter(config)

        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

        # Get tracer
        tracer = trace.get_tracer(__name__)

        logger.info(
            f"Tracing setup complete: {config.service_name} v{config.service_version}"
        )
        return tracer

    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        # Return a no-op tracer
        return trace.NoOpTracer()


def _create_exporter(config: TracingConfig) -> Any:
    """Create trace exporter based on configuration."""
    exporter_type = config.exporter_type.lower()
    exporter_config = config.exporter_config

    if exporter_type == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    elif exporter_type == "jaeger":
        return JaegerExporter(
            agent_host_name=exporter_config.get("agent_host", "localhost"),
            agent_port=exporter_config.get("agent_port", 14268),
            collector_endpoint=exporter_config.get("collector_endpoint"),
        )

    elif exporter_type == "otlp":
        return OTLPSpanExporter(
            endpoint=exporter_config.get("endpoint", "http://localhost:4317"),
            headers=exporter_config.get("headers", {}),
        )

    elif exporter_type == "zipkin":
        return ZipkinExporter(
            endpoint=exporter_config.get(
                "endpoint", "http://localhost:9411/api/v2/spans"
            )
        )

    else:
        logger.warning(f"Unknown exporter type: {exporter_type}, using console")
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()


def get_tracer() -> Tracer:
    """Get the configured tracer."""
    return trace.get_tracer(__name__)


def create_trace_context(
    name: str, attributes: dict[str, Any] | None = None
) -> TraceContext:
    """Create a trace context."""
    tracer = get_tracer()
    return TraceContext(tracer, name, attributes)


@asynccontextmanager
async def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Any:
    """Async context manager for tracing spans."""
    tracer = get_tracer()
    span = tracer.start_span(name, attributes=attributes)
    try:
        yield span
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.set_status(Status(StatusCode.OK))
        span.end()


def trace_function(
    name: str | None = None, attributes: dict[str, Any] | None = None
) -> Any:
    """Decorator to trace function execution."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracer.start_span(span_name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def trace_async_function(
    name: str | None = None, attributes: dict[str, Any] | None = None
) -> Any:
    """Decorator to trace async function execution."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracer.start_span(span_name, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


class TraceMiddleware:
    """Middleware to automatically trace HTTP requests."""

    def __init__(self, app_name: str = "ai-agent-app"):
        self.app_name = app_name
        self.tracer = get_tracer()

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Process request with tracing."""
        span_name = f"{request.method} {request.url.path}"

        with self.tracer.start_span(span_name) as span:
            # Add request attributes
            span.set_attributes(
                {
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.user_agent": request.headers.get("user-agent", ""),
                    "http.request_id": request.headers.get("x-request-id", ""),
                }
            )

            try:
                # Process request
                response = await call_next(request)

                # Add response attributes
                span.set_attributes(
                    {
                        "http.status_code": response.status_code,
                        "http.response_size": (
                            len(response.body) if hasattr(response, "body") else 0
                        ),
                    }
                )

                return response

            except Exception as e:
                # Add error attributes
                span.set_attributes(
                    {
                        "error": True,
                        "error.message": str(e),
                        "error.type": type(e).__name__,
                    }
                )
                span.record_exception(e)
                raise


def instrument_application(
    app: Any, service_name: str = "ai-agent-app", service_version: str = "1.0.0"
) -> None:
    """Instrument FastAPI application for tracing."""
    try:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls="health,metrics",
        )
        logger.info(f"Instrumented FastAPI app: {service_name}")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI app: {e}")


def instrument_http_client() -> None:
    """Instrument HTTP client for tracing."""
    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("Instrumented HTTPX client")
    except Exception as e:
        logger.error(f"Failed to instrument HTTPX client: {e}")


def instrument_redis() -> None:
    """Instrument Redis for tracing."""
    try:
        RedisInstrumentor().instrument()
        logger.info("Instrumented Redis")
    except Exception as e:
        logger.error(f"Failed to instrument Redis: {e}")


def instrument_database() -> None:
    """Instrument database for tracing."""
    try:
        SQLAlchemyInstrumentor().instrument()
        logger.info("Instrumented SQLAlchemy")
    except Exception as e:
        logger.error(f"Failed to instrument SQLAlchemy: {e}")


def setup_auto_instrumentation() -> None:
    """Setup automatic instrumentation for common libraries."""
    try:
        # Instrument HTTP clients
        instrument_http_client()

        # Instrument Redis
        instrument_redis()

        # Instrument database
        instrument_database()

        logger.info("Auto-instrumentation setup complete")
    except Exception as e:
        logger.error(f"Failed to setup auto-instrumentation: {e}")


def get_trace_id() -> str | None:
    """Get current trace ID."""
    span = trace.get_current_span()
    if span and span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_span_id() -> str | None:
    """Get current span ID."""
    span = trace.get_current_span()
    if span and span.is_recording():
        return format(span.get_span_context().span_id, "016x")
    return None


def add_span_attribute(key: str, value: Any) -> None:
    """Add attribute to current span."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add event to current span."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes or {})


def record_exception(exception: Exception) -> None:
    """Record exception in current span."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)


# Global tracing configuration
_tracing_config: TracingConfig | None = None


def configure_tracing(config: TracingConfig) -> None:
    """Configure global tracing settings."""
    global _tracing_config
    _tracing_config = config


def get_tracing_config() -> TracingConfig | None:
    """Get current tracing configuration."""
    return _tracing_config
