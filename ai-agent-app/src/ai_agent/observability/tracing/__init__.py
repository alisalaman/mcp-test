"""Distributed tracing implementation."""

from .tracer import setup_tracing, get_tracer, TraceContext
from .spans import SpanManager, create_span, trace_function, trace_async_function
from .instrumentation import (
    instrument_fastapi,
    instrument_httpx,
    instrument_redis,
    instrument_database,
)

__all__ = [
    "setup_tracing",
    "get_tracer",
    "TraceContext",
    "SpanManager",
    "create_span",
    "trace_function",
    "trace_async_function",
    "instrument_fastapi",
    "instrument_httpx",
    "instrument_redis",
    "instrument_database",
]
