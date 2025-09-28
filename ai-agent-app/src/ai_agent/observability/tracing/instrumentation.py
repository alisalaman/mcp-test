"""Instrumentation for various libraries and frameworks."""

from typing import Any
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from ..logging import get_logger

logger = get_logger(__name__)


def instrument_fastapi(app: Any, tracer_provider: Any | None = None) -> None:
    """Instrument FastAPI application for tracing.

    Args:
        app: FastAPI application instance
        tracer_provider: Optional tracer provider
    """
    try:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
        logger.info("FastAPI application instrumented for tracing")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")
        raise


def instrument_httpx(tracer_provider: Any | None = None) -> None:
    """Instrument httpx client for tracing.

    Args:
        tracer_provider: Optional tracer provider
    """
    try:
        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("httpx client instrumented for tracing")
    except Exception as e:
        logger.error(f"Failed to instrument httpx: {e}")
        raise


def instrument_redis(tracer_provider: Any | None = None) -> None:
    """Instrument Redis client for tracing.

    Args:
        tracer_provider: Optional tracer provider
    """
    try:
        RedisInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Redis client instrumented for tracing")
    except Exception as e:
        logger.error(f"Failed to instrument Redis: {e}")
        raise


def instrument_database(
    engine: Any,
    tracer_provider: Any | None = None,
    enable_commenter: bool = True,
    commenter_options: dict[str, Any] | None = None,
) -> None:
    """Instrument SQLAlchemy database engine for tracing.

    Args:
        engine: SQLAlchemy engine instance
        tracer_provider: Optional tracer provider
        enable_commenter: Whether to enable SQL commenter
        commenter_options: Options for SQL commenter
    """
    try:
        SQLAlchemyInstrumentor().instrument(
            engine=engine,
            tracer_provider=tracer_provider,
            enable_commenter=enable_commenter,
            commenter_options=commenter_options or {},
        )
        logger.info("Database engine instrumented for tracing")
    except Exception as e:
        logger.error(f"Failed to instrument database: {e}")
        raise


def instrument_all(
    app: Any | None = None,
    engine: Any | None = None,
    tracer_provider: Any | None = None,
) -> None:
    """Instrument all available components.

    Args:
        app: FastAPI application instance (optional)
        engine: SQLAlchemy engine instance (optional)
        tracer_provider: Optional tracer provider
    """
    logger.info("Starting instrumentation of all components")

    # Instrument HTTP client
    instrument_httpx(tracer_provider)

    # Instrument Redis
    instrument_redis(tracer_provider)

    # Instrument database if engine provided
    if engine:
        instrument_database(engine, tracer_provider)

    # Instrument FastAPI if app provided
    if app:
        instrument_fastapi(app, tracer_provider)

    logger.info("All components instrumented successfully")


def uninstrument_all() -> None:
    """Uninstrument all components."""
    logger.info("Uninstrumenting all components")

    try:
        FastAPIInstrumentor.uninstrument()
        logger.info("FastAPI uninstrumented")
    except Exception as e:
        logger.warning(f"Failed to uninstrument FastAPI: {e}")

    try:
        HTTPXClientInstrumentor().uninstrument()
        logger.info("httpx uninstrumented")
    except Exception as e:
        logger.warning(f"Failed to uninstrument httpx: {e}")

    try:
        RedisInstrumentor().uninstrument()
        logger.info("Redis uninstrumented")
    except Exception as e:
        logger.warning(f"Failed to uninstrument Redis: {e}")

    try:
        SQLAlchemyInstrumentor().uninstrument()
        logger.info("Database uninstrumented")
    except Exception as e:
        logger.warning(f"Failed to uninstrument database: {e}")

    logger.info("All components uninstrumented")
