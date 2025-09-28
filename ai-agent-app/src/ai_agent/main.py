"""FastAPI application entry point."""

from datetime import datetime, UTC

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from .api.rate_limiting import rate_limit_exceeded_handler

# Import version from package
from . import __description__, __version__

# Import API components
from .api.v1.router import router as v1_router
from .api.websocket.endpoints import router as websocket_router
from .api.error_handlers import (
    authentication_exception_handler,
    authorization_exception_handler,
    circuit_breaker_exception_handler,
    external_service_exception_handler,
    general_exception_handler,
    rate_limit_exception_handler,
    timeout_exception_handler,
    unexpected_exception_handler,
    validation_exception_handler,
    validation_exception_handler_custom,
)
from .api.middleware import (
    CorrelationIDMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from .api.openapi import custom_openapi
from .api.rate_limiting import limiter
from .config.settings import get_settings
from .domain.exceptions import (
    AIAgentException,
    AuthenticationException,
    AuthorizationException,
    CircuitBreakerOpenException,
    ExternalServiceException,
    RateLimitException,
    TimeoutException,
    ValidationException,
)

# Get settings
settings = get_settings()

# Rate limiter is imported from api.rate_limiting module

# Create FastAPI application
app = FastAPI(
    title="AI Agent Application",
    description=__description__,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add middleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=settings.security.cors_methods,
    allow_headers=settings.security.cors_headers,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationException, validation_exception_handler_custom)
app.add_exception_handler(AuthenticationException, authentication_exception_handler)
app.add_exception_handler(AuthorizationException, authorization_exception_handler)
app.add_exception_handler(RateLimitException, rate_limit_exception_handler)
app.add_exception_handler(ExternalServiceException, external_service_exception_handler)
app.add_exception_handler(
    CircuitBreakerOpenException, circuit_breaker_exception_handler
)
app.add_exception_handler(TimeoutException, timeout_exception_handler)
app.add_exception_handler(AIAgentException, general_exception_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)

# Include API routers
app.include_router(v1_router)
app.include_router(websocket_router)

# Set custom OpenAPI schema
app.openapi_schema = custom_openapi(app)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "AI Agent Application",
        "version": __version__,
        "status": "running",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    """Main entry point for production deployment."""
    import uvicorn

    uvicorn.run(
        "ai_agent.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )


def dev_main() -> None:
    """Development entry point with hot reload."""
    import uvicorn

    uvicorn.run(
        "ai_agent.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug",
        access_log=True,
    )


if __name__ == "__main__":
    dev_main()
