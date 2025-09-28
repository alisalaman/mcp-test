"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import version from package
from . import __description__, __version__

# Create FastAPI application
app = FastAPI(
    title="AI Agent Application",
    description=__description__,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "timestamp": "2024-01-01T00:00:00Z",  # Will be replaced with actual timestamp
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
