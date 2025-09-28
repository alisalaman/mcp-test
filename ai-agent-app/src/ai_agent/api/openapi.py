"""OpenAPI configuration for the AI Agent API."""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Create custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AI Agent API",
        version="1.0.0",
        description="""
        ## Production-ready AI Agent Application API

        This API provides comprehensive endpoints for managing AI agents, sessions,
        messages, and tools with built-in resilience patterns.

        ### Features
        - **Session Management**: Create and manage conversation sessions
        - **Message Handling**: CRUD operations for messages with full-text search
        - **Agent Execution**: Execute AI agents with streaming support
        - **Tool Integration**: Manage MCP server tools and capabilities
        - **Real-time**: WebSocket support for live updates
        - **Resilience**: Built-in retry logic and circuit breakers

        ### Authentication
        Use API keys in the `Authorization` header: `Bearer <your-api-key>`

        ### Rate Limiting
        - Default: 100 requests/minute
        - Authenticated: 1000 requests/minute
        - Premium: 5000 requests/minute

        ### Error Handling
        All errors follow standardized format with correlation IDs for tracing.
        """,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "Enter your API key as 'Bearer <api-key>'",
        },
        "JWTAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authenticated users",
        },
    }

    # Add global security requirement
    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"JWTAuth": []}]

    # Add common responses
    openapi_schema["components"]["responses"] = {
        "ValidationError": {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ValidationErrorResponse"}
                }
            },
        },
        "RateLimitError": {
            "description": "Rate Limit Exceeded",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            },
        },
        "UnauthorizedError": {
            "description": "Authentication Required",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            },
        },
        "NotFoundError": {
            "description": "Resource Not Found",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            },
        },
        "InternalServerError": {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            },
        },
    }

    # Add common parameters
    openapi_schema["components"]["parameters"] = {
        "LimitParam": {
            "name": "limit",
            "in": "query",
            "description": "Number of items to return",
            "required": False,
            "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
        "OffsetParam": {
            "name": "offset",
            "in": "query",
            "description": "Number of items to skip",
            "required": False,
            "schema": {"type": "integer", "minimum": 0, "default": 0},
        },
        "SearchParam": {
            "name": "search",
            "in": "query",
            "description": "Search term for filtering",
            "required": False,
            "schema": {"type": "string"},
        },
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema
