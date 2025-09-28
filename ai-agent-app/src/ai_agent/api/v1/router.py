"""Main API router that combines all v1 endpoints."""

from fastapi import APIRouter

from ai_agent.api.v1 import agents, health, mcp_servers, messages, sessions, tools

# Create main v1 router
router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
router.include_router(sessions.router)
router.include_router(messages.router)
router.include_router(agents.router)
router.include_router(tools.router)
router.include_router(mcp_servers.router)
router.include_router(health.router)
