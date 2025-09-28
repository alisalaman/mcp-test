"""Agents API endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ai_agent.api.dependencies import get_agent_service, get_current_user
from ai_agent.core.agents.service import AgentService, CreateAgentRequest
from ai_agent.domain.models import (
    Agent,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentStatus,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/", response_model=Agent, status_code=201)
async def create_agent(
    request: CreateAgentRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Agent:
    """Create a new AI agent."""
    return await agent_service.create_agent(request)


@router.get("/", response_model=list[Agent])
async def list_agents(
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
    status: Annotated[
        AgentStatus | None, Query(description="Filter by agent status")
    ] = None,
    search: Annotated[
        str | None, Query(description="Search in agent name/description")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Agent]:
    """List AI agents with filtering."""
    agents: list[Agent] = await agent_service.list_agents(
        status=status,
        search=search,
        limit=limit,
        offset=offset,
    )
    return agents


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: UUID,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Agent:
    """Get specific agent by ID."""
    agent = await agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: UUID,
    request: CreateAgentRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Agent:
    """Update agent configuration."""
    return await agent_service.update_agent(agent_id, request)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: UUID,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete an agent."""
    await agent_service.delete_agent(agent_id)


@router.post("/{agent_id}/execute", response_model=AgentExecutionResponse)
async def execute_agent(
    agent_id: UUID,
    request: AgentExecutionRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> AgentExecutionResponse:
    """Execute agent with a message."""
    return await agent_service.execute_agent(agent_id, request)


@router.post("/{agent_id}/execute/stream")
async def execute_agent_stream(
    agent_id: UUID,
    request: AgentExecutionRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    """Execute agent with streaming response."""
    return StreamingResponse(
        agent_service.execute_agent_stream(agent_id, request),
        media_type="text/event-stream",
    )
