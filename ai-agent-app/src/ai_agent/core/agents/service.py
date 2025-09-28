"""Agent service for managing AI agents and execution."""

import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ai_agent.domain.models import (
    Agent,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentStatus,
    Message,
    MessageRole,
)
from ai_agent.infrastructure.database.base import Repository


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""

    name: str
    description: str | None = None
    system_prompt: str | None = None
    llm_config: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentService:
    """Service for managing AI agents and execution."""

    def __init__(self, repository: Repository, current_user: str):
        self.repository = repository
        self.current_user = current_user

    async def create_agent(self, request: CreateAgentRequest) -> Agent:
        """Create a new AI agent."""
        agent = Agent(
            name=request.name,
            description=request.description,
            system_prompt=request.system_prompt,
            llm_config=request.llm_config,
            tools=request.tools,
            metadata=request.metadata,
        )
        return await self.repository.create_agent(agent)

    async def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        return await self.repository.get_agent(agent_id)

    async def list_agents(
        self,
        status: AgentStatus | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Agent]:
        """List agents with filtering."""
        # Get all agents from repository
        agents: list[Agent] = await self.repository.list_agents(
            limit=limit, offset=offset
        )

        # Apply filtering
        if status:
            agents = [agent for agent in agents if agent.status == status]

        if search:
            search_lower = search.lower()
            agents = [
                agent
                for agent in agents
                if search_lower in agent.name.lower()
                or (agent.description and search_lower in agent.description.lower())
            ]

        return agents

    async def update_agent(self, agent_id: UUID, request: CreateAgentRequest) -> Agent:
        """Update agent configuration."""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")

        # Update fields
        agent.name = request.name
        agent.description = request.description
        agent.system_prompt = request.system_prompt
        agent.llm_config = request.llm_config
        agent.tools = request.tools
        agent.metadata.update(request.metadata)

        return await self.repository.update_agent(agent)

    async def delete_agent(self, agent_id: UUID) -> None:
        """Delete an agent."""
        await self.repository.delete_agent(agent_id)

    async def execute_agent(
        self, agent_id: UUID, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Execute agent with a message."""
        start_time = time.time()

        # Get agent
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")

        # Update agent status
        agent.status = AgentStatus.PROCESSING
        await self.repository.update_agent(agent)

        try:
            # Create user message
            await self.repository.create_message(
                Message(
                    session_id=request.session_id,
                    role=MessageRole.USER,
                    content=request.message,
                    metadata=request.metadata,
                )
            )

            # Simulate agent execution (in real implementation, this would use LangGraph)
            # For now, create a simple response
            response_content = f"Agent {agent.name} processed: {request.message}"

            # Create agent response message
            agent_message = await self.repository.create_message(
                Message(
                    session_id=request.session_id,
                    role=MessageRole.ASSISTANT,
                    content=response_content,
                    metadata={"agent_id": str(agent_id)},
                )
            )

            # Update agent status
            agent.status = AgentStatus.COMPLETED
            await self.repository.update_agent(agent)

            execution_time = int((time.time() - start_time) * 1000)

            return AgentExecutionResponse(
                session_id=request.session_id,
                message_id=agent_message.id,
                content=response_content,
                metadata={"agent_id": str(agent_id)},
                execution_time_ms=execution_time,
            )

        except Exception:
            # Update agent status on error
            agent.status = AgentStatus.ERROR
            await self.repository.update_agent(agent)
            raise

    async def execute_agent_stream(
        self, agent_id: UUID, request: AgentExecutionRequest
    ) -> AsyncGenerator[str, None]:
        """Execute agent with streaming response."""
        # This would implement streaming response
        # For now, return a simple generator
        response = await self.execute_agent(agent_id, request)
        yield f"data: {response.model_dump_json()}\n\n"
