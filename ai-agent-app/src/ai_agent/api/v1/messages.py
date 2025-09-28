"""Messages API endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_agent.api.dependencies import get_message_service, get_current_user
from ai_agent.core.messages.service import MessageService
from ai_agent.domain.models import Message, CreateMessageRequest, MessageRole

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post("/sessions/{session_id}/messages", response_model=Message, status_code=201)
async def create_message(
    session_id: UUID,
    request: CreateMessageRequest,
    message_service: Annotated[MessageService, Depends(get_message_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Message:
    """Create a new message in a session."""
    return await message_service.create_message(session_id, request)


@router.get("/sessions/{session_id}/messages", response_model=list[Message])
async def list_messages(
    session_id: UUID,
    message_service: Annotated[MessageService, Depends(get_message_service)],
    current_user: Annotated[str, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    role: Annotated[
        MessageRole | None, Query(description="Filter by message role")
    ] = None,
    search: Annotated[
        str | None, Query(description="Search in message content")
    ] = None,
) -> list[Message]:
    """List messages in a session with filtering."""
    messages: list[Message] = await message_service.list_messages(
        session_id=session_id,
        limit=limit,
        offset=offset,
        role=role,
        search=search,
    )
    return messages


@router.get("/{message_id}", response_model=Message)
async def get_message(
    message_id: UUID,
    message_service: Annotated[MessageService, Depends(get_message_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Message:
    """Get a specific message by ID."""
    message = await message_service.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.put("/{message_id}", response_model=Message)
async def update_message(
    message_id: UUID,
    request: CreateMessageRequest,
    message_service: Annotated[MessageService, Depends(get_message_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Message:
    """Update message content and metadata."""
    return await message_service.update_message(message_id, request)


@router.delete("/{message_id}", status_code=204)
async def delete_message(
    message_id: UUID,
    message_service: Annotated[MessageService, Depends(get_message_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete a specific message."""
    await message_service.delete_message(message_id)
