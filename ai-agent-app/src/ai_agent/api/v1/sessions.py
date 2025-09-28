"""Sessions API endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_agent.api.dependencies import get_session_service, get_current_user
from ai_agent.core.sessions.service import SessionService
from ai_agent.domain.models import Session, CreateSessionRequest

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=Session, status_code=201)
async def create_session(
    request: CreateSessionRequest,
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Session:
    """
    Create a new conversation session.

    - **user_id**: Optional user identifier
    - **title**: Optional session title
    - **metadata**: Additional session metadata
    """
    return await session_service.create_session(request)


@router.get("/", response_model=list[Session])
async def list_sessions(
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
    user_id: Annotated[str | None, Query(description="Filter by user ID")] = None,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Number of sessions to return")
    ] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of sessions to skip")] = 0,
    sort_by: Annotated[
        str, Query(regex="^(created_at|updated_at|last_activity)$")
    ] = "last_activity",
    sort_order: Annotated[str, Query(regex="^(asc|desc)$")] = "desc",
) -> list[Session]:
    """
    List conversation sessions with filtering and pagination.

    - **user_id**: Filter sessions by user
    - **limit**: Maximum number of sessions to return (1-100)
    - **offset**: Number of sessions to skip for pagination
    - **sort_by**: Field to sort by (created_at, updated_at, last_activity)
    - **sort_order**: Sort order (asc, desc)
    """
    sessions: list[Session] = await session_service.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return sessions


@router.get("/{session_id}", response_model=Session)
async def get_session(
    session_id: UUID,
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Session:
    """Get a specific session by ID."""
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.put("/{session_id}", response_model=Session)
async def update_session(
    session_id: UUID,
    request: CreateSessionRequest,
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> Session:
    """Update session metadata."""
    return await session_service.update_session(session_id, request)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete a session and all its messages."""
    await session_service.delete_session(session_id)


@router.post("/bulk-create", response_model=list[Session], status_code=201)
async def bulk_create_sessions(
    requests: list[CreateSessionRequest],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> list[Session]:
    """Create multiple sessions in batch."""
    sessions: list[Session] = await session_service.bulk_create_sessions(requests)
    return sessions


@router.delete("/bulk-delete", status_code=204)
async def bulk_delete_sessions(
    session_ids: list[UUID],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> None:
    """Delete multiple sessions in batch."""
    await session_service.bulk_delete_sessions(session_ids)
