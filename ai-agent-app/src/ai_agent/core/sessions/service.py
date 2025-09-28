"""Session service for managing conversation sessions."""

from uuid import UUID

from ai_agent.domain.models import Session, CreateSessionRequest
from ai_agent.infrastructure.database.base import Repository


class SessionService:
    """Service for managing conversation sessions."""

    def __init__(self, repository: Repository, current_user: str):
        self.repository = repository
        self.current_user = current_user

    async def create_session(self, request: CreateSessionRequest) -> Session:
        """Create a new session."""
        session = Session(
            user_id=request.user_id or self.current_user,
            title=request.title,
            metadata=request.metadata,
        )
        return await self.repository.create_session(session)

    async def get_session(self, session_id: UUID) -> Session | None:
        """Get session by ID."""
        return await self.repository.get_session(session_id)

    async def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "last_activity",
        sort_order: str = "desc",
    ) -> list[Session]:
        """List sessions with filtering and pagination."""
        # Use the provided user_id or current user
        filter_user_id = user_id or self.current_user
        sessions: list[Session] = await self.repository.list_sessions(
            user_id=filter_user_id,
            limit=limit,
            offset=offset,
        )
        return sessions

    async def update_session(
        self, session_id: UUID, request: CreateSessionRequest
    ) -> Session:
        """Update session metadata."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        # Update fields
        session.title = request.title
        session.metadata.update(request.metadata)

        return await self.repository.update_session(session)

    async def delete_session(self, session_id: UUID) -> None:
        """Delete a session and all its messages."""
        await self.repository.delete_session(session_id)

    async def bulk_create_sessions(
        self, requests: list[CreateSessionRequest]
    ) -> list[Session]:
        """Create multiple sessions in batch."""
        sessions = []
        for request in requests:
            session = await self.create_session(request)
            sessions.append(session)
        return sessions

    async def bulk_delete_sessions(self, session_ids: list[UUID]) -> None:
        """Delete multiple sessions in batch."""
        for session_id in session_ids:
            await self.delete_session(session_id)
