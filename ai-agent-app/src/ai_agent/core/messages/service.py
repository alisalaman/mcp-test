"""Message service for managing conversation messages."""

from uuid import UUID

from ai_agent.domain.models import Message, CreateMessageRequest, MessageRole
from ai_agent.infrastructure.database.base import Repository


class MessageService:
    """Service for managing conversation messages."""

    def __init__(self, repository: Repository, current_user: str):
        self.repository = repository
        self.current_user = current_user

    async def create_message(
        self, session_id: UUID, request: CreateMessageRequest
    ) -> Message:
        """Create a new message in a session."""
        # Verify session exists and user has access
        session = await self.repository.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        if session.user_id and session.user_id != self.current_user:
            raise ValueError("Access denied to session")

        message = Message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata,
        )
        return await self.repository.create_message(message)

    async def get_message(self, message_id: UUID) -> Message | None:
        """Get message by ID."""
        return await self.repository.get_message(message_id)

    async def list_messages(
        self,
        session_id: UUID,
        limit: int = 50,
        offset: int = 0,
        role: MessageRole | None = None,
        search: str | None = None,
    ) -> list[Message]:
        """List messages in a session with filtering."""
        # Verify session access
        session = await self.repository.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        if session.user_id and session.user_id != self.current_user:
            raise ValueError("Access denied to session")

        # Get messages from repository
        messages: list[Message] = await self.repository.get_messages_by_session(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

        # Apply filtering
        if role:
            messages = [msg for msg in messages if msg.role == role]

        if search:
            search_lower = search.lower()
            messages = [msg for msg in messages if search_lower in msg.content.lower()]

        return messages

    async def update_message(
        self, message_id: UUID, request: CreateMessageRequest
    ) -> Message:
        """Update message content and metadata."""
        message = await self.get_message(message_id)
        if not message:
            raise ValueError("Message not found")

        # Verify session access
        session = await self.repository.get_session(message.session_id)
        if not session:
            raise ValueError("Session not found")

        if session.user_id and session.user_id != self.current_user:
            raise ValueError("Access denied to message")

        # Update fields
        message.content = request.content
        message.metadata.update(request.metadata)

        return await self.repository.update_message(message)

    async def delete_message(self, message_id: UUID) -> None:
        """Delete a specific message."""
        message = await self.get_message(message_id)
        if not message:
            raise ValueError("Message not found")

        # Verify session access
        session = await self.repository.get_session(message.session_id)
        if not session:
            raise ValueError("Session not found")

        if session.user_id and session.user_id != self.current_user:
            raise ValueError("Access denied to message")

        await self.repository.delete_message(message_id)
