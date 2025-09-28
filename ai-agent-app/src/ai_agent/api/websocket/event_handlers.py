"""WebSocket event handlers for real-time communication."""

from datetime import datetime, UTC
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from ai_agent.api.websocket.connection_manager import manager
from ai_agent.config.settings import get_settings


class WebSocketEvent(BaseModel):
    """Base WebSocket event model."""

    type: str
    data: dict[str, Any]
    timestamp: str | None = None
    correlation_id: str | None = None

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class AgentExecutionEvent(WebSocketEvent):
    """Event for agent execution updates."""

    type: str = "agent_execution"
    data: dict[str, Any]  # Contains agent_id, session_id, status, progress, etc.


class SessionUpdateEvent(WebSocketEvent):
    """Event for session updates."""

    type: str = "session_update"
    data: dict[str, Any]  # Contains session_id, updates, etc.


class MessageEvent(WebSocketEvent):
    """Event for new messages."""

    type: str = "message"
    data: dict[str, Any]  # Contains message_id, session_id, content, etc.


class SystemNotificationEvent(WebSocketEvent):
    """Event for system notifications."""

    type: str = "system_notification"
    data: dict[str, Any]  # Contains notification type, message, etc.


class EventHandler:
    """Handles WebSocket events and message routing."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def handle_agent_execution_start(
        self, agent_id: UUID, session_id: UUID, user_id: str
    ) -> None:
        """Handle agent execution start event."""
        event = AgentExecutionEvent(
            data={
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "status": "started",
                "progress": 0,
                "message": "Agent execution started",
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

        # Send to user's other connections
        await manager.send_to_user(event.model_dump(), user_id)

    async def handle_agent_execution_progress(
        self,
        agent_id: UUID,
        session_id: UUID,
        user_id: str,
        progress: int,
        message: str,
    ) -> None:
        """Handle agent execution progress event."""
        event = AgentExecutionEvent(
            data={
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "status": "processing",
                "progress": progress,
                "message": message,
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

    async def handle_agent_execution_complete(
        self, agent_id: UUID, session_id: UUID, user_id: str, result: dict[str, Any]
    ) -> None:
        """Handle agent execution completion event."""
        event = AgentExecutionEvent(
            data={
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "status": "completed",
                "progress": 100,
                "result": result,
                "message": "Agent execution completed",
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

    async def handle_agent_execution_error(
        self, agent_id: UUID, session_id: UUID, user_id: str, error: str
    ) -> None:
        """Handle agent execution error event."""
        event = AgentExecutionEvent(
            data={
                "agent_id": str(agent_id),
                "session_id": str(session_id),
                "status": "error",
                "progress": 0,
                "error": error,
                "message": "Agent execution failed",
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

    async def handle_new_message(
        self, message_id: UUID, session_id: UUID, content: str, role: str
    ) -> None:
        """Handle new message event."""
        event = MessageEvent(
            data={
                "message_id": str(message_id),
                "session_id": str(session_id),
                "content": content,
                "role": role,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

    async def handle_session_update(
        self, session_id: UUID, updates: dict[str, Any]
    ) -> None:
        """Handle session update event."""
        event = SessionUpdateEvent(
            data={
                "session_id": str(session_id),
                "updates": updates,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Send to session participants
        await manager.send_to_session(event.model_dump(), session_id)

    async def handle_system_notification(
        self,
        notification_type: str,
        message: str,
        user_id: str | None = None,
        session_id: UUID | None = None,
    ) -> None:
        """Handle system notification event."""
        event = SystemNotificationEvent(
            data={
                "notification_type": notification_type,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        if user_id:
            # Send to specific user
            await manager.send_to_user(event.model_dump(), user_id)
        elif session_id:
            # Send to session participants
            await manager.send_to_session(event.model_dump(), session_id)
        else:
            # Broadcast to all users
            await manager.broadcast(event.model_dump())

    async def handle_client_message(
        self, connection_id: str, message: dict[str, Any]
    ) -> None:
        """Handle incoming message from client."""
        try:
            message_type = message.get("type")

            if message_type == "ping":
                # Respond to ping
                await manager.send_personal_message({"type": "pong"}, connection_id)

            elif message_type == "subscribe":
                # Handle subscription request
                subscription = message.get("subscription")
                if subscription:
                    await manager.subscribe(connection_id, subscription)
                    await manager.send_personal_message(
                        {
                            "type": "subscription_confirmed",
                            "subscription": subscription,
                        },
                        connection_id,
                    )

            elif message_type == "unsubscribe":
                # Handle unsubscription request
                subscription = message.get("subscription")
                if subscription:
                    await manager.unsubscribe(connection_id, subscription)
                    await manager.send_personal_message(
                        {
                            "type": "unsubscription_confirmed",
                            "subscription": subscription,
                        },
                        connection_id,
                    )

            elif message_type == "join_session":
                # Handle join session request
                session_id = message.get("session_id")
                if session_id:
                    # Update connection info with session
                    if connection_id in manager.connection_info:
                        manager.connection_info[connection_id].session_id = UUID(
                            session_id
                        )

                        # Add to session connections
                        session_key = str(session_id)
                        if session_key not in manager.session_connections:
                            manager.session_connections[session_key] = set()
                        manager.session_connections[session_key].add(connection_id)

                    await manager.send_personal_message(
                        {"type": "session_joined", "session_id": session_id},
                        connection_id,
                    )

            else:
                # Unknown message type
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                    },
                    connection_id,
                )

        except Exception as e:
            await manager.send_personal_message(
                {"type": "error", "message": f"Error processing message: {str(e)}"},
                connection_id,
            )


# Global event handler instance
event_handler = EventHandler()
