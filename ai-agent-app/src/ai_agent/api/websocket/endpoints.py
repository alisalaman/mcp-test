"""WebSocket endpoints for real-time communication."""

import asyncio
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from ai_agent.api.websocket.auth import websocket_auth
from ai_agent.api.websocket.connection_manager import manager
from ai_agent.api.websocket.event_handlers import event_handler
from ai_agent.config.settings import get_settings

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str | None = Query(None, description="Session ID to join"),
) -> None:
    """
    Main WebSocket endpoint for real-time communication.

    Query Parameters:
        - session_id: Optional session ID to join
        - token: Optional JWT token for authentication
    """
    settings = get_settings()

    # Check if WebSockets are enabled
    if not settings.features.enable_websockets:
        await websocket.close(code=1008, reason="WebSockets disabled")
        return

    try:
        # Authenticate the connection
        user_id, parsed_session_id = await websocket_auth.authenticate_websocket(
            websocket
        )

        # If session_id provided, check authorization
        if parsed_session_id:
            if not await websocket_auth.authorize_session_access(
                user_id, parsed_session_id
            ):
                await websocket.close(code=1008, reason="Access denied to session")
                return

        # Connect to WebSocket manager
        connection_id = await manager.connect(websocket, user_id, parsed_session_id)

        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "connected",
                "connection_id": connection_id,
                "user_id": user_id,
                "session_id": str(parsed_session_id) if parsed_session_id else None,
                "timestamp": asyncio.get_event_loop().time(),
            },
            connection_id,
        )

        # Send system notification
        await event_handler.handle_system_notification(
            "connection", f"Connected to WebSocket as {user_id}", user_id=user_id
        )

        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle the message
                await event_handler.handle_client_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON message"}, connection_id
                )
            except Exception as e:
                await manager.send_personal_message(
                    {"type": "error", "message": f"Error processing message: {str(e)}"},
                    connection_id,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if "connection_id" in locals():
            await manager.disconnect(connection_id)


@router.websocket("/session/{session_id}")
async def websocket_session_endpoint(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """
    WebSocket endpoint for a specific session.

    Path Parameters:
        - session_id: Session ID to join
    """
    try:
        # Parse session ID to validate format
        UUID(session_id)
    except ValueError:
        await websocket.close(code=1008, reason="Invalid session ID")
        return

    # Use the main endpoint with session_id
    await websocket_endpoint(websocket, session_id=session_id)


@router.get("/status")
async def websocket_status() -> dict[str, Any]:
    """Get WebSocket connection status."""
    return {
        "active_connections": manager.get_connection_count(),
        "user_connections": {
            user_id: manager.get_user_connection_count(user_id)
            for user_id in manager.user_connections.keys()
        },
        "session_connections": {
            session_id: manager.get_session_connection_count(UUID(session_id))
            for session_id in manager.session_connections.keys()
        },
    }
