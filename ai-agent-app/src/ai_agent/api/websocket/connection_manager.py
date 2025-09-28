"""WebSocket connection manager for real-time communication."""

import asyncio
import json
import uuid
from typing import Any
from uuid import UUID

from fastapi import WebSocket
from pydantic import BaseModel

from ai_agent.config.settings import get_settings


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""

    connection_id: str
    user_id: str
    session_id: UUID | None = None
    connected_at: float
    last_ping: float
    subscriptions: set[str] = set()


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.connection_info: dict[str, ConnectionInfo] = {}
        self.user_connections: dict[str, set[str]] = {}  # user_id -> connection_ids
        self.session_connections: dict[str, set[str]] = (
            {}
        )  # session_id -> connection_ids
        self.subscription_connections: dict[str, set[str]] = (
            {}
        )  # subscription -> connection_ids
        self._lock = asyncio.Lock()
        self.settings = get_settings()

    async def connect(
        self, websocket: WebSocket, user_id: str, session_id: UUID | None = None
    ) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        current_time = asyncio.get_event_loop().time()

        # Store connection
        self.active_connections[connection_id] = websocket
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            session_id=session_id,
            connected_at=current_time,
            last_ping=current_time,
        )

        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)

        # Add to session connections if session_id provided
        if session_id:
            session_key = str(session_id)
            if session_key not in self.session_connections:
                self.session_connections[session_key] = set()
            self.session_connections[session_key].add(connection_id)

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket connection."""
        async with self._lock:
            if connection_id in self.active_connections:
                # Remove from active connections
                del self.active_connections[connection_id]

                # Get connection info
                if connection_id in self.connection_info:
                    info = self.connection_info[connection_id]

                    # Remove from user connections
                    if info.user_id in self.user_connections:
                        self.user_connections[info.user_id].discard(connection_id)
                        if not self.user_connections[info.user_id]:
                            del self.user_connections[info.user_id]

                    # Remove from session connections
                    if info.session_id:
                        session_key = str(info.session_id)
                        if session_key in self.session_connections:
                            self.session_connections[session_key].discard(connection_id)
                            if not self.session_connections[session_key]:
                                del self.session_connections[session_key]

                    # Remove from subscriptions
                    for subscription in info.subscriptions:
                        if subscription in self.subscription_connections:
                            self.subscription_connections[subscription].discard(
                                connection_id
                            )
                            if not self.subscription_connections[subscription]:
                                del self.subscription_connections[subscription]

                    # Remove connection info
                    del self.connection_info[connection_id]

    async def send_personal_message(
        self, message: dict[str, Any], connection_id: str
    ) -> None:
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)

    async def send_to_user(self, message: dict[str, Any], user_id: str) -> None:
        """Send a message to all connections for a user."""
        if user_id in self.user_connections:
            for connection_id in list(self.user_connections[user_id]):
                await self.send_personal_message(message, connection_id)

    async def send_to_session(self, message: dict[str, Any], session_id: UUID) -> None:
        """Send a message to all connections for a session."""
        session_key = str(session_id)
        if session_key in self.session_connections:
            for connection_id in list(self.session_connections[session_key]):
                await self.send_personal_message(message, connection_id)

    async def send_to_subscription(
        self, message: dict[str, Any], subscription: str
    ) -> None:
        """Send a message to all connections subscribed to a topic."""
        if subscription in self.subscription_connections:
            for connection_id in list(self.subscription_connections[subscription]):
                await self.send_personal_message(message, connection_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all active connections."""
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)

    async def subscribe(self, connection_id: str, subscription: str) -> None:
        """Subscribe a connection to a topic."""
        async with self._lock:
            if connection_id in self.connection_info:
                self.connection_info[connection_id].subscriptions.add(subscription)

                if subscription not in self.subscription_connections:
                    self.subscription_connections[subscription] = set()
                self.subscription_connections[subscription].add(connection_id)

    async def unsubscribe(self, connection_id: str, subscription: str) -> None:
        """Unsubscribe a connection from a topic."""
        async with self._lock:
            if connection_id in self.connection_info:
                self.connection_info[connection_id].subscriptions.discard(subscription)

                if subscription in self.subscription_connections:
                    self.subscription_connections[subscription].discard(connection_id)
                    if not self.subscription_connections[subscription]:
                        del self.subscription_connections[subscription]

    async def ping_connection(self, connection_id: str) -> None:
        """Send a ping to a connection to check if it's alive."""
        if connection_id in self.active_connections:
            try:
                await self.send_personal_message({"type": "ping"}, connection_id)
                if connection_id in self.connection_info:
                    self.connection_info[connection_id].last_ping = (
                        asyncio.get_event_loop().time()
                    )
            except Exception:
                await self.disconnect(connection_id)

    async def cleanup_stale_connections(self) -> None:
        """Remove stale connections that haven't responded to pings."""
        current_time = asyncio.get_event_loop().time()
        timeout = 30  # 30 seconds timeout

        stale_connections = []
        for connection_id, info in self.connection_info.items():
            if current_time - info.last_ping > timeout:
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            await self.disconnect(connection_id)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_user_connection_count(self, user_id: str) -> int:
        """Get the number of connections for a user."""
        return len(self.user_connections.get(user_id, set()))

    def get_session_connection_count(self, session_id: UUID) -> int:
        """Get the number of connections for a session."""
        return len(self.session_connections.get(str(session_id), set()))


# Global WebSocket manager instance
manager = WebSocketManager()
