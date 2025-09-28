"""WebSocket authentication and authorization."""

from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer
from jose import jwt

from ai_agent.config.settings import get_settings


class WebSocketAuth:
    """Handles WebSocket authentication and authorization."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.security = HTTPBearer(auto_error=False)

    async def authenticate_websocket(
        self, websocket: WebSocket
    ) -> tuple[str, UUID | None]:
        """
        Authenticate WebSocket connection.

        Returns:
            tuple: (user_id, session_id) or raises WebSocketDisconnect
        """
        # Get query parameters
        query_params = websocket.query_params
        token = query_params.get("token")
        session_id = query_params.get("session_id")

        # Parse session_id if provided
        parsed_session_id = None
        if session_id:
            try:
                parsed_session_id = UUID(session_id)
            except ValueError:
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION, reason="Invalid session ID"
                )
                raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

        # Check for API key in headers (if available)
        api_key = websocket.headers.get("x-api-key")
        if api_key:
            # In production, validate API key against database
            user_id = f"api_user_{api_key[:8]}"
            return user_id, parsed_session_id

        # Check for JWT token
        if token:
            try:
                # Decode JWT token
                payload = jwt.decode(
                    token,
                    self.settings.security.secret_key,
                    algorithms=[self.settings.security.algorithm],
                )
                user_id = payload.get("sub")
                if user_id:
                    return user_id, parsed_session_id
            except jwt.ExpiredSignatureError:
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION, reason="Token expired"
                )
                raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)
            except jwt.InvalidTokenError:
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
                )
                raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

        # For development, allow anonymous access
        if self.settings.is_development:
            return "anonymous_user", parsed_session_id

        # In production, require authentication
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
        )
        raise WebSocketDisconnect(code=status.WS_1008_POLICY_VIOLATION)

    async def authorize_session_access(self, user_id: str, session_id: UUID) -> bool:
        """
        Check if user has access to a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            bool: True if user has access, False otherwise
        """
        # In a real implementation, this would check the database
        # For now, we'll allow access for development
        if self.settings.is_development:
            return True

        # TODO: Implement proper session authorization
        # This would involve checking if the user owns the session or has permission
        return True

    async def authorize_subscription(self, user_id: str, subscription: str) -> bool:
        """
        Check if user can subscribe to a topic.

        Args:
            user_id: User identifier
            subscription: Subscription topic

        Returns:
            bool: True if user can subscribe, False otherwise
        """
        # In a real implementation, this would check user permissions
        # For now, we'll allow all subscriptions for development
        if self.settings.is_development:
            return True

        # TODO: Implement proper subscription authorization
        # This would involve checking user roles and permissions
        return True


# Global WebSocket auth instance
websocket_auth = WebSocketAuth()
