"""Message queuing system for offline WebSocket clients."""

import asyncio
from datetime import datetime, timedelta, UTC
from typing import Any

from pydantic import BaseModel
import structlog

from ai_agent.api.websocket.connection_manager import manager
from ai_agent.config.settings import get_settings

logger = structlog.get_logger()


class QueuedMessage(BaseModel):
    """A message queued for delivery to offline clients."""

    message_id: str
    user_id: str
    session_id: str | None = None
    subscription: str | None = None
    message: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None = None
    priority: int = 0  # Higher number = higher priority
    retry_count: int = 0
    max_retries: int = 3


class MessageQueue:
    """Manages message queuing for offline clients."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.queue: list[QueuedMessage] = []
        self.user_queues: dict[str, list[QueuedMessage]] = {}
        self.session_queues: dict[str, list[QueuedMessage]] = {}
        self.subscription_queues: dict[str, list[QueuedMessage]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._delivery_task: asyncio.Task[None] | None = None

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background tasks for queue management."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())

        if not self._delivery_task or self._delivery_task.done():
            self._delivery_task = asyncio.create_task(self._deliver_queued_messages())

    async def queue_message(
        self,
        user_id: str,
        message: dict[str, Any],
        session_id: str | None = None,
        subscription: str | None = None,
        priority: int = 0,
        ttl_seconds: int = 3600,  # 1 hour default TTL
    ) -> str:
        """
        Queue a message for delivery to a user.

        Args:
            user_id: Target user ID
            message: Message to queue
            session_id: Optional session ID
            subscription: Optional subscription topic
            priority: Message priority (higher = more important)
            ttl_seconds: Time to live in seconds

        Returns:
            str: Message ID
        """
        message_id = f"msg_{asyncio.get_event_loop().time()}_{user_id}"
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None

        queued_message = QueuedMessage(
            message_id=message_id,
            user_id=user_id,
            session_id=session_id,
            subscription=subscription,
            message=message,
            created_at=now,
            expires_at=expires_at,
            priority=priority,
        )

        async with self._lock:
            # Add to main queue
            self.queue.append(queued_message)

            # Add to user-specific queue
            if user_id not in self.user_queues:
                self.user_queues[user_id] = []
            self.user_queues[user_id].append(queued_message)

            # Add to session-specific queue if applicable
            if session_id:
                if session_id not in self.session_queues:
                    self.session_queues[session_id] = []
                self.session_queues[session_id].append(queued_message)

            # Add to subscription-specific queue if applicable
            if subscription:
                if subscription not in self.subscription_queues:
                    self.subscription_queues[subscription] = []
                self.subscription_queues[subscription].append(queued_message)

        return message_id

    async def get_queued_messages(
        self,
        user_id: str,
        limit: int = 50,
        session_id: str | None = None,
        subscription: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get queued messages for a user.

        Args:
            user_id: User ID
            limit: Maximum number of messages to return
            session_id: Optional session filter
            subscription: Optional subscription filter

        Returns:
            List of queued messages
        """
        async with self._lock:
            user_messages = self.user_queues.get(user_id, [])

            # Filter by session and subscription if specified
            filtered_messages = []
            for msg in user_messages:
                if session_id and msg.session_id != session_id:
                    continue
                if subscription and msg.subscription != subscription:
                    continue
                filtered_messages.append(msg)

            # Sort by priority and creation time
            filtered_messages.sort(key=lambda x: (-x.priority, x.created_at))

            # Return limited results
            return [msg.model_dump() for msg in filtered_messages[:limit]]

    async def mark_message_delivered(self, message_id: str) -> bool:
        """
        Mark a message as delivered and remove it from queues.

        Args:
            message_id: Message ID to mark as delivered

        Returns:
            bool: True if message was found and removed
        """
        async with self._lock:
            # Find the message
            message_to_remove = None
            for msg in self.queue:
                if msg.message_id == message_id:
                    message_to_remove = msg
                    break

            if not message_to_remove:
                return False

            # Remove from all queues
            self.queue.remove(message_to_remove)

            if message_to_remove.user_id in self.user_queues:
                self.user_queues[message_to_remove.user_id].remove(message_to_remove)
                if not self.user_queues[message_to_remove.user_id]:
                    del self.user_queues[message_to_remove.user_id]

            if (
                message_to_remove.session_id
                and message_to_remove.session_id in self.session_queues
            ):
                self.session_queues[message_to_remove.session_id].remove(
                    message_to_remove
                )
                if not self.session_queues[message_to_remove.session_id]:
                    del self.session_queues[message_to_remove.session_id]

            if (
                message_to_remove.subscription
                and message_to_remove.subscription in self.subscription_queues
            ):
                self.subscription_queues[message_to_remove.subscription].remove(
                    message_to_remove
                )
                if not self.subscription_queues[message_to_remove.subscription]:
                    del self.subscription_queues[message_to_remove.subscription]

            return True

    async def _cleanup_expired_messages(self) -> None:
        """Background task to clean up expired messages."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                now = datetime.now(UTC)
                expired_messages = []

                async with self._lock:
                    for msg in self.queue:
                        if msg.expires_at and msg.expires_at < now:
                            expired_messages.append(msg)

                    # Remove expired messages
                    for msg in expired_messages:
                        await self.mark_message_delivered(msg.message_id)

                if expired_messages:
                    logger.debug(
                        "Cleaned up expired messages",
                        count=len(expired_messages),
                    )

            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))

    async def _deliver_queued_messages(self) -> None:
        """Background task to deliver queued messages to reconnected users."""
        while True:
            try:
                await asyncio.sleep(5)  # Run every 5 seconds

                # Get all active user connections
                active_users = set(manager.user_connections.keys())

                async with self._lock:
                    # Check for queued messages for active users
                    for user_id in active_users:
                        if user_id in self.user_queues:
                            messages = self.user_queues[user_id]

                            # Deliver up to 10 messages per user per cycle
                            for msg in messages[:10]:
                                try:
                                    # Try to deliver the message
                                    await manager.send_to_user(msg.message, user_id)

                                    # Mark as delivered
                                    await self.mark_message_delivered(msg.message_id)

                                except Exception as e:
                                    # Increment retry count
                                    msg.retry_count += 1

                                    # Remove if max retries exceeded
                                    if msg.retry_count >= msg.max_retries:
                                        await self.mark_message_delivered(
                                            msg.message_id
                                        )
                                        logger.warning(
                                            "Removed message after max retries",
                                            message_id=msg.message_id,
                                            max_retries=msg.max_retries,
                                        )

                                    logger.warning(
                                        "Failed to deliver message",
                                        message_id=msg.message_id,
                                        error=str(e),
                                    )

            except Exception as e:
                logger.error("Error in delivery task", error=str(e))

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            return {
                "total_messages": len(self.queue),
                "user_queues": len(self.user_queues),
                "session_queues": len(self.session_queues),
                "subscription_queues": len(self.subscription_queues),
                "messages_by_user": {
                    user_id: len(messages)
                    for user_id, messages in self.user_queues.items()
                },
                "messages_by_session": {
                    session_id: len(messages)
                    for session_id, messages in self.session_queues.items()
                },
            }

    async def shutdown(self) -> None:
        """Shutdown the message queue and stop background tasks."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        if self._delivery_task and not self._delivery_task.done():
            self._delivery_task.cancel()


# Global message queue instance
message_queue = MessageQueue()
