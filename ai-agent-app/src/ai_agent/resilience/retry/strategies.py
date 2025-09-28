"""Retry strategies and algorithms."""

import random
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import structlog

from .config import RetryConfig

logger = structlog.get_logger()


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""

    @abstractmethod
    async def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for the given attempt number."""
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff with jitter retry strategy."""

    async def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        # Calculate base exponential delay
        delay = config.base_delay * (config.multiplier ** (attempt - 1))

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter if enabled
        if config.jitter:
            # Add up to 25% jitter to prevent thundering herd
            jitter_factor = random.uniform(0.75, 1.25)
            delay *= jitter_factor

        return delay


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff retry strategy."""

    async def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate linear backoff delay."""
        delay = config.base_delay * attempt

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter if enabled
        if config.jitter:
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor

        return delay


class FixedDelayStrategy(RetryStrategy):
    """Fixed delay retry strategy."""

    async def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate fixed delay."""
        delay = config.base_delay

        # Add jitter if enabled
        if config.jitter:
            jitter_factor = random.uniform(0.9, 1.1)
            delay *= jitter_factor

        return delay


class RetryContext:
    """Context for retry operations."""

    def __init__(self, config: RetryConfig, strategy: RetryStrategy):
        self.config = config
        self.strategy = strategy
        self.attempt = 0
        self.start_time = datetime.now(UTC)
        self.last_error: Exception | None = None
        self.total_delay = 0.0

    async def should_retry(self, error: Exception) -> bool:
        """Determine if the operation should be retried."""
        self.attempt += 1
        self.last_error = error

        # Check if we've exceeded max attempts
        if self.attempt >= self.config.max_attempts:
            logger.warning(
                "Max retry attempts exceeded",
                attempt=self.attempt,
                max_attempts=self.config.max_attempts,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return False

        # Check if error is retryable
        error_type_name = f"{type(error).__module__}.{type(error).__name__}"
        if (
            self.config.retryable_errors
            and error_type_name not in self.config.retryable_errors
        ):
            logger.info(
                "Error not in retryable list",
                error_type=error_type_name,
                retryable_errors=self.config.retryable_errors,
            )
            return False

        return True

    async def get_delay(self) -> float:
        """Get delay for current attempt."""
        delay = await self.strategy.calculate_delay(self.attempt, self.config)
        self.total_delay += delay
        return delay

    def get_metrics(self) -> dict[str, Any]:
        """Get retry metrics."""
        duration = (datetime.now(UTC) - self.start_time).total_seconds()
        return {
            "attempts": self.attempt,
            "total_delay": self.total_delay,
            "duration": duration,
            "last_error": str(self.last_error) if self.last_error else None,
            "error_type": type(self.last_error).__name__ if self.last_error else None,
        }
