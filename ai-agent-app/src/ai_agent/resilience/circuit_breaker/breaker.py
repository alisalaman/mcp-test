"""Circuit breaker implementation for external service protection."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from ..exceptions import CircuitBreakerOpenException
from .config import CircuitBreakerConfig

logger = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics with rotation support."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0
    half_open_calls: int = 0
    state_changes: int = 0
    last_state_change: float | None = None
    # Rotation tracking
    _rotation_count: int = 0
    _max_rotation_count: int = 1000  # Reset metrics after 1000 operations


class CircuitBreaker:
    """Circuit breaker implementation for external services."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

    def _check_and_rotate_metrics(self) -> None:
        """Check if metrics need rotation and reset if necessary."""
        if self.metrics._rotation_count >= self.metrics._max_rotation_count:
            logger.info(
                "Rotating circuit breaker metrics",
                circuit_name=self.name,
                rotation_count=self.metrics._rotation_count,
            )
            # Reset counters but preserve state
            self.metrics.failure_count = 0
            self.metrics.success_count = 0
            self.metrics.total_requests = 0
            self.metrics.half_open_calls = 0
            self.metrics._rotation_count = 0
            # Keep timing information for state transitions

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        async with self._lock:
            await self._update_state()

            if self.state == CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker is open, rejecting request",
                    circuit_name=self.name,
                    failure_count=self.metrics.failure_count,
                    last_failure_time=self.metrics.last_failure_time,
                )
                raise CircuitBreakerOpenException(self.name)

            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.half_open_calls >= self.config.half_open_max_calls:
                    logger.warning(
                        "Half-open circuit has reached max calls, rejecting request",
                        circuit_name=self.name,
                        half_open_calls=self.metrics.half_open_calls,
                        max_calls=self.config.half_open_max_calls,
                    )
                    raise CircuitBreakerOpenException(self.name)

            self.metrics.total_requests += 1
            self.metrics._rotation_count += 1

            # Check if metrics need rotation
            self._check_and_rotate_metrics()

            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception as e:
                await self._record_failure(e)
                raise

    async def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions."""
        current_time = time.time()

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if (
                self.metrics.last_failure_time
                and (current_time - self.metrics.last_failure_time)
                >= self.config.recovery_timeout
            ):
                await self._transition_to_half_open()

        elif self.state == CircuitState.HALF_OPEN:
            # In half-open state, allow limited traffic for testing
            pass

        elif self.state == CircuitState.CLOSED:
            # Check if failure threshold has been exceeded
            if self.metrics.failure_count >= self.config.failure_threshold:
                await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()

        logger.warning(
            "Circuit breaker opening",
            circuit_name=self.name,
            failure_count=self.metrics.failure_count,
            failure_threshold=self.config.failure_threshold,
        )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.metrics.half_open_calls = 0

        logger.info(
            "Circuit breaker transitioning to half-open",
            circuit_name=self.name,
            recovery_timeout=self.config.recovery_timeout,
        )

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.metrics.failure_count = 0
        self.metrics.half_open_calls = 0

        logger.info(
            "Circuit breaker closing after successful test",
            circuit_name=self.name,
            success_count=self.metrics.success_count,
        )

    async def _record_success(self) -> None:
        """Record successful operation."""
        self.metrics.success_count += 1
        self.metrics.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Check if we have enough successes to close the circuit
            if self.metrics.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self.metrics.failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record failed operation."""
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Return to open state on failure during testing
            await self._transition_to_open()
            logger.warning(
                "Circuit breaker returning to open state after failure in half-open",
                circuit_name=self.name,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        elif (
            self.state == CircuitState.CLOSED
            and self.metrics.failure_count >= self.config.failure_threshold
        ):
            # Open circuit if failure threshold is reached
            await self._transition_to_open()
            logger.warning(
                "Circuit breaker opened due to failure threshold",
                circuit_name=self.name,
                failure_count=self.metrics.failure_count,
                threshold=self.config.failure_threshold,
            )

    def is_expected_exception(self, error: Exception) -> bool:
        """Check if the exception is expected by the circuit breaker.

        Args:
            error: Exception to check

        Returns:
            True if exception is expected
        """
        if not self.config.expected_exception:
            return True  # If no specific exceptions, consider all as expected

        error_type_name = type(error).__name__
        error_full_name = f"{type(error).__module__}.{type(error).__name__}"

        # Check both short name and full name
        return (
            error_type_name in self.config.expected_exception
            or error_full_name in self.config.expected_exception
        )

    def get_state_info(self) -> dict[str, Any]:
        """Get current circuit breaker state information.

        Returns:
            Dictionary with state information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "total_requests": self.metrics.total_requests,
            "half_open_calls": self.metrics.half_open_calls,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "state_changes": self.metrics.state_changes,
            "last_state_change": self.metrics.last_state_change,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        logger.info("Circuit breaker reset", circuit_name=self.name)

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        logger.warning("Circuit breaker forced open", circuit_name=self.name)

    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.metrics.state_changes += 1
        self.metrics.last_state_change = time.time()
        self.metrics.failure_count = 0
        self.metrics.half_open_calls = 0
        logger.info("Circuit breaker forced closed", circuit_name=self.name)
