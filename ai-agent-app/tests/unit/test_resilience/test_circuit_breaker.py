"""Tests for circuit breaker functionality."""

import time

import pytest

from ai_agent.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerSettings,
    CircuitState,
)
from ai_agent.resilience.exceptions import CircuitBreakerOpenException


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def circuit_config(self):
        """Create circuit breaker config for testing."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.1,  # Very fast for testing
            success_threshold=2,
            half_open_max_calls=2,
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_config):
        """Create circuit breaker for testing."""
        return CircuitBreaker("test_service", circuit_config)

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_service"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.failure_count == 0
        assert circuit_breaker.metrics.success_count == 0

    @pytest.mark.asyncio
    async def test_successful_call_closed_state(self, circuit_breaker):
        """Test successful call in closed state."""

        async def test_func():
            return "success"

        result = await circuit_breaker.call(test_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_failure_in_closed_state(self, circuit_breaker):
        """Test failure in closed state."""

        async def test_func():
            raise Exception("Test failure")

        with pytest.raises(Exception, match="Test failure"):
            await circuit_breaker.call(test_func)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker):
        """Test circuit opens after failure threshold."""

        async def test_func():
            raise ValueError("Test failure")

        # Cause failures up to threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(test_func)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.metrics.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, circuit_breaker):
        """Test open circuit blocks new calls."""
        # Open the circuit
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.metrics.failure_count = 3

        async def test_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.call(test_func)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.metrics.last_failure_time = time.time() - 0.2  # Past timeout

        # Update state (this would normally happen in call)
        await circuit_breaker._update_state()
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test successful call in half-open closes circuit."""
        # Set to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.metrics.success_count = 0

        async def test_func():
            return "success"

        # Make enough successful calls to meet the success threshold
        for _ in range(2):  # success_threshold is 2
            result = await circuit_breaker.call(test_func)
            assert result == "success"

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test failure in half-open reopens circuit."""
        # Set to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN

        async def test_func():
            raise Exception("Test failure")

        with pytest.raises(Exception, match="Test failure"):
            await circuit_breaker.call(test_func)

        assert circuit_breaker.state == CircuitState.OPEN

    def test_is_expected_exception(self, circuit_breaker):
        """Test expected exception checking."""
        # Test with no expected exceptions (should return True)
        assert circuit_breaker.is_expected_exception(Exception("test")) is True

        # Test with specific expected exceptions
        circuit_breaker.config.expected_exception = ["ValueError", "RuntimeError"]
        assert circuit_breaker.is_expected_exception(ValueError("test")) is True
        assert circuit_breaker.is_expected_exception(RuntimeError("test")) is True
        assert circuit_breaker.is_expected_exception(Exception("test")) is False

    def test_get_state_info(self, circuit_breaker):
        """Test getting circuit breaker state info."""
        info = circuit_breaker.get_state_info()
        assert info["name"] == "test_service"
        assert info["state"] == "closed"
        assert "failure_count" in info
        assert "success_count" in info

    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker."""
        # Set some state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.metrics.failure_count = 5

        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.failure_count == 0

    def test_force_open(self, circuit_breaker):
        """Test forcing circuit breaker open."""
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN

    def test_force_close(self, circuit_breaker):
        """Test forcing circuit breaker closed."""
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.force_close()
        assert circuit_breaker.state == CircuitState.CLOSED


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""

    @pytest.fixture
    def circuit_settings(self):
        """Create circuit breaker settings for testing."""
        return CircuitBreakerSettings(
            llm_failure_threshold=3,
            llm_recovery_timeout=0.1,
            db_failure_threshold=2,
            db_recovery_timeout=0.1,
        )

    @pytest.fixture
    def circuit_manager(self, circuit_settings):
        """Create circuit breaker manager for testing."""
        return CircuitBreakerManager(circuit_settings)

    def test_circuit_manager_initialization(self, circuit_manager):
        """Test circuit manager initialization."""
        assert circuit_manager is not None
        assert circuit_manager.settings is not None

    def test_get_breaker_creates_new(self, circuit_manager):
        """Test getting breaker creates new one."""
        breaker = circuit_manager.get_breaker("test_service")
        assert breaker is not None
        assert breaker.name == "test_service"

    def test_get_breaker_returns_existing(self, circuit_manager):
        """Test getting breaker returns existing one."""
        breaker1 = circuit_manager.get_breaker("test_service")
        breaker2 = circuit_manager.get_breaker("test_service")
        assert breaker1 is breaker2

    def test_get_breaker_with_known_service(self, circuit_manager):
        """Test getting breaker for known service types."""
        llm_breaker = circuit_manager.get_breaker("llm")
        assert llm_breaker.name == "llm"

        db_breaker = circuit_manager.get_breaker("database")
        assert db_breaker.name == "database"

    def test_get_all_breakers(self, circuit_manager):
        """Test getting all breakers."""
        circuit_manager.get_breaker("service1")
        circuit_manager.get_breaker("service2")

        all_breakers = circuit_manager.get_all_breakers()
        assert len(all_breakers) == 2
        assert "service1" in all_breakers
        assert "service2" in all_breakers

    def test_get_breaker_stats(self, circuit_manager):
        """Test getting breaker statistics."""
        circuit_manager.get_breaker("test_service")
        stats = circuit_manager.get_breaker_stats()
        assert "test_service" in stats

    def test_reset_breaker(self, circuit_manager):
        """Test resetting specific breaker."""
        breaker = circuit_manager.get_breaker("test_service")
        breaker.state = CircuitState.OPEN

        circuit_manager.reset_breaker("test_service")
        assert breaker.state == CircuitState.CLOSED

    def test_reset_all_breakers(self, circuit_manager):
        """Test resetting all breakers."""
        breaker1 = circuit_manager.get_breaker("service1")
        breaker2 = circuit_manager.get_breaker("service2")
        breaker1.state = CircuitState.OPEN
        breaker2.state = CircuitState.OPEN

        circuit_manager.reset_all_breakers()
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_force_open_breaker(self, circuit_manager):
        """Test forcing breaker open."""
        breaker = circuit_manager.get_breaker("test_service")
        circuit_manager.force_open_breaker("test_service")
        assert breaker.state == CircuitState.OPEN

    def test_force_close_breaker(self, circuit_manager):
        """Test forcing breaker closed."""
        breaker = circuit_manager.get_breaker("test_service")
        breaker.state = CircuitState.OPEN
        circuit_manager.force_close_breaker("test_service")
        assert breaker.state == CircuitState.CLOSED

    def test_remove_breaker(self, circuit_manager):
        """Test removing breaker."""
        circuit_manager.get_breaker("test_service")
        assert "test_service" in circuit_manager._breakers

        circuit_manager.remove_breaker("test_service")
        assert "test_service" not in circuit_manager._breakers

    def test_clear_all_breakers(self, circuit_manager):
        """Test clearing all breakers."""
        circuit_manager.get_breaker("service1")
        circuit_manager.get_breaker("service2")

        circuit_manager.clear_all_breakers()
        assert len(circuit_manager._breakers) == 0

    def test_get_global_stats(self, circuit_manager):
        """Test getting global statistics."""
        circuit_manager.get_breaker("service1")
        circuit_manager.get_breaker("service2")

        stats = circuit_manager.get_global_stats()
        assert "total_breakers" in stats
        assert "open_breakers" in stats
        assert "closed_breakers" in stats


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_circuit_config_defaults(self):
        """Test circuit config with default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.fallback_enabled is True
        assert config.success_threshold == 3

    def test_circuit_settings_get_configs(self):
        """Test circuit settings get config methods."""
        settings = CircuitBreakerSettings()

        llm_config = settings.get_llm_config()
        assert isinstance(llm_config, CircuitBreakerConfig)
        assert llm_config.failure_threshold == 5

        db_config = settings.get_database_config()
        assert isinstance(db_config, CircuitBreakerConfig)
        assert db_config.failure_threshold == 3

        mcp_config = settings.get_mcp_config()
        assert isinstance(mcp_config, CircuitBreakerConfig)
        assert mcp_config.failure_threshold == 3

        secret_config = settings.get_secret_config()
        assert isinstance(secret_config, CircuitBreakerConfig)
        assert secret_config.failure_threshold == 2
