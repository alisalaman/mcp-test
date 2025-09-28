"""Tests for retry mechanisms."""

import pytest
import tenacity
from ai_agent.resilience.retry import RetryConfig, RetryManager, RetrySettings


class TestRetryManager:
    """Test retry manager functionality."""

    @pytest.fixture
    def retry_settings(self):
        """Create retry settings for testing."""
        return RetrySettings(
            llm_max_attempts=3,
            llm_base_delay=0.1,
            llm_max_delay=1.0,
            llm_multiplier=2.0,
        )

    @pytest.fixture
    def retry_manager(self, retry_settings):
        """Create retry manager for testing."""
        return RetryManager(retry_settings)

    def test_retry_manager_initialization(self, retry_manager):
        """Test retry manager initialization."""
        assert retry_manager is not None
        assert retry_manager.settings is not None

    def test_get_llm_retry_decorator(self, retry_manager):
        """Test getting LLM retry decorator."""
        decorator = retry_manager.get_llm_retry_decorator()
        assert callable(decorator)

    def test_get_database_retry_decorator(self, retry_manager):
        """Test getting database retry decorator."""
        decorator = retry_manager.get_database_retry_decorator()
        assert callable(decorator)

    def test_get_mcp_retry_decorator(self, retry_manager):
        """Test getting MCP retry decorator."""
        decorator = retry_manager.get_mcp_retry_decorator()
        assert callable(decorator)

    def test_get_secret_retry_decorator(self, retry_manager):
        """Test getting secret retry decorator."""
        decorator = retry_manager.get_secret_retry_decorator()
        assert callable(decorator)

    def test_get_retry_decorator_unknown_service(self, retry_manager):
        """Test getting retry decorator for unknown service."""
        with pytest.raises(ValueError, match="Unknown service type"):
            retry_manager.get_retry_decorator("unknown_service")

    def test_get_config(self, retry_manager):
        """Test getting retry configuration."""
        llm_config = retry_manager.get_config("llm")
        assert isinstance(llm_config, RetryConfig)
        assert llm_config.max_attempts == 3

    def test_create_custom_decorator(self, retry_manager):
        """Test creating custom retry decorator."""
        config = RetryConfig(max_attempts=5, base_delay=0.1)
        decorator = retry_manager.create_custom_decorator(config)
        assert callable(decorator)

    def test_clear_cache(self, retry_manager):
        """Test clearing decorator cache."""
        # Get a decorator to populate cache
        retry_manager.get_llm_retry_decorator()
        assert len(retry_manager._decorators) > 0

        retry_manager.clear_cache()
        assert len(retry_manager._decorators) == 0

    def test_get_stats(self, retry_manager):
        """Test getting retry manager statistics."""
        stats = retry_manager.get_stats()
        assert "cached_decorators" in stats
        assert "service_types" in stats


class TestRetryDecorator:
    """Test retry decorator functionality."""

    @pytest.fixture
    def retry_settings(self):
        """Create retry settings for testing."""
        return RetrySettings(
            llm_max_attempts=3,
            llm_base_delay=0.01,  # Very fast for testing
            llm_max_delay=0.1,
            llm_multiplier=2.0,
        )

    @pytest.fixture
    def retry_manager(self, retry_settings):
        """Create retry manager for testing."""
        return RetryManager(retry_settings)

    @pytest.mark.asyncio
    async def test_successful_retry(self, retry_manager):
        """Test retry decorator with successful call."""
        call_count = 0

        @retry_manager.get_llm_retry_decorator()
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_function()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_failure_then_success(self, retry_manager):
        """Test retry decorator with initial failure then success."""
        call_count = 0

        @retry_manager.get_llm_retry_decorator()
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")  # Use retryable exception
            return "success"

        result = await test_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, retry_manager):
        """Test retry decorator with all attempts failing."""
        call_count = 0

        @retry_manager.get_llm_retry_decorator()
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")  # Use retryable exception

        with pytest.raises(tenacity.RetryError):  # tenacity wraps in RetryError
            await test_function()

        assert call_count == 3  # Max attempts

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self, retry_manager):
        """Test retry decorator with non-retryable exception."""
        call_count = 0

        @retry_manager.get_llm_retry_decorator()
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError, match="Non-retryable error"):
            await test_function()

        assert call_count == 1  # No retries for non-retryable exceptions

    @pytest.mark.asyncio
    async def test_retry_with_correlation_id(self, retry_manager):
        """Test retry decorator with correlation ID."""
        call_count = 0

        @retry_manager.get_llm_retry_decorator("test-correlation-id")
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")  # Use retryable exception
            return "success"

        result = await test_function()
        assert result == "success"
        assert call_count == 2


class TestRetryConfig:
    """Test retry configuration."""

    def test_retry_config_defaults(self):
        """Test retry config with default values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.multiplier == 2.0
        assert config.jitter is True

    def test_retry_config_validation(self):
        """Test retry config validation."""
        # Test max_delay validation
        with pytest.raises(
            ValueError, match="max_delay must be greater than base_delay"
        ):
            RetryConfig(base_delay=10.0, max_delay=5.0)

    def test_retry_settings_get_configs(self):
        """Test retry settings get config methods."""
        settings = RetrySettings()

        llm_config = settings.get_llm_config()
        assert isinstance(llm_config, RetryConfig)
        assert llm_config.max_attempts == 3

        db_config = settings.get_database_config()
        assert isinstance(db_config, RetryConfig)
        assert db_config.max_attempts == 5

        mcp_config = settings.get_mcp_config()
        assert isinstance(mcp_config, RetryConfig)
        assert mcp_config.max_attempts == 3

        secret_config = settings.get_secret_config()
        assert isinstance(secret_config, RetryConfig)
        assert secret_config.max_attempts == 2
