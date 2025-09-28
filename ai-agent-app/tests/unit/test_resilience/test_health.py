"""Tests for health checking functionality."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_agent.resilience.health import (
    CustomHealthChecker,
    DatabaseHealthChecker,
    HealthCheckResult,
    HealthMonitorConfig,
    HealthStatus,
    HTTPHealthChecker,
    RedisHealthChecker,
    ServiceHealthMonitor,
)


class TestHealthCheckResult:
    """Test health check result functionality."""

    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is healthy",
            timestamp=datetime.now(UTC),
            response_time_ms=100.0,
            details={"key": "value"},
        )

        assert result.service_name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.response_time_ms == 100.0
        assert result.details == {"key": "value"}

    def test_health_check_result_to_dict(self):
        """Test converting health check result to dictionary."""
        timestamp = datetime.now(UTC)
        result = HealthCheckResult(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is healthy",
            timestamp=timestamp,
            response_time_ms=100.0,
            details={"key": "value"},
        )

        result_dict = result.to_dict()
        assert result_dict["service_name"] == "test_service"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "Service is healthy"
        assert result_dict["response_time_ms"] == 100.0
        assert result_dict["details"] == {"key": "value"}
        assert result_dict["timestamp"] == timestamp.isoformat()


class TestDatabaseHealthChecker:
    """Test database health checker functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        connection = AsyncMock()
        connection.execute = AsyncMock(return_value=None)
        connection.acquire = AsyncMock()
        connection.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        connection.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return connection

    @pytest.fixture
    def db_checker(self, mock_connection):
        """Create database health checker for testing."""

        async def connection_func():
            return mock_connection

        return DatabaseHealthChecker("test_db", connection_func, timeout=1.0)

    @pytest.mark.asyncio
    async def test_database_health_check_success(self, db_checker, mock_connection):
        """Test successful database health check."""
        mock_connection.execute.return_value = None

        result = await db_checker.check_health()

        assert result.service_name == "test_db"
        assert result.status == HealthStatus.HEALTHY
        assert "Database is healthy" in result.message
        assert result.response_time_ms > 0
        mock_connection.execute.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, db_checker, mock_connection):
        """Test database health check failure."""
        mock_connection.execute.side_effect = Exception("Connection failed")

        result = await db_checker.check_health()

        assert result.service_name == "test_db"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Database health check failed" in result.message
        assert result.error == "Connection failed"

    @pytest.mark.asyncio
    async def test_database_health_check_timeout(self, db_checker):
        """Test database health check timeout."""

        async def slow_connection():
            await asyncio.sleep(2.0)  # Longer than timeout
            return AsyncMock()

        checker = DatabaseHealthChecker("test_db", slow_connection, timeout=0.1)
        result = await checker.run_check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message


class TestHTTPHealthChecker:
    """Test HTTP health checker functionality."""

    @pytest.fixture
    def http_checker(self):
        """Create HTTP health checker for testing."""
        return HTTPHealthChecker(
            "test_api",
            "https://httpbin.org/status/200",
            timeout=5.0,
            expected_status=200,
        )

    @pytest.mark.asyncio
    async def test_http_health_check_success(self, http_checker):
        """Test successful HTTP health check."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await http_checker.check_health()

            assert result.service_name == "test_api"
            assert result.status == HealthStatus.HEALTHY
            assert "HTTP service is healthy" in result.message
            assert result.details["status_code"] == 200

    @pytest.mark.asyncio
    async def test_http_health_check_wrong_status(self, http_checker):
        """Test HTTP health check with wrong status code."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await http_checker.check_health()

            assert result.status == HealthStatus.UNHEALTHY
            assert "unexpected status" in result.message

    @pytest.mark.asyncio
    async def test_http_health_check_connection_error(self, http_checker):
        """Test HTTP health check with connection error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("Connection failed")
            )

            result = await http_checker.check_health()

            assert result.status == HealthStatus.UNHEALTHY
            assert "HTTP health check failed" in result.message


class TestRedisHealthChecker:
    """Test Redis health checker functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_client = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)
        redis_client.info = AsyncMock(
            return_value={
                "redis_version": "6.0.0",
                "uptime_in_seconds": 3600,
                "connected_clients": 5,
                "used_memory_human": "1M",
            }
        )
        return redis_client

    @pytest.fixture
    def redis_checker(self, mock_redis):
        """Create Redis health checker for testing."""
        return RedisHealthChecker("test_redis", mock_redis, timeout=1.0)

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, redis_checker, mock_redis):
        """Test successful Redis health check."""
        result = await redis_checker.check_health()

        assert result.service_name == "test_redis"
        assert result.status == HealthStatus.HEALTHY
        assert "Redis is healthy" in result.message
        assert result.details["ping_result"] is True
        assert "redis_version" in result.details

    @pytest.mark.asyncio
    async def test_redis_health_check_failure(self, mock_redis):
        """Test Redis health check failure."""
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        checker = RedisHealthChecker("test_redis", mock_redis, timeout=1.0)
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Redis health check failed" in result.message


class TestCustomHealthChecker:
    """Test custom health checker functionality."""

    @pytest.fixture
    def custom_checker(self):
        """Create custom health checker for testing."""

        async def check_func():
            return {"healthy": True, "message": "Custom check passed"}

        return CustomHealthChecker("custom_service", check_func, timeout=1.0)

    @pytest.mark.asyncio
    async def test_custom_health_check_success(self, custom_checker):
        """Test successful custom health check."""
        result = await custom_checker.check_health()

        assert result.service_name == "custom_service"
        assert result.status == HealthStatus.HEALTHY
        assert "Custom check passed" in result.message

    @pytest.mark.asyncio
    async def test_custom_health_check_boolean_result(self):
        """Test custom health check with boolean result."""

        async def check_func():
            return True

        checker = CustomHealthChecker("custom_service", check_func, timeout=1.0)
        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_custom_health_check_failure(self):
        """Test custom health check failure."""

        async def check_func():
            raise Exception("Custom check failed")

        checker = CustomHealthChecker("custom_service", check_func, timeout=1.0)
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Custom health check failed" in result.message


class TestServiceHealthMonitor:
    """Test service health monitor functionality."""

    @pytest.fixture
    def mock_checker(self):
        """Create mock health checker."""
        checker = AsyncMock()
        checker.service_name = "test_service"
        checker.run_check = AsyncMock(
            return_value=HealthCheckResult(
                service_name="test_service",
                status=HealthStatus.HEALTHY,
                message="Service is healthy",
                timestamp=datetime.now(UTC),
                response_time_ms=100.0,
                details={},
            )
        )
        return checker

    @pytest.fixture
    def health_monitor(self):
        """Create health monitor for testing."""
        config = HealthMonitorConfig(
            check_interval=0.1, timeout=1.0  # Very fast for testing
        )
        return ServiceHealthMonitor(config)

    def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor is not None
        assert health_monitor.config is not None
        assert len(health_monitor.checkers) == 0

    def test_add_checker(self, health_monitor, mock_checker):
        """Test adding health checker."""
        health_monitor.add_checker("test_service", mock_checker)
        assert "test_service" in health_monitor.checkers
        assert "test_service" in health_monitor.metrics

    def test_remove_checker(self, health_monitor, mock_checker):
        """Test removing health checker."""
        health_monitor.add_checker("test_service", mock_checker)
        health_monitor.remove_checker("test_service")
        assert "test_service" not in health_monitor.checkers
        assert "test_service" not in health_monitor.metrics

    @pytest.mark.asyncio
    async def test_check_service_now(self, health_monitor, mock_checker):
        """Test checking service immediately."""
        health_monitor.add_checker("test_service", mock_checker)

        result = await health_monitor.check_service_now("test_service")
        assert result is not None
        assert result.service_name == "test_service"
        mock_checker.run_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_service_now_not_found(self, health_monitor):
        """Test checking non-existent service."""
        result = await health_monitor.check_service_now("nonexistent")
        assert result is None

    def test_get_service_health(self, health_monitor, mock_checker):
        """Test getting service health information."""
        health_monitor.add_checker("test_service", mock_checker)

        # Add some metrics
        metrics = health_monitor.metrics["test_service"]
        metrics.add_check_result(
            HealthCheckResult(
                service_name="test_service",
                status=HealthStatus.HEALTHY,
                message="Service is healthy",
                timestamp=datetime.now(UTC),
                response_time_ms=100.0,
                details={},
            )
        )

        health_info = health_monitor.get_service_health("test_service")
        assert health_info is not None
        assert health_info["service_name"] == "test_service"
        assert health_info["is_healthy"] is True

    def test_get_all_health(self, health_monitor, mock_checker):
        """Test getting all health information."""
        health_monitor.add_checker("test_service", mock_checker)

        all_health = health_monitor.get_all_health()
        assert "test_service" in all_health

    def test_get_global_health(self, health_monitor):
        """Test getting global health status."""
        global_health = health_monitor.get_global_health()
        assert "overall_status" in global_health
        assert "total_services" in global_health
        assert global_health["total_services"] == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_monitor, mock_checker):
        """Test starting and stopping monitoring."""
        health_monitor.add_checker("test_service", mock_checker)

        # Start monitoring
        await health_monitor.start_monitoring()
        assert health_monitor._monitoring_task is not None

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        await health_monitor.stop_monitoring()
        assert health_monitor._monitoring_task is None


class TestServiceHealthMetrics:
    """Test service health metrics functionality."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")
        assert metrics.service_name == "test_service"
        assert metrics.total_checks == 0
        assert metrics.successful_checks == 0
        assert metrics.failed_checks == 0
        assert metrics.consecutive_failures == 0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")
        assert metrics.success_rate == 0.0

        metrics.total_checks = 10
        metrics.successful_checks = 8
        assert metrics.success_rate == 0.8

    def test_is_healthy_property(self):
        """Test is_healthy property."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")
        assert metrics.is_healthy is False  # No checks yet

        metrics.total_checks = 10
        metrics.successful_checks = 8
        metrics.consecutive_failures = 2
        assert metrics.is_healthy is True

        metrics.consecutive_failures = 3
        assert metrics.is_healthy is False

    def test_is_degraded_property(self):
        """Test is_degraded property."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")
        metrics.total_checks = 10
        metrics.successful_checks = 6  # 60% success rate
        assert metrics.is_degraded is True

        metrics.successful_checks = 8  # 80% success rate
        assert metrics.is_degraded is False

    def test_add_check_result(self):
        """Test adding check result."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")

        # Add successful check
        result = HealthCheckResult(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            message="Success",
            timestamp=datetime.now(UTC),
            response_time_ms=100.0,
            details={},
        )

        metrics.add_check_result(result)
        assert metrics.total_checks == 1
        assert metrics.successful_checks == 1
        assert metrics.failed_checks == 0
        assert metrics.consecutive_failures == 0

    def test_get_status(self):
        """Test getting health status."""
        from ai_agent.resilience.health.monitor import ServiceHealthMetrics

        metrics = ServiceHealthMetrics("test_service")
        assert metrics.get_status() == HealthStatus.UNKNOWN

        metrics.total_checks = 10
        metrics.successful_checks = 8
        assert metrics.get_status() == HealthStatus.HEALTHY

        metrics.consecutive_failures = 3
        assert metrics.get_status() == HealthStatus.UNHEALTHY
