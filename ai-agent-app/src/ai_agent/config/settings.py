"""
Configuration management system for AI Agent application.

This module implements environment-specific configuration with validation,
secret management integration, and factory pattern for configuration selection.
"""

import os
import sys
from enum import Enum

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    # Connection settings
    host: str = "localhost"
    port: int = 5432
    name: str = "ai_agent"
    user: str = "postgres"
    password: str = ""

    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: float = 30.0

    # SQLAlchemy settings
    echo: bool = False
    echo_pool: bool = False

    @property
    def url(self) -> str:
        """Database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_", env_file=".env", extra="ignore"
    )

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    # Connection pool settings
    max_connections: int = 20
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    @property
    def url(self) -> str:
        """Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class RetrySettings(BaseSettings):
    """Retry configuration for external services."""

    model_config = SettingsConfigDict(
        env_prefix="RETRY_", env_file=".env", extra="ignore"
    )

    # LLM API settings
    llm_max_attempts: int = 3
    llm_base_delay: float = 1.0
    llm_max_delay: float = 60.0
    llm_multiplier: float = 2.0

    # Database settings
    db_max_attempts: int = 5
    db_base_delay: float = 0.5
    db_max_delay: float = 30.0
    db_multiplier: float = 1.5

    # MCP server settings
    mcp_max_attempts: int = 3
    mcp_base_delay: float = 2.0
    mcp_max_delay: float = 120.0
    mcp_multiplier: float = 2.0

    # Secret manager settings
    secret_max_attempts: int = 2
    secret_base_delay: float = 1.0
    secret_max_delay: float = 10.0
    secret_multiplier: float = 2.0


class CircuitBreakerSettings(BaseSettings):
    """Circuit breaker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CIRCUIT_", env_file=".env", extra="ignore"
    )

    # LLM API circuit breaker
    llm_failure_threshold: int = 5
    llm_recovery_timeout: float = 60.0
    llm_expected_exception: list[str] = Field(
        default_factory=lambda: ["httpx.TimeoutException", "openai.RateLimitError"]
    )

    # Database circuit breaker
    db_failure_threshold: int = 3
    db_recovery_timeout: float = 30.0
    db_expected_exception: list[str] = Field(
        default_factory=lambda: ["asyncpg.exceptions.ConnectionDoesNotExistError"]
    )

    # MCP server circuit breaker
    mcp_failure_threshold: int = 3
    mcp_recovery_timeout: float = 120.0
    mcp_expected_exception: list[str] = Field(
        default_factory=lambda: ["httpx.ConnectError", "httpx.TimeoutException"]
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_", env_file=".env", extra="ignore"
    )

    # API rate limits (per minute)
    api_default_limit: int = 100
    api_authenticated_limit: int = 1000
    api_premium_limit: int = 5000

    # External service rate limits
    llm_requests_per_minute: int = 60
    llm_tokens_per_minute: int = 50000

    # WebSocket connection limits
    websocket_connections_per_ip: int = 10
    websocket_messages_per_minute: int = 1000


class SecuritySettings(BaseSettings):
    """Security configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_", env_file=".env", extra="ignore"
    )

    # JWT settings
    secret_key: str = Field(min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # API key settings
    api_key_length: int = 32
    api_key_prefix: str = "sk-"

    # CORS settings
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    cors_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE"]
    )
    cors_headers: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore"
    )

    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: str | None = None

    # Metrics settings
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Tracing settings
    tracing_enabled: bool = True
    tracing_sample_rate: float = 0.1
    tracing_service_name: str = "ai-agent-app"

    # Health check settings
    health_check_timeout: float = 5.0
    health_check_interval: int = 30


class FeatureFlags(BaseSettings):
    """Feature flag configuration."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_", env_file=".env", extra="ignore"
    )

    # Resilience features
    enable_circuit_breakers: bool = True
    enable_retries: bool = True
    enable_rate_limiting: bool = True
    enable_fallbacks: bool = True

    # API features
    enable_websockets: bool = True
    enable_streaming: bool = True
    enable_bulk_operations: bool = True

    # Observability features
    enable_detailed_metrics: bool = False
    enable_request_tracing: bool = True
    enable_debug_endpoints: bool = False

    # External service features
    enable_multiple_llm_providers: bool = True
    enable_provider_fallback: bool = True
    enable_mcp_hot_reload: bool = False


class ApplicationSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # Application info
    app_name: str = "AI Agent Application"
    app_version: str = "0.1.0"
    app_description: str = "Production-ready AI agent with LangGraph and MCP"

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Storage backend selection
    use_database: bool = False
    use_redis: bool = False
    use_memory: bool = True

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(
        default_factory=CircuitBreakerSettings
    )
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    security: SecuritySettings = Field(
        default_factory=lambda: SecuritySettings(
            secret_key=os.getenv(
                "SECURITY_SECRET_KEY", "dev-secret-key-change-in-production-32chars"
            )
        )
    )
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int, info: ValidationInfo) -> int:
        if (
            hasattr(info, "data")
            and info.data.get("environment") == Environment.PRODUCTION
            and v < 2
        ):
            raise ValueError("Production environment should have at least 2 workers")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


# Environment-specific configurations
class DevelopmentSettings(ApplicationSettings):
    """Development environment settings."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    use_memory: bool = True
    use_database: bool = False
    use_redis: bool = False

    # Relaxed security for development
    security: SecuritySettings = Field(
        default_factory=lambda: SecuritySettings(
            secret_key=os.getenv(
                "SECURITY_SECRET_KEY", "dev-secret-key-change-in-production-32chars"
            ),
            cors_origins=["http://localhost:3000", "http://localhost:3001"],
        )
    )

    # Verbose logging for development
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings(
            log_level=LogLevel.DEBUG,
            tracing_sample_rate=1.0,
        )
    )


class TestingSettings(ApplicationSettings):
    """Testing environment settings."""

    environment: Environment = Environment.TESTING
    debug: bool = True
    use_memory: bool = True
    use_database: bool = False
    use_redis: bool = False

    # Test-specific security
    security: SecuritySettings = Field(
        default_factory=lambda: SecuritySettings(
            secret_key="test-secret-key-for-testing-only-32chars", cors_origins=["*"]
        )
    )

    # Minimal logging for testing
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings(
            log_level=LogLevel.WARNING, tracing_enabled=False, metrics_enabled=False
        )
    )


class StagingSettings(ApplicationSettings):
    """Staging environment settings."""

    environment: Environment = Environment.STAGING
    debug: bool = False
    use_memory: bool = False
    use_database: bool = True
    use_redis: bool = True
    workers: int = 2

    # Production-like observability
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings(
            log_level=LogLevel.INFO,
            tracing_sample_rate=0.1,
        )
    )


class ProductionSettings(ApplicationSettings):
    """Production environment settings."""

    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    use_memory: bool = False
    use_database: bool = True
    use_redis: bool = True
    workers: int = 4

    # Strict security for production
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings(
            log_level=LogLevel.INFO,
            tracing_sample_rate=0.01,
        )
    )


def get_settings() -> ApplicationSettings:
    """Get application settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "development":
        return DevelopmentSettings()
    elif environment == "production":
        return ProductionSettings()
    elif environment == "staging":
        return StagingSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return ApplicationSettings()


# Configuration validation
class ConfigurationValidator:
    """Validates application configuration."""

    @staticmethod
    def validate_settings(settings: ApplicationSettings) -> list[str]:
        """Validate settings and return list of errors."""
        errors = []

        # Production-specific validations
        if settings.is_production:
            if settings.debug:
                errors.append("Debug mode should be disabled in production")

            if (
                settings.security.secret_key
                == "dev-secret-key-change-in-production-32chars"
            ):
                errors.append("Production secret key must be changed from default")

            if not settings.use_database and not settings.use_redis:
                errors.append("Production should use persistent storage")

        # Database validation
        if settings.use_database:
            try:
                # Test database connection string format
                _ = settings.database.url
            except Exception as e:
                errors.append(f"Invalid database configuration: {e}")

        # Redis validation
        if settings.use_redis:
            try:
                # Test Redis connection string format
                _ = settings.redis.url
            except Exception as e:
                errors.append(f"Invalid Redis configuration: {e}")

        return errors

    @staticmethod
    def validate_or_exit(settings: ApplicationSettings) -> None:
        """Validate settings or exit with error."""
        errors = ConfigurationValidator.validate_settings(settings)
        if errors:
            import structlog

            logger = structlog.get_logger()
            logger.error("Configuration validation failed", errors=errors)
            for error in errors:
                logger.error("Configuration error", error=error)
            sys.exit(1)


# Global settings instance
settings = get_settings()
