"""Circuit breaker configuration models and settings."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for a specific service."""

    failure_threshold: int = Field(
        default=5, ge=1, le=100, description="Number of failures before opening circuit"
    )
    recovery_timeout: float = Field(
        default=60.0,
        ge=0.1,
        le=3600.0,
        description="Timeout in seconds before attempting recovery",
    )
    expected_exception: list[str] = Field(
        default_factory=list,
        description="List of exception types that should trigger circuit breaker",
    )
    fallback_enabled: bool = Field(
        default=True, description="Whether to enable fallback when circuit is open"
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of successful calls to close circuit from half-open",
    )
    half_open_max_calls: int = Field(
        default=5, ge=1, le=20, description="Maximum calls allowed in half-open state"
    )


class CircuitBreakerSettings(BaseSettings):
    """Circuit breaker configuration for external services."""

    model_config = SettingsConfigDict(
        env_prefix="CIRCUIT_", env_file=".env", extra="ignore"
    )

    # LLM API circuit breaker
    llm_failure_threshold: int = 5
    llm_recovery_timeout: float = 60.0
    llm_expected_exception: list[str] = Field(
        default_factory=lambda: [
            "httpx.TimeoutException",
            "openai.RateLimitError",
            "openai.APITimeoutError",
            "openai.InternalServerError",
        ]
    )

    # Database circuit breaker
    db_failure_threshold: int = 3
    db_recovery_timeout: float = 30.0
    db_expected_exception: list[str] = Field(
        default_factory=lambda: [
            "asyncpg.exceptions.ConnectionDoesNotExistError",
            "asyncpg.exceptions.ConnectionFailureError",
            "asyncpg.exceptions.TooManyConnectionsError",
        ]
    )

    # MCP server circuit breaker
    mcp_failure_threshold: int = 3
    mcp_recovery_timeout: float = 120.0
    mcp_expected_exception: list[str] = Field(
        default_factory=lambda: [
            "httpx.ConnectError",
            "httpx.TimeoutException",
            "httpx.HTTPStatusError",
        ]
    )

    # Secret manager circuit breaker
    secret_failure_threshold: int = 2
    secret_recovery_timeout: float = 30.0
    secret_expected_exception: list[str] = Field(
        default_factory=lambda: [
            "boto3.exceptions.Boto3Error",
            "azure.core.exceptions.ServiceRequestError",
            "google.api_core.exceptions.RetryError",
        ]
    )

    def get_llm_config(self) -> CircuitBreakerConfig:
        """Get LLM-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=self.llm_failure_threshold,
            recovery_timeout=self.llm_recovery_timeout,
            expected_exception=self.llm_expected_exception,
            success_threshold=3,
            half_open_max_calls=5,
        )

    def get_database_config(self) -> CircuitBreakerConfig:
        """Get database-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=self.db_failure_threshold,
            recovery_timeout=self.db_recovery_timeout,
            expected_exception=self.db_expected_exception,
            success_threshold=2,
            half_open_max_calls=3,
        )

    def get_mcp_config(self) -> CircuitBreakerConfig:
        """Get MCP-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=self.mcp_failure_threshold,
            recovery_timeout=self.mcp_recovery_timeout,
            expected_exception=self.mcp_expected_exception,
            success_threshold=2,
            half_open_max_calls=3,
        )

    def get_secret_config(self) -> CircuitBreakerConfig:
        """Get secret manager-specific circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=self.secret_failure_threshold,
            recovery_timeout=self.secret_recovery_timeout,
            expected_exception=self.secret_expected_exception,
            success_threshold=1,
            half_open_max_calls=2,
        )
