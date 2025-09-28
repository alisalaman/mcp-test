"""Retry configuration models and settings."""

from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetryConfig(BaseModel):
    """Retry configuration for external services."""

    max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts"
    )
    base_delay: float = Field(
        default=1.0, ge=0.01, le=60.0, description="Base delay in seconds"
    )
    max_delay: float = Field(
        default=60.0, ge=0.1, le=300.0, description="Maximum delay in seconds"
    )
    multiplier: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Exponential backoff multiplier"
    )
    jitter: bool = Field(
        default=True, description="Add jitter to prevent thundering herd"
    )
    retryable_errors: list[str] = Field(
        default_factory=list, description="List of retryable exception types"
    )

    @field_validator("max_delay")
    @classmethod
    def validate_max_delay(cls, v: float, info: Any) -> float:
        """Ensure max_delay is greater than base_delay."""
        if info.data.get("base_delay") and v <= info.data["base_delay"]:
            raise ValueError("max_delay must be greater than base_delay")
        return v


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

    def get_llm_config(self) -> RetryConfig:
        """Get LLM-specific retry configuration."""
        return RetryConfig(
            max_attempts=self.llm_max_attempts,
            base_delay=self.llm_base_delay,
            max_delay=self.llm_max_delay,
            multiplier=self.llm_multiplier,
            retryable_errors=[
                "httpx.TimeoutException",
                "openai.RateLimitError",
                "openai.APITimeoutError",
                "openai.InternalServerError",
            ],
        )

    def get_database_config(self) -> RetryConfig:
        """Get database-specific retry configuration."""
        return RetryConfig(
            max_attempts=self.db_max_attempts,
            base_delay=self.db_base_delay,
            max_delay=self.db_max_delay,
            multiplier=self.db_multiplier,
            retryable_errors=[
                "asyncpg.exceptions.ConnectionDoesNotExistError",
                "asyncpg.exceptions.ConnectionFailureError",
                "asyncpg.exceptions.TooManyConnectionsError",
                "ConnectionError",
                "TimeoutError",
            ],
        )

    def get_mcp_config(self) -> RetryConfig:
        """Get MCP-specific retry configuration."""
        return RetryConfig(
            max_attempts=self.mcp_max_attempts,
            base_delay=self.mcp_base_delay,
            max_delay=self.mcp_max_delay,
            multiplier=self.mcp_multiplier,
            retryable_errors=[
                "httpx.ConnectError",
                "httpx.TimeoutException",
                "httpx.HTTPStatusError",
                "ConnectionError",
                "TimeoutError",
            ],
        )

    def get_secret_config(self) -> RetryConfig:
        """Get secret manager-specific retry configuration."""
        return RetryConfig(
            max_attempts=self.secret_max_attempts,
            base_delay=self.secret_base_delay,
            max_delay=self.secret_max_delay,
            multiplier=self.secret_multiplier,
            retryable_errors=[
                "boto3.exceptions.Boto3Error",
                "azure.core.exceptions.ServiceRequestError",
                "google.api_core.exceptions.RetryError",
                "ConnectionError",
                "TimeoutError",
            ],
        )
