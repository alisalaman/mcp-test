"""Base secret provider interface and common functionality."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SecretProviderType(str, Enum):
    """Supported secret provider types."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"


class SecretError(Exception):
    """Base exception for secret management errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        secret_name: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.secret_name = secret_name
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
        self.correlation_id = str(uuid4())


class SecretNotFoundError(SecretError):
    """Secret not found error."""

    def __init__(self, secret_name: str, provider: str):
        super().__init__(
            f"Secret '{secret_name}' not found",
            provider=provider,
            secret_name=secret_name,
            error_code="SECRET_NOT_FOUND",
        )


class SecretAccessError(SecretError):
    """Secret access denied error."""

    def __init__(self, secret_name: str, provider: str, reason: str):
        super().__init__(
            f"Access denied to secret '{secret_name}': {reason}",
            provider=provider,
            secret_name=secret_name,
            error_code="ACCESS_DENIED",
            details={"reason": reason},
        )


class SecretValidationError(SecretError):
    """Secret validation error."""

    def __init__(self, secret_name: str, provider: str, validation_errors: list[str]):
        super().__init__(
            f"Secret '{secret_name}' validation failed",
            provider=provider,
            secret_name=secret_name,
            error_code="VALIDATION_ERROR",
            details={"validation_errors": validation_errors},
        )


class SecretMetadata(BaseModel):
    """Secret metadata information."""

    name: str
    provider: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    rotation_enabled: bool = False
    next_rotation: datetime | None = None
    size_bytes: int | None = None


class SecretValue(BaseModel):
    """Secret value with metadata."""

    value: str
    metadata: SecretMetadata
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))


class SecretProvider(ABC):
    """Abstract base class for secret providers."""

    def __init__(self, provider_type: SecretProviderType, config: dict[str, Any]):
        self.provider_type = provider_type
        self.config = config
        self.logger = logging.getLogger(f"secret_provider.{provider_type.value}")
        self._cache: dict[str, SecretValue] = {}
        self._cache_ttl: int = config.get("cache_ttl", 300)  # 5 minutes default
        self._lock = asyncio.Lock()

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to secret provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to secret provider."""
        pass

    @abstractmethod
    async def get_secret(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Retrieve a secret value."""
        pass

    @abstractmethod
    async def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Store a secret value."""
        pass

    @abstractmethod
    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret or specific version."""
        pass

    @abstractmethod
    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List available secrets."""
        pass

    @abstractmethod
    async def rotate_secret(self, secret_name: str) -> SecretMetadata:
        """Rotate a secret value."""
        pass

    @abstractmethod
    async def get_secret_metadata(self, secret_name: str) -> SecretMetadata:
        """Get secret metadata without retrieving the value."""
        pass

    async def get_secret_cached(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Get secret with caching support."""
        cache_key = f"{secret_name}:{version or 'latest'}"

        async with self._lock:
            # Check cache first
            if cache_key in self._cache:
                cached_secret = self._cache[cache_key]
                age = (datetime.now(UTC) - cached_secret.retrieved_at).total_seconds()

                if age < self._cache_ttl:
                    self.logger.debug(f"Retrieved secret from cache: {secret_name}")
                    return cached_secret
                else:
                    # Cache expired, remove it
                    del self._cache[cache_key]

            # Retrieve from provider
            secret_value = await self.get_secret(secret_name, version)

            # Cache the result
            self._cache[cache_key] = secret_value

            return secret_value

    async def invalidate_cache(self, secret_name: str | None = None) -> None:
        """Invalidate cache for a specific secret or all secrets."""
        async with self._lock:
            if secret_name:
                # Remove specific secret from cache
                keys_to_remove = [
                    key
                    for key in self._cache.keys()
                    if key.startswith(f"{secret_name}:")
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                self.logger.debug(f"Invalidated cache for secret: {secret_name}")
            else:
                # Clear all cache
                self._cache.clear()
                self.logger.debug("Invalidated all secret cache")

    async def health_check(self) -> dict[str, Any]:
        """Check provider health and return status information."""
        try:
            # Try to list secrets as a health check
            await self.list_secrets()
            return {
                "status": "healthy",
                "provider": self.provider_type.value,
                "timestamp": datetime.now(UTC).isoformat(),
                "cache_size": len(self._cache),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_type.value,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _validate_secret_name(self, secret_name: str) -> None:
        """Validate secret name format."""
        if not secret_name:
            raise SecretValidationError(
                secret_name=secret_name,
                provider=self.provider_type.value,
                validation_errors=["Secret name cannot be empty"],
            )

        # Basic validation - can be extended per provider
        if len(secret_name) > 200:
            raise SecretValidationError(
                secret_name=secret_name,
                provider=self.provider_type.value,
                validation_errors=["Secret name too long (max 200 characters)"],
            )

    def _validate_secret_value(self, secret_value: str) -> None:
        """Validate secret value."""
        if not secret_value:
            raise SecretValidationError(
                secret_name="unknown",
                provider=self.provider_type.value,
                validation_errors=["Secret value cannot be empty"],
            )

        # Basic validation - can be extended per provider
        if len(secret_value) > 65536:  # 64KB limit
            raise SecretValidationError(
                secret_name="unknown",
                provider=self.provider_type.value,
                validation_errors=["Secret value too large (max 64KB)"],
            )

    def _create_secret_metadata(
        self,
        secret_name: str,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: str | None = None,
        tags: dict[str, str] | None = None,
        rotation_enabled: bool = False,
        next_rotation: datetime | None = None,
        size_bytes: int | None = None,
    ) -> SecretMetadata:
        """Create secret metadata object."""
        return SecretMetadata(
            name=secret_name,
            provider=self.provider_type.value,
            created_at=created_at or datetime.now(UTC),
            updated_at=updated_at or datetime.now(UTC),
            version=version,
            tags=tags or {},
            rotation_enabled=rotation_enabled,
            next_rotation=next_rotation,
            size_bytes=size_bytes,
        )

    def _log_secret_access(
        self,
        operation: str,
        secret_name: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log secret access for audit purposes."""
        log_data = {
            "operation": operation,
            "secret_name": secret_name,
            "provider": self.provider_type.value,
            "success": success,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if error:
            log_data["error"] = error

        if success:
            self.logger.info("Secret access", extra=log_data)
        else:
            self.logger.warning("Secret access failed", extra=log_data)
