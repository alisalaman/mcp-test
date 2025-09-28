"""Secret manager factory and configuration."""

import asyncio
import logging
from typing import Any

from .base import SecretProvider, SecretProviderType
from .aws_secrets import AWSSecretProvider
from .azure_keyvault import AzureKeyVaultProvider
from .gcp_secrets import GCPSecretProvider
from .local_secrets import LocalSecretProvider


class SecretManagerFactory:
    """Factory for creating secret provider instances."""

    _providers = {
        SecretProviderType.AWS: AWSSecretProvider,
        SecretProviderType.AZURE: AzureKeyVaultProvider,
        SecretProviderType.GCP: GCPSecretProvider,
        SecretProviderType.LOCAL: LocalSecretProvider,
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger("secret_manager_factory")
        self._instances: dict[str, SecretProvider] = {}
        self._lock = asyncio.Lock()

    def create_provider(
        self, provider_type: SecretProviderType, config: dict[str, Any]
    ) -> SecretProvider:
        """Create a secret provider instance."""
        if provider_type not in self._providers:
            raise ValueError(f"Unsupported provider type: {provider_type}")

        provider_class = self._providers[provider_type]
        return provider_class(config)  # type: ignore[abstract]

    async def get_provider(
        self, provider_type: SecretProviderType, config: dict[str, Any]
    ) -> SecretProvider:
        """Get or create a cached provider instance."""
        cache_key = f"{provider_type.value}:{hash(str(sorted(config.items())))}"

        async with self._lock:
            if cache_key not in self._instances:
                provider = self.create_provider(provider_type, config)
                await provider.connect()
                self._instances[cache_key] = provider
                self.logger.info(
                    f"Created new provider instance: {provider_type.value}"
                )

            return self._instances[cache_key]

    async def close_all(self) -> None:
        """Close all provider instances."""
        async with self._lock:
            for provider in self._instances.values():
                try:
                    await provider.disconnect()
                except Exception as e:
                    self.logger.warning(f"Error closing provider: {e}")

            self._instances.clear()
            self.logger.info("Closed all provider instances")

    def get_supported_providers(self) -> list[SecretProviderType]:
        """Get list of supported provider types."""
        return list(self._providers.keys())


class SecretManager:
    """Main secret manager that handles multiple providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.factory = SecretManagerFactory()
        self.logger = logging.getLogger("secret_manager")
        self._primary_provider: SecretProvider | None = None
        self._fallback_providers: list[SecretProvider] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the secret manager with configured providers."""
        if self._initialized:
            return

        try:
            # Get primary provider configuration
            primary_config = self.config.get("primary", {})
            primary_type = SecretProviderType(primary_config.get("type", "local"))

            # Create primary provider
            self._primary_provider = await self.factory.get_provider(
                primary_type, primary_config.get("config", {})
            )

            # Create fallback providers
            fallback_configs = self.config.get("fallbacks", [])
            for fallback_config in fallback_configs:
                fallback_type = SecretProviderType(fallback_config["type"])
                fallback_provider = await self.factory.get_provider(
                    fallback_type, fallback_config.get("config", {})
                )
                self._fallback_providers.append(fallback_provider)

            self._initialized = True
            self.logger.info(
                f"Initialized secret manager with primary provider: {primary_type.value}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize secret manager: {e}")
            raise

    async def get_secret(
        self, secret_name: str, version: str | None = None, use_fallback: bool = True
    ) -> Any:
        """Get a secret from the primary provider with optional fallback."""
        if not self._initialized:
            await self.initialize()

        # Try primary provider first
        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        try:
            return await self._primary_provider.get_secret(secret_name, version)
        except Exception as e:
            self.logger.warning(
                f"Primary provider failed for secret {secret_name}: {e}"
            )

            if not use_fallback:
                raise

            # Try fallback providers
            for fallback_provider in self._fallback_providers:
                try:
                    return await fallback_provider.get_secret(secret_name, version)
                except Exception as fallback_error:
                    self.logger.warning(
                        f"Fallback provider failed for secret {secret_name}: {fallback_error}"
                    )
                    continue

            # All providers failed
            raise e

    async def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Set a secret using the primary provider."""
        if not self._initialized:
            await self.initialize()

        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        return await self._primary_provider.set_secret(
            secret_name, secret_value, description, tags
        )

    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret using the primary provider."""
        if not self._initialized:
            await self.initialize()

        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        return await self._primary_provider.delete_secret(secret_name, version)

    async def list_secrets(self, prefix: str | None = None) -> list[Any]:
        """List secrets from the primary provider."""
        if not self._initialized:
            await self.initialize()

        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        return await self._primary_provider.list_secrets(prefix)

    async def rotate_secret(self, secret_name: str) -> Any:
        """Rotate a secret using the primary provider."""
        if not self._initialized:
            await self.initialize()

        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        return await self._primary_provider.rotate_secret(secret_name)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all providers."""
        if not self._initialized:
            await self.initialize()

        if not self._primary_provider:
            raise RuntimeError("No primary provider available")

        health_status: dict[str, Any] = {
            "primary": await self._primary_provider.health_check(),
            "fallbacks": [],
        }

        for fallback_provider in self._fallback_providers:
            health_status["fallbacks"].append(await fallback_provider.health_check())

        return health_status

    async def close(self) -> None:
        """Close all provider connections."""
        await self.factory.close_all()
        self._initialized = False
        self.logger.info("Secret manager closed")


# Global secret manager instance
_secret_manager: SecretManager | None = None


async def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance."""
    global _secret_manager

    if _secret_manager is None:
        # Load configuration from environment or config file
        config = {
            "primary": {
                "type": "local",  # Default to local for development
                "config": {"secrets_dir": "secrets", "auto_load_env": True},
            },
            "fallbacks": [],
        }

        _secret_manager = SecretManager(config)
        await _secret_manager.initialize()

    if _secret_manager is None:
        raise RuntimeError("Failed to initialize secret manager")

    return _secret_manager


async def close_secret_manager() -> None:
    """Close the global secret manager instance."""
    global _secret_manager

    if _secret_manager:
        await _secret_manager.close()
        _secret_manager = None


# Convenience functions for common operations
async def get_secret(secret_name: str, version: str | None = None) -> str:
    """Get a secret value by name."""
    manager = await get_secret_manager()
    secret_value = await manager.get_secret(secret_name, version)
    if hasattr(secret_value, "value"):
        return str(secret_value.value)
    return str(secret_value)


async def set_secret(
    secret_name: str,
    secret_value: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Set a secret value by name."""
    manager = await get_secret_manager()
    await manager.set_secret(secret_name, secret_value, description, tags)


async def delete_secret(secret_name: str, version: str | None = None) -> bool:
    """Delete a secret by name."""
    manager = await get_secret_manager()
    return await manager.delete_secret(secret_name, version)


async def list_secrets(prefix: str | None = None) -> list[Any]:
    """List all secrets with optional prefix filter."""
    manager = await get_secret_manager()
    return await manager.list_secrets(prefix)


async def rotate_secret(secret_name: str) -> None:
    """Rotate a secret by name."""
    manager = await get_secret_manager()
    await manager.rotate_secret(secret_name)


async def health_check() -> dict[str, Any]:
    """Check secret manager health."""
    manager = await get_secret_manager()
    return await manager.health_check()
