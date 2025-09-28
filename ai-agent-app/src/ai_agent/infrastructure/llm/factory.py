"""LLM provider factory and management system."""

import asyncio
from typing import Any
from uuid import uuid4
import structlog

from .base import BaseLLMProvider, LLMProviderType, ModelInfo
from .openai_client import OpenAIProvider
from .anthropic_client import AnthropicProvider
from .google_client import GoogleProvider

logger = structlog.get_logger()


class LLMProviderConfig:
    """Configuration for an LLM provider."""

    def __init__(
        self,
        provider_type: LLMProviderType,
        name: str,
        config: dict[str, Any],
        enabled: bool = True,
        priority: int = 1,
        fallback_providers: list[str] | None = None,
    ):
        self.id = str(uuid4())
        self.provider_type = provider_type
        self.name = name
        self.config = config
        self.enabled = enabled
        self.priority = priority  # Lower number = higher priority
        self.fallback_providers = fallback_providers or []
        self.created_at = asyncio.get_event_loop().time()


class LLMProviderManager:
    """Manages multiple LLM providers with load balancing and failover."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._configs: dict[str, LLMProviderConfig] = {}
        self._health_status: dict[str, bool] = {}
        self._last_health_check: dict[str, float] = {}
        self._health_check_interval = 60  # 1 minute
        self._lock = asyncio.Lock()

    async def register_provider(
        self,
        provider_type: LLMProviderType,
        name: str,
        config: dict[str, Any],
        enabled: bool = True,
        priority: int = 1,
        fallback_providers: list[str] | None = None,
    ) -> str:
        """Register a new LLM provider."""
        async with self._lock:
            provider_config = LLMProviderConfig(
                provider_type=provider_type,
                name=name,
                config=config,
                enabled=enabled,
                priority=priority,
                fallback_providers=fallback_providers,
            )

            # Create the provider instance
            provider = self._create_provider(provider_config)

            # Store the provider and config
            self._providers[provider_config.id] = provider
            self._configs[provider_config.id] = provider_config
            self._health_status[provider_config.id] = True

            logger.info(
                "LLM provider registered",
                provider_id=provider_config.id,
                provider_type=provider_type.value,
                name=name,
            )

            return provider_config.id

    async def unregister_provider(self, provider_id: str) -> bool:
        """Unregister an LLM provider."""
        async with self._lock:
            if provider_id in self._providers:
                del self._providers[provider_id]
                del self._configs[provider_id]
                del self._health_status[provider_id]
                if provider_id in self._last_health_check:
                    del self._last_health_check[provider_id]

                logger.info("LLM provider unregistered", provider_id=provider_id)
                return True
            return False

    async def get_provider(self, provider_id: str) -> BaseLLMProvider | None:
        """Get a specific provider by ID."""
        return self._providers.get(provider_id)

    async def get_provider_by_name(self, name: str) -> BaseLLMProvider | None:
        """Get a provider by name."""
        for config_id, config in self._configs.items():
            if config.name == name and config.enabled:
                return self._providers.get(config_id)
        return None

    async def get_best_provider(
        self,
        provider_type: LLMProviderType | None = None,
        model: str | None = None,
    ) -> BaseLLMProvider | None:
        """Get the best available provider based on priority and health."""
        async with self._lock:
            # Filter providers by type and health
            available_providers = []

            for config_id, config in self._configs.items():
                if not config.enabled:
                    continue

                if provider_type and config.provider_type != provider_type:
                    continue

                # Check health status
                is_healthy = await self._check_provider_health(config_id)
                if not is_healthy:
                    continue

                available_providers.append((config.priority, config_id, config))

            if not available_providers:
                return None

            # Sort by priority (lower number = higher priority)
            available_providers.sort(key=lambda x: x[0])

            # Return the highest priority provider
            _, best_config_id, _ = available_providers[0]
            return self._providers.get(best_config_id)

    async def get_providers_by_type(
        self, provider_type: LLMProviderType
    ) -> list[BaseLLMProvider]:
        """Get all providers of a specific type."""
        providers = []
        for config_id, config in self._configs.items():
            if config.provider_type == provider_type and config.enabled:
                is_healthy = await self._check_provider_health(config_id)
                if is_healthy:
                    providers.append(self._providers[config_id])
        return providers

    async def get_all_models(self) -> list[ModelInfo]:
        """Get all available models from all providers."""
        all_models = []

        for provider_id, provider in self._providers.items():
            try:
                is_healthy = await self._check_provider_health(provider_id)
                if is_healthy:
                    models = await provider.get_models()
                    all_models.extend(models)
            except Exception as e:
                logger.warning(
                    "Failed to get models from provider",
                    provider_id=provider_id,
                    error=str(e),
                )

        return all_models

    async def get_models_by_provider_type(
        self, provider_type: LLMProviderType
    ) -> list[ModelInfo]:
        """Get models from providers of a specific type."""
        models = []

        for config_id, config in self._configs.items():
            if config.provider_type == provider_type and config.enabled:
                try:
                    is_healthy = await self._check_provider_health(config_id)
                    if is_healthy:
                        provider_models = await self._providers[config_id].get_models()
                        models.extend(provider_models)
                except Exception as e:
                    logger.warning(
                        "Failed to get models from provider",
                        provider_id=config_id,
                        error=str(e),
                    )

        return models

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all providers."""
        health_results = {}

        for provider_id in self._providers:
            health_results[provider_id] = await self._check_provider_health(provider_id)

        return health_results

    async def _check_provider_health(self, provider_id: str) -> bool:
        """Check if a provider is healthy."""
        current_time = asyncio.get_event_loop().time()

        # Check if we need to perform a new health check
        if (
            provider_id in self._last_health_check
            and current_time - self._last_health_check[provider_id]
            < self._health_check_interval
        ):
            return self._health_status.get(provider_id, False)

        # Perform health check
        try:
            provider = self._providers.get(provider_id)
            if not provider:
                self._health_status[provider_id] = False
                return False

            is_healthy = await provider.health_check()
            self._health_status[provider_id] = is_healthy
            self._last_health_check[provider_id] = current_time

            return is_healthy

        except Exception as e:
            logger.warning(
                "Provider health check failed", provider_id=provider_id, error=str(e)
            )
            self._health_status[provider_id] = False
            self._last_health_check[provider_id] = current_time
            return False

    def _create_provider(self, config: LLMProviderConfig) -> BaseLLMProvider:
        """Create a provider instance based on configuration."""
        if config.provider_type == LLMProviderType.OPENAI:
            return OpenAIProvider(config.config)
        elif config.provider_type == LLMProviderType.AZURE_OPENAI:
            return OpenAIProvider(config.config)
        elif config.provider_type == LLMProviderType.ANTHROPIC:
            return AnthropicProvider(config.config)
        elif config.provider_type == LLMProviderType.GOOGLE:
            return GoogleProvider(config.config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")

    async def get_provider_stats(self) -> dict[str, Any]:
        """Get statistics about all providers."""
        stats = {
            "total_providers": len(self._providers),
            "enabled_providers": sum(
                1 for config in self._configs.values() if config.enabled
            ),
            "healthy_providers": sum(
                1 for healthy in self._health_status.values() if healthy
            ),
            "providers_by_type": {},
            "provider_details": {},
        }

        # Use Counter for cleaner counting logic
        from collections import Counter

        provider_counts = Counter(
            str(config.provider_type.value) for config in self._configs.values()
        )
        stats["providers_by_type"] = dict(provider_counts)

        # Provider details
        for config_id, config in self._configs.items():
            config_dict: dict[str, Any] = {
                "name": str(config.name),
                "type": str(config.provider_type.value),
                "enabled": bool(config.enabled),
                "healthy": bool(self._health_status.get(config_id, False)),
                "priority": int(config.priority),
                "created_at": str(config.created_at),
            }
            stats["provider_details"][config_id] = config_dict  # type: ignore

        return stats


# Global provider manager instance
_provider_manager = LLMProviderManager()


async def get_llm_provider(
    provider_id: str | None = None,
    provider_type: LLMProviderType | None = None,
    provider_name: str | None = None,
) -> BaseLLMProvider | None:
    """Get an LLM provider instance."""
    if provider_id:
        return await _provider_manager.get_provider(provider_id)
    elif provider_name:
        return await _provider_manager.get_provider_by_name(provider_name)
    else:
        return await _provider_manager.get_best_provider(provider_type)


async def register_llm_provider(
    provider_type: LLMProviderType,
    name: str,
    config: dict[str, Any],
    enabled: bool = True,
    priority: int = 1,
    fallback_providers: list[str] | None = None,
) -> str:
    """Register a new LLM provider."""
    return await _provider_manager.register_provider(
        provider_type=provider_type,
        name=name,
        config=config,
        enabled=enabled,
        priority=priority,
        fallback_providers=fallback_providers,
    )


async def get_all_models() -> list[ModelInfo]:
    """Get all available models from all providers."""
    return await _provider_manager.get_all_models()


async def health_check_all_providers() -> dict[str, bool]:
    """Perform health check on all providers."""
    return await _provider_manager.health_check_all()


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    @staticmethod
    async def create_openai_provider(
        api_key: str,
        name: str = "OpenAI Default",
        base_url: str | None = None,
        default_model: str = "gpt-4o",
        **kwargs: Any,
    ) -> str:
        """Create an OpenAI provider."""
        config = {
            "api_key": api_key,
            "base_url": base_url,
            "default_model": default_model,
            **kwargs,
        }
        return await register_llm_provider(
            provider_type=LLMProviderType.OPENAI, name=name, config=config
        )

    @staticmethod
    async def create_azure_openai_provider(
        api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        name: str = "Azure OpenAI Default",
        azure_api_version: str = "2024-02-15-preview",
        **kwargs: Any,
    ) -> str:
        """Create an Azure OpenAI provider."""
        config = {
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "azure_api_version": azure_api_version,
            **kwargs,
        }
        return await register_llm_provider(
            provider_type=LLMProviderType.AZURE_OPENAI, name=name, config=config
        )

    @staticmethod
    async def create_anthropic_provider(
        api_key: str,
        name: str = "Anthropic Default",
        default_model: str = "claude-3-5-sonnet-20241022",
        **kwargs: Any,
    ) -> str:
        """Create an Anthropic provider."""
        config = {"api_key": api_key, "default_model": default_model, **kwargs}
        return await register_llm_provider(
            provider_type=LLMProviderType.ANTHROPIC, name=name, config=config
        )

    @staticmethod
    async def create_google_provider(
        api_key: str,
        name: str = "Google Default",
        default_model: str = "gemini-1.5-pro",
        **kwargs: Any,
    ) -> str:
        """Create a Google provider."""
        config = {"api_key": api_key, "default_model": default_model, **kwargs}
        return await register_llm_provider(
            provider_type=LLMProviderType.GOOGLE, name=name, config=config
        )
