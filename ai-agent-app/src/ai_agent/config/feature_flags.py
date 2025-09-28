"""Feature flag management system."""

from typing import Any


class FeatureFlags:
    """Feature flags configuration."""

    def __init__(self) -> None:
        # Resilience features
        self.enable_circuit_breakers = True
        self.enable_retry_logic = True
        self.enable_timeout_handling = True

        # API features
        self.enable_websockets = True
        self.enable_debug_endpoints = False
        self.enable_metrics_endpoint = True

        # Observability features
        self.enable_structured_logging = True
        self.enable_distributed_tracing = True
        self.enable_metrics_collection = True

        # External service features
        self.enable_llm_providers = True
        self.enable_mcp_servers = True
        self.enable_secret_management = True


class FeatureFlagManager:
    """Feature flag management with runtime control."""

    def __init__(self, flags: FeatureFlags) -> None:
        self.flags = flags
        self._runtime_overrides: dict[str, Any] = {}

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        # Check runtime overrides first
        if flag_name in self._runtime_overrides:
            return bool(self._runtime_overrides[flag_name])

        # Fall back to configuration
        return bool(getattr(self.flags, flag_name, False))

    def enable(self, flag_name: str) -> None:
        """Enable a feature flag at runtime."""
        self._runtime_overrides[flag_name] = True

    def disable(self, flag_name: str) -> None:
        """Disable a feature flag at runtime."""
        self._runtime_overrides[flag_name] = False

    def reset_override(self, flag_name: str) -> None:
        """Reset a feature flag to its configuration value."""
        self._runtime_overrides.pop(flag_name, None)

    def get_all_flags(self) -> dict[str, Any]:
        """Get all feature flags with current values."""
        flags_dict = (
            self.flags.model_dump() if hasattr(self.flags, "model_dump") else {}
        )
        flags_dict.update(self._runtime_overrides)
        return flags_dict
