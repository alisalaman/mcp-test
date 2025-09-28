"""Fallback strategies for graceful degradation.

This module provides fallback mechanisms to ensure system resilience
when external services are unavailable or degraded.
"""

from .handlers import FallbackHandler, ServiceFallbackHandler
from .manager import FallbackManager
from .strategies import (
    CachedFallbackStrategy,
    DefaultValueFallbackStrategy,
    FallbackStrategy,
)

__all__ = [
    "FallbackManager",
    "FallbackStrategy",
    "CachedFallbackStrategy",
    "DefaultValueFallbackStrategy",
    "FallbackHandler",
    "ServiceFallbackHandler",
]
