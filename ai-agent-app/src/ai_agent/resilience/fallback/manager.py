"""Fallback manager for centralized fallback handling."""

from collections.abc import Callable
from typing import Any

import structlog

from .handlers import (
    DatabaseFallbackHandler,
    FallbackHandler,
    LLMFallbackHandler,
    MCPFallbackHandler,
    SecretFallbackHandler,
    ServiceFallbackHandler,
)
from .strategies import FallbackConfig, FallbackStrategy

logger = structlog.get_logger()


class FallbackManager:
    """Manages fallback strategies for all services."""

    def __init__(self, config: FallbackConfig):
        """Initialize fallback manager.

        Args:
            config: Fallback configuration
        """
        self.config = config
        self.handlers: dict[str, FallbackHandler] = {}
        self._initialize_default_handlers()

    def _initialize_default_handlers(self) -> None:
        """Initialize default fallback handlers for common services."""
        # LLM handler
        self.handlers["llm"] = LLMFallbackHandler(self.config)

        # Database handler
        self.handlers["database"] = DatabaseFallbackHandler(self.config)

        # MCP handler
        self.handlers["mcp"] = MCPFallbackHandler(self.config)

        # Secret handler
        self.handlers["secret"] = SecretFallbackHandler(self.config)

    def add_handler(self, service_name: str, handler: FallbackHandler) -> None:
        """Add a fallback handler for a service.

        Args:
            service_name: Name of the service
            handler: Fallback handler instance
        """
        self.handlers[service_name] = handler
        logger.info("Added fallback handler", service_name=service_name)

    def get_handler(self, service_name: str) -> FallbackHandler | None:
        """Get fallback handler for a service.

        Args:
            service_name: Name of the service

        Returns:
            Fallback handler or None if not found
        """
        return self.handlers.get(service_name)

    def create_custom_handler(
        self,
        service_name: str,
        strategies: list[FallbackStrategy],
        default_value: Any = None,
    ) -> ServiceFallbackHandler:
        """Create a custom fallback handler.

        Args:
            service_name: Name of the service
            strategies: List of fallback strategies
            default_value: Default value for the service

        Returns:
            Custom fallback handler
        """
        handler = ServiceFallbackHandler(service_name, self.config, default_value)

        # Add custom strategies
        for strategy in strategies:
            handler.add_strategy(strategy)

        self.handlers[service_name] = handler
        logger.info("Created custom fallback handler", service_name=service_name)

        return handler

    async def execute_fallback(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Execute fallback for a service.

        Args:
            service_name: Name of the service
            original_func: Original function that failed
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Fallback result
        """
        handler = self.get_handler(service_name)

        if not handler:
            logger.warning(
                "No fallback handler found for service", service_name=service_name
            )
            return None

        return await handler.execute_fallback(
            service_name, original_func, *args, **kwargs
        )

    async def handle_service_call(  # type: ignore[no-untyped-def]
        self, service_name: str, original_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Handle a service call with fallback.

        Args:
            service_name: Name of the service
            original_func: Original function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result
        """
        handler = self.get_handler(service_name)

        if not handler:
            logger.warning(
                "No fallback handler found, calling original function",
                service_name=service_name,
            )
            return await original_func(*args, **kwargs)

        if isinstance(handler, ServiceFallbackHandler):
            return await handler.handle_service_call(original_func, *args, **kwargs)
        else:
            # Generic handler
            try:
                return await original_func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Service call failed, attempting fallback",
                    service_name=service_name,
                    error=str(e),
                )
                return await handler.execute_fallback(
                    service_name, original_func, *args, **kwargs
                )

    def get_service_fallback_info(self, service_name: str) -> dict[str, Any]:
        """Get fallback information for a service.

        Args:
            service_name: Name of the service

        Returns:
            Fallback information dictionary
        """
        handler = self.get_handler(service_name)

        if not handler:
            return {
                "service_name": service_name,
                "has_handler": False,
                "strategies": [],
                "config": self.config.__dict__,
            }

        return {
            "service_name": service_name,
            "has_handler": True,
            "handler_type": type(handler).__name__,
            "strategies": [type(s).__name__ for s in handler.strategies],
            "config": self.config.__dict__,
        }

    def get_all_fallback_info(self) -> dict[str, dict[str, Any]]:
        """Get fallback information for all services.

        Returns:
            Dictionary of service names to fallback information
        """
        return {
            service_name: self.get_service_fallback_info(service_name)
            for service_name in self.handlers.keys()
        }

    def remove_handler(self, service_name: str) -> None:
        """Remove fallback handler for a service.

        Args:
            service_name: Name of the service
        """
        if service_name in self.handlers:
            del self.handlers[service_name]
            logger.info("Removed fallback handler", service_name=service_name)
        else:
            logger.warning("Fallback handler not found", service_name=service_name)

    def clear_all_handlers(self) -> None:
        """Clear all fallback handlers."""
        self.handlers.clear()
        logger.info("Cleared all fallback handlers")

    def get_stats(self) -> dict[str, Any]:
        """Get fallback manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_handlers": len(self.handlers),
            "services": list(self.handlers.keys()),
            "config": self.config.__dict__,
        }
