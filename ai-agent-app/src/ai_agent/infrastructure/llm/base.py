"""Base LLM provider interface and common types."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections.abc import AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel, Field


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"


class LLMModelType(str, Enum):
    """LLM model types."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class LLMErrorCode(str, Enum):
    """LLM provider error codes."""

    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN_ERROR = "unknown_error"


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        error_code: LLMErrorCode,
        provider: str,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.provider = provider
        self.model = model
        self.details = details or {}


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    model: str
    provider: str
    usage: dict[str, int]
    metadata: dict[str, Any]
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: float = Field(default_factory=time.time)


@dataclass
class LLMStreamChunk:
    """LLM streaming response chunk."""

    content: str
    model: str
    provider: str
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    is_final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class FunctionCall(BaseModel):
    """Function call definition."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


class ToolCall(BaseModel):
    """Tool call definition."""

    id: str
    type: str = "function"
    function: FunctionCall


class LLMRequest(BaseModel):
    """Standardized LLM request."""

    messages: list[dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """LLM model information."""

    id: str
    name: str
    provider: str
    type: LLMModelType
    max_tokens: int | None = None
    supports_functions: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False
    cost_per_token: float | None = None
    description: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_type = self._get_provider_type()
        self._models_cache: list[ModelInfo] | None = None
        self._cache_ttl = 300  # 5 minutes

    @abstractmethod
    def _get_provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion response."""
        pass

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamChunk]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        """Get available models for this provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass

    async def generate_with_functions(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with function calling support."""
        request = LLMRequest(
            messages=messages,
            model=model or self._get_default_model(),
            tools=functions,
            **kwargs,
        )
        return await self.generate(request)

    async def stream_with_functions(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk]:
        """Stream response with function calling support."""
        request = LLMRequest(
            messages=messages,
            model=model or self._get_default_model(),
            tools=functions,
            stream=True,
            **kwargs,
        )
        stream = self.stream(request)
        async for chunk in stream:
            yield chunk

    def _get_default_model(self) -> str:
        """Get the default model for this provider."""
        return str(self.config.get("default_model", ""))

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate the LLM request."""
        if not request.messages:
            raise LLMError(
                "No messages provided",
                LLMErrorCode.INVALID_REQUEST,
                self.provider_type.value,
            )

        # Validate message structure
        for i, message in enumerate(request.messages):
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMError(
                    f"Invalid message content at index {i}",
                    LLMErrorCode.INVALID_REQUEST,
                    self.provider_type.value,
                )

        # Validate model name format
        if not request.model or not request.model.strip():
            raise LLMError(
                "No model specified",
                LLMErrorCode.INVALID_REQUEST,
                self.provider_type.value,
            )

        # Validate temperature range
        if not 0 <= request.temperature <= 2:
            raise LLMError(
                "Temperature must be between 0 and 2",
                LLMErrorCode.INVALID_REQUEST,
                self.provider_type.value,
            )

    def _create_response(
        self,
        content: str,
        model: str,
        usage: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Create a standardized response."""
        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider_type.value,
            usage=usage,
            metadata=metadata or {},
        )

    def _create_stream_chunk(
        self,
        content: str,
        model: str,
        is_final: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> LLMStreamChunk:
        """Create a streaming chunk."""
        return LLMStreamChunk(
            content=content,
            model=model,
            provider=self.provider_type.value,
            is_final=is_final,
            metadata=metadata or {},
        )

    async def _get_cached_models(self) -> list[ModelInfo] | None:
        """Get cached models if still valid."""
        if self._models_cache is None:
            return None

        # Check if cache is still valid (simple TTL check)
        current_time = time.time()
        if (
            hasattr(self, "_models_cache_time")
            and current_time - self._models_cache_time < self._cache_ttl
        ):
            return self._models_cache

        return None

    async def _cache_models(self, models: list[ModelInfo]) -> None:
        """Cache the models list."""
        self._models_cache = models
        self._models_cache_time = time.time()

    def _handle_error(self, error: Exception, context: str = "") -> LLMError:
        """Convert provider-specific errors to standardized LLMError."""
        error_message = str(error)

        # Use exception type checking instead of string matching for better reliability
        error_mapping = {
            # Authentication errors
            "AuthenticationError": LLMErrorCode.AUTHENTICATION_ERROR,
            "UnauthorizedError": LLMErrorCode.AUTHENTICATION_ERROR,
            "InvalidApiKeyError": LLMErrorCode.AUTHENTICATION_ERROR,
            # Rate limiting errors
            "RateLimitError": LLMErrorCode.RATE_LIMIT_ERROR,
            "TooManyRequestsError": LLMErrorCode.RATE_LIMIT_ERROR,
            # Quota errors
            "QuotaExceededError": LLMErrorCode.QUOTA_EXCEEDED,
            "BillingError": LLMErrorCode.QUOTA_EXCEEDED,
            # Model errors
            "ModelNotFoundError": LLMErrorCode.MODEL_NOT_FOUND,
            "InvalidModelError": LLMErrorCode.MODEL_NOT_FOUND,
            # Timeout errors
            "TimeoutError": LLMErrorCode.TIMEOUT_ERROR,
            "RequestTimeoutError": LLMErrorCode.TIMEOUT_ERROR,
            # Network errors
            "ConnectionError": LLMErrorCode.NETWORK_ERROR,
            "NetworkError": LLMErrorCode.NETWORK_ERROR,
            "ConnectionTimeoutError": LLMErrorCode.NETWORK_ERROR,
        }

        # Try to match by exception type name first
        error_type_name = type(error).__name__
        error_code = error_mapping.get(error_type_name, LLMErrorCode.PROVIDER_ERROR)

        # Fallback to string matching for unknown exception types
        if error_code == LLMErrorCode.PROVIDER_ERROR:
            if (
                "authentication" in error_message.lower()
                or "unauthorized" in error_message.lower()
            ):
                error_code = LLMErrorCode.AUTHENTICATION_ERROR
            elif (
                "rate limit" in error_message.lower()
                or "too many requests" in error_message.lower()
            ):
                error_code = LLMErrorCode.RATE_LIMIT_ERROR
            elif "quota" in error_message.lower() or "billing" in error_message.lower():
                error_code = LLMErrorCode.QUOTA_EXCEEDED
            elif (
                "model" in error_message.lower()
                and "not found" in error_message.lower()
            ):
                error_code = LLMErrorCode.MODEL_NOT_FOUND
            elif "timeout" in error_message.lower():
                error_code = LLMErrorCode.TIMEOUT_ERROR
            elif (
                "network" in error_message.lower()
                or "connection" in error_message.lower()
            ):
                error_code = LLMErrorCode.NETWORK_ERROR

        return LLMError(
            message=f"{context}: {error_message}" if context else error_message,
            error_code=error_code,
            provider=self.provider_type.value,
            details={"original_error": str(error), "error_type": type(error).__name__},
        )
