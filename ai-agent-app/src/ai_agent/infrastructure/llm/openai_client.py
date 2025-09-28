"""OpenAI LLM provider implementation."""

from typing import Any
from collections.abc import AsyncGenerator
import structlog

from openai import AsyncOpenAI

from .base import (
    BaseLLMProvider,
    LLMError,
    LLMErrorCode,
    LLMProviderType,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMModelType,
    ModelInfo,
)

logger = structlog.get_logger()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            timeout=config.get("timeout", 30.0),
            max_retries=config.get("max_retries", 3),
        )
        self.default_model = config.get("default_model", "gpt-4o")
        self.azure_endpoint = config.get("azure_endpoint")
        self.azure_api_version = config.get("azure_api_version", "2024-02-15-preview")
        self.azure_deployment = config.get("azure_deployment")

    def _get_provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        if self.azure_endpoint:
            return LLMProviderType.AZURE_OPENAI
        return LLMProviderType.OPENAI

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion response."""
        self._validate_request(request)

        try:
            # Prepare OpenAI request
            openai_request = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            # Add tools if provided
            if request.tools:
                openai_request["tools"] = request.tools
                if request.tool_choice:
                    openai_request["tool_choice"] = request.tool_choice

            # Make the request
            response = await self.client.chat.completions.create(**openai_request)

            # Extract content and usage
            content = response.choices[0].message.content or ""
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else {}
            )

            # Extract tool calls if present
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

            metadata = {
                "finish_reason": response.choices[0].finish_reason,
                "tool_calls": tool_calls,
                "model": response.model,
                "created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }

            return self._create_response(
                content=content, model=response.model, usage=usage, metadata=metadata
            )

        except Exception as e:
            logger.error("OpenAI generation failed", error=str(e), model=request.model)
            raise self._handle_error(e, "OpenAI generation") from e

    def stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response."""

        async def _stream() -> AsyncGenerator[LLMStreamChunk, None]:
            self._validate_request(request)

            try:
                # Prepare OpenAI request
                openai_request = {
                    "model": request.model,
                    "messages": request.messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": True,
                }

                # Add tools if provided
                if request.tools:
                    openai_request["tools"] = request.tools
                    if request.tool_choice:
                        openai_request["tool_choice"] = request.tool_choice

                # Stream the response
                stream = await self.client.chat.completions.create(**openai_request)

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield self._create_stream_chunk(
                            content=chunk.choices[0].delta.content,
                            model=chunk.model,
                            is_final=chunk.choices[0].finish_reason is not None,
                            metadata={
                                "finish_reason": chunk.choices[0].finish_reason,
                                "index": chunk.choices[0].index,
                            },
                        )

            except Exception as e:
                logger.error(
                    "OpenAI streaming failed", error=str(e), model=request.model
                )
                raise self._handle_error(e, "OpenAI streaming") from e

        return _stream()

    async def get_models(self) -> list[ModelInfo]:
        """Get available models for this provider."""
        # Check cache first
        cached_models = await self._get_cached_models()
        if cached_models:
            return cached_models

        try:
            # Fetch models from OpenAI
            response = await self.client.models.list()
            models = []

            for model in response.data:
                # Determine model type based on model ID
                model_type = LLMModelType.CHAT
                if "embedding" in model.id.lower():
                    model_type = LLMModelType.EMBEDDING
                elif "gpt-3.5" in model.id or "gpt-4" in model.id:
                    model_type = LLMModelType.CHAT

                # Check capabilities
                supports_functions = "gpt-3.5" in model.id or "gpt-4" in model.id
                supports_streaming = True  # Most OpenAI models support streaming
                supports_vision = (
                    "vision" in model.id.lower() or "gpt-4-vision" in model.id
                )

                model_info = ModelInfo(
                    id=model.id,
                    name=model.id,
                    provider=self.provider_type.value,
                    type=model_type,
                    max_tokens=self._get_model_max_tokens(model.id),
                    supports_functions=supports_functions,
                    supports_streaming=supports_streaming,
                    supports_vision=supports_vision,
                    description=f"OpenAI {model.id} model",
                )
                models.append(model_info)

            # Cache the models
            await self._cache_models(models)
            return models

        except Exception as e:
            logger.error("Failed to fetch OpenAI models", error=str(e))
            raise self._handle_error(e, "OpenAI models fetch") from e

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Simple health check by listing models
            await self.client.models.list()
            return True
        except Exception as e:
            logger.warning("OpenAI health check failed", error=str(e))
            return False

    def _get_model_max_tokens(self, model_id: str) -> int | None:
        """Get max tokens for a specific model."""
        # Model-specific token limits
        token_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-vision": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4o-2024-07-18": 128000,
        }

        # Check for exact match first
        if model_id in token_limits:
            return token_limits[model_id]

        # Check for partial matches
        for model_pattern, tokens in token_limits.items():
            if model_pattern in model_id:
                return tokens

        # Default fallback
        return 4096

    async def generate_embedding(
        self, text: str, model: str = "text-embedding-ada-002"
    ) -> list[float]:
        """Generate text embedding."""
        try:
            response = await self.client.embeddings.create(model=model, input=text)
            return list(response.data[0].embedding)
        except Exception as e:
            logger.error("OpenAI embedding failed", error=str(e), model=model)
            raise self._handle_error(e, "OpenAI embedding") from e

    async def generate_with_vision(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4-vision-preview",
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with vision capabilities."""
        request = LLMRequest(messages=messages, model=model, **kwargs)
        return await self.generate(request)

    def _handle_error(self, error: Exception, context: str = "") -> LLMError:
        """Handle OpenAI-specific errors."""
        error_message = str(error)

        # Map OpenAI-specific errors
        if "insufficient_quota" in error_message.lower():
            error_code = LLMErrorCode.QUOTA_EXCEEDED
        elif "rate_limit_exceeded" in error_message.lower():
            error_code = LLMErrorCode.RATE_LIMIT_ERROR
        elif "invalid_api_key" in error_message.lower():
            error_code = LLMErrorCode.AUTHENTICATION_ERROR
        elif "model_not_found" in error_message.lower():
            error_code = LLMErrorCode.MODEL_NOT_FOUND
        elif "timeout" in error_message.lower():
            error_code = LLMErrorCode.TIMEOUT_ERROR
        else:
            # Use parent error handling
            return super()._handle_error(error, context)

        return LLMError(
            message=f"{context}: {error_message}" if context else error_message,
            error_code=error_code,
            provider=self.provider_type.value,
            details={"original_error": str(error), "error_type": type(error).__name__},
        )
