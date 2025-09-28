"""Anthropic LLM provider implementation."""

import json
from typing import Any
from collections.abc import AsyncGenerator
import structlog

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

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


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            timeout=config.get("timeout", 30.0),
            max_retries=config.get("max_retries", 3),
        )
        self.default_model = config.get("default_model", "claude-3-5-sonnet-20241022")
        self.max_tokens = config.get("max_tokens", 4096)

    def _get_provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        return LLMProviderType.ANTHROPIC

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion response."""
        self._validate_request(request)

        try:
            # Convert messages to Anthropic format
            messages = self._convert_messages(request.messages)

            # Prepare Anthropic request
            anthropic_request = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens,
                "temperature": request.temperature,
            }

            # Add tools if provided
            if request.tools:
                anthropic_request["tools"] = self._convert_tools(request.tools)
                if request.tool_choice:
                    anthropic_request["tool_choice"] = request.tool_choice

            # Make the request
            response = await self.client.messages.create(**anthropic_request)

            # Extract content
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )

            # Calculate usage (Anthropic doesn't provide detailed token counts)
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0),
            }

            metadata = {
                "stop_reason": response.stop_reason,
                "tool_calls": tool_calls,
                "model": response.model,
                "role": response.role,
                "usage": response.usage.model_dump() if response.usage else {},
            }

            return self._create_response(
                content=content, model=response.model, usage=usage, metadata=metadata
            )

        except Exception as e:
            logger.error(
                "Anthropic generation failed", error=str(e), model=request.model
            )
            raise self._handle_error(e, "Anthropic generation") from e

    def stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response."""

        async def _stream() -> AsyncGenerator[LLMStreamChunk, None]:
            self._validate_request(request)

            try:
                # Convert messages to Anthropic format
                messages = self._convert_messages(request.messages)

                # Prepare Anthropic request
                anthropic_request = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens or self.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                }

                # Add tools if provided
                if request.tools:
                    anthropic_request["tools"] = self._convert_tools(request.tools)
                    if request.tool_choice:
                        anthropic_request["tool_choice"] = request.tool_choice

                # Stream the response
                async with self.client.messages.stream(**anthropic_request) as stream:
                    async for event in stream:
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text") and event.delta.text:
                                yield self._create_stream_chunk(
                                    content=event.delta.text,
                                    model=request.model,
                                    is_final=False,
                                    metadata={"type": event.type, "index": event.index},
                                )
                        elif event.type == "message_stop":
                            yield self._create_stream_chunk(
                                content="",
                                model=request.model,
                                is_final=True,
                                metadata={
                                    "type": event.type,
                                    "stop_reason": getattr(event, "stop_reason", None),
                                },
                            )

            except Exception as e:
                logger.error(
                    "Anthropic streaming failed", error=str(e), model=request.model
                )
                raise self._handle_error(e, "Anthropic streaming") from e

        return _stream()

    async def get_models(self) -> list[ModelInfo]:
        """Get available models for this provider."""
        # Check cache first
        cached_models = await self._get_cached_models()
        if cached_models:
            return cached_models

        try:
            # Anthropic has a fixed set of models
            models = [
                ModelInfo(
                    id="claude-3-opus-20240229",
                    name="Claude 3 Opus",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=4096,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Most powerful Claude 3 model for complex tasks",
                ),
                ModelInfo(
                    id="claude-3-sonnet-20240229",
                    name="Claude 3 Sonnet",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=4096,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Balanced Claude 3 model for most tasks",
                ),
                ModelInfo(
                    id="claude-3-haiku-20240307",
                    name="Claude 3 Haiku",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=4096,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Fastest Claude 3 model for simple tasks",
                ),
                ModelInfo(
                    id="claude-3-5-sonnet-20241022",
                    name="Claude 3.5 Sonnet",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=8192,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Latest Claude 3.5 model with improved capabilities",
                ),
            ]

            # Cache the models
            await self._cache_models(models)
            return models

        except Exception as e:
            logger.error("Failed to get Anthropic models", error=str(e))
            raise self._handle_error(e, "Anthropic models fetch") from e

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Simple health check by making a minimal request
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            logger.warning("Anthropic health check failed", error=str(e))
            return False

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[MessageParam]:
        """Convert messages to Anthropic format."""
        converted: list[MessageParam] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # Anthropic handles system messages differently
                # We'll prepend system messages to the first user message
                if converted and converted[0]["role"] == "user":
                    converted[0][
                        "content"
                    ] = f"System: {content}\n\n{converted[0]['content']}"
                else:
                    # If no user message yet, create one
                    converted.append({"role": "user", "content": f"System: {content}"})
            else:
                converted.append({"role": role, "content": content})

        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[ToolParam]:
        """Convert tools to Anthropic format."""
        converted = []
        for tool in tools:
            anthropic_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"],
            }
            converted.append(anthropic_tool)
        return converted

    async def generate_with_vision(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with vision capabilities."""
        request = LLMRequest(messages=messages, model=model, **kwargs)
        return await self.generate(request)

    def _handle_error(self, error: Exception, context: str = "") -> LLMError:
        """Handle Anthropic-specific errors."""
        error_message = str(error)

        # Map Anthropic-specific errors
        if "authentication_error" in error_message.lower():
            error_code = LLMErrorCode.AUTHENTICATION_ERROR
        elif "rate_limit_error" in error_message.lower():
            error_code = LLMErrorCode.RATE_LIMIT_ERROR
        elif "quota_exceeded" in error_message.lower():
            error_code = LLMErrorCode.QUOTA_EXCEEDED
        elif "invalid_model" in error_message.lower():
            error_code = LLMErrorCode.MODEL_NOT_FOUND
        elif "timeout" in error_message.lower():
            error_code = LLMErrorCode.TIMEOUT_ERROR
        elif "network" in error_message.lower():
            error_code = LLMErrorCode.NETWORK_ERROR
        else:
            # Use parent error handling
            return super()._handle_error(error, context)

        return LLMError(
            message=f"{context}: {error_message}" if context else error_message,
            error_code=error_code,
            provider=self.provider_type.value,
            details={"original_error": str(error), "error_type": type(error).__name__},
        )
