"""Google Gemini LLM provider implementation."""

import asyncio
import json
from typing import Any
from collections.abc import AsyncGenerator
import structlog

import google.generativeai as genai
from google.generativeai.types import (
    HarmCategory,
    HarmBlockThreshold,
)

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


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        # Configure the Google AI client
        genai.configure(api_key=config.get("api_key"))

        self.default_model = config.get("default_model", "gemini-1.5-pro")
        self.max_tokens = config.get("max_tokens", 8192)

        # Safety settings
        self.safety_settings = config.get(
            "safety_settings",
            {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        )

        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=self.max_tokens,
            candidate_count=1,
        )

    def _get_provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        return LLMProviderType.GOOGLE

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion response."""
        self._validate_request(request)

        try:
            # Get the model
            model = genai.GenerativeModel(
                model_name=request.model,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config,
            )

            # Convert messages to Gemini format
            content = self._convert_messages_to_content(request.messages)

            # Add tools if provided
            tools = None
            if request.tools:
                tools = self._convert_tools(request.tools)
                model = genai.GenerativeModel(
                    model_name=request.model,
                    tools=tools,
                    safety_settings=self.safety_settings,
                    generation_config=self.generation_config,
                )

            # Generate content
            response = await asyncio.to_thread(
                model.generate_content,
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens or self.max_tokens,
                ),
            )

            # Extract content
            content_text = ""
            tool_calls: list[dict[str, Any]] = []

            if response.text:
                content_text = response.text

            # Extract function calls if present
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_calls.append(
                                {
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(
                                            part.function_call.args
                                        ),
                                    },
                                }
                            )

            # Calculate usage (Gemini doesn't provide detailed token counts)
            usage = {
                "prompt_tokens": 0,  # Gemini doesn't provide this
                "completion_tokens": (
                    len(content_text.split()) if content_text else 0
                ),  # Rough estimate
                "total_tokens": len(content_text.split()) if content_text else 0,
            }

            metadata = {
                "finish_reason": getattr(response, "finish_reason", None),
                "tool_calls": tool_calls,
                "model": request.model,
                "safety_ratings": self._extract_safety_ratings(response),
                "candidates": (
                    len(response.candidates) if hasattr(response, "candidates") else 0
                ),
            }

            return self._create_response(
                content=content_text,
                model=request.model,
                usage=usage,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                "Google Gemini generation failed", error=str(e), model=request.model
            )
            raise self._handle_error(e, "Google Gemini generation") from e

    def stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        """Generate a streaming response."""

        async def _stream() -> AsyncGenerator[LLMStreamChunk, None]:
            self._validate_request(request)

            try:
                # Get the model
                model = genai.GenerativeModel(
                    model_name=request.model,
                    safety_settings=self.safety_settings,
                    generation_config=self.generation_config,
                )

                # Convert messages to Gemini format
                content = self._convert_messages_to_content(request.messages)

                # Add tools if provided
                if request.tools:
                    tools = self._convert_tools(request.tools)
                    model = genai.GenerativeModel(
                        model_name=request.model,
                        tools=tools,
                        safety_settings=self.safety_settings,
                        generation_config=self.generation_config,
                    )

                # Stream the response
                response_stream = await asyncio.to_thread(
                    model.generate_content,
                    content,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=request.temperature,
                        max_output_tokens=request.max_tokens or self.max_tokens,
                    ),
                )

                async for chunk in response_stream:
                    if hasattr(chunk, "text") and chunk.text:
                        yield self._create_stream_chunk(
                            content=chunk.text,
                            model=request.model,
                            is_final=False,
                            metadata={
                                "chunk_type": "text",
                                "safety_ratings": self._extract_safety_ratings(chunk),
                            },
                        )
                    elif hasattr(chunk, "candidates") and chunk.candidates:
                        # Check if this is the final chunk
                        is_final = any(
                            hasattr(candidate, "finish_reason")
                            and candidate.finish_reason
                            for candidate in chunk.candidates
                        )

                        if is_final:
                            yield self._create_stream_chunk(
                                content="",
                                model=request.model,
                                is_final=True,
                                metadata={
                                    "chunk_type": "final",
                                    "finish_reason": getattr(
                                        chunk.candidates[0], "finish_reason", None
                                    ),
                                },
                            )

            except Exception as e:
                logger.error(
                    "Google Gemini streaming failed", error=str(e), model=request.model
                )
                raise self._handle_error(e, "Google Gemini streaming") from e

        return _stream()

    async def get_models(self) -> list[ModelInfo]:
        """Get available models for this provider."""
        # Check cache first
        cached_models = await self._get_cached_models()
        if cached_models:
            return cached_models

        try:
            # Google has a fixed set of models
            models = [
                ModelInfo(
                    id="gemini-1.5-pro",
                    name="Gemini 1.5 Pro",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=8192,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Most capable Gemini 1.5 model for complex tasks",
                ),
                ModelInfo(
                    id="gemini-1.5-flash",
                    name="Gemini 1.5 Flash",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=8192,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Fast Gemini 1.5 model for most tasks",
                ),
                ModelInfo(
                    id="gemini-1.0-pro",
                    name="Gemini 1.0 Pro",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=30720,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Original Gemini 1.0 Pro model",
                ),
                ModelInfo(
                    id="gemini-1.0-pro-vision",
                    name="Gemini 1.0 Pro Vision",
                    provider=self.provider_type.value,
                    type=LLMModelType.CHAT,
                    max_tokens=16384,
                    supports_functions=True,
                    supports_streaming=True,
                    supports_vision=True,
                    description="Gemini 1.0 Pro with enhanced vision capabilities",
                ),
            ]

            # Cache the models
            await self._cache_models(models)
            return models

        except Exception as e:
            logger.error("Failed to get Google models", error=str(e))
            raise self._handle_error(e, "Google models fetch") from e

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Simple health check by making a minimal request
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = await asyncio.to_thread(
                model.generate_content,
                "Hello",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10),
            )
            return bool(response.text)
        except Exception as e:
            logger.warning("Google Gemini health check failed", error=str(e))
            return False

    def _convert_messages_to_content(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to Gemini content format."""
        content_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                content_parts.append(f"System: {content}")
            elif role == "user":
                content_parts.append(f"User: {content}")
            elif role == "assistant":
                content_parts.append(f"Assistant: {content}")
            else:
                content_parts.append(f"{role.title()}: {content}")

        return "\n\n".join(content_parts)

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tools to Gemini format."""
        converted = []
        for tool in tools:
            gemini_tool = {
                "function_declarations": [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"],
                    }
                ]
            }
            converted.append(gemini_tool)
        return converted

    def _extract_safety_ratings(self, response: Any) -> list[dict[str, Any]]:
        """Extract safety ratings from response."""
        safety_ratings = []

        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "safety_ratings"):
                    for rating in candidate.safety_ratings:
                        safety_ratings.append(
                            {
                                "category": str(rating.category),
                                "probability": str(rating.probability),
                            }
                        )

        return safety_ratings

    async def generate_with_vision(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-1.5-pro",
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with vision capabilities."""
        request = LLMRequest(messages=messages, model=model, **kwargs)
        return await self.generate(request)

    async def generate_embedding(
        self, text: str, model: str = "text-embedding-004"
    ) -> list[float]:
        """Generate text embedding."""
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=model,
                content=text,
                task_type="retrieval_document",
            )
            return list(result["embedding"])
        except Exception as e:
            logger.error("Google embedding failed", error=str(e), model=model)
            raise self._handle_error(e, "Google embedding") from e

    def _handle_error(self, error: Exception, context: str = "") -> LLMError:
        """Handle Google-specific errors."""
        error_message = str(error)

        # Map Google-specific errors
        if (
            "permission_denied" in error_message.lower()
            or "authentication" in error_message.lower()
        ):
            error_code = LLMErrorCode.AUTHENTICATION_ERROR
        elif (
            "quota_exceeded" in error_message.lower()
            or "resource_exhausted" in error_message.lower()
        ):
            error_code = LLMErrorCode.QUOTA_EXCEEDED
        elif "invalid_argument" in error_message.lower():
            error_code = LLMErrorCode.INVALID_REQUEST
        elif "not_found" in error_message.lower():
            error_code = LLMErrorCode.MODEL_NOT_FOUND
        elif "deadline_exceeded" in error_message.lower():
            error_code = LLMErrorCode.TIMEOUT_ERROR
        elif "unavailable" in error_message.lower():
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
