"""Centralized error handling for LLM providers."""

from typing import Any

from ..domain.models import LLMError, LLMErrorCode, LLMProviderType


class LLMErrorHandler:
    """Centralized error handler for LLM providers."""

    # Common error patterns and their mappings
    COMMON_ERROR_PATTERNS = {
        # Authentication errors
        "authentication": LLMErrorCode.AUTHENTICATION_ERROR,
        "unauthorized": LLMErrorCode.AUTHENTICATION_ERROR,
        "invalid_api_key": LLMErrorCode.AUTHENTICATION_ERROR,
        "permission_denied": LLMErrorCode.AUTHENTICATION_ERROR,
        "invalid_credentials": LLMErrorCode.AUTHENTICATION_ERROR,
        "access_denied": LLMErrorCode.AUTHENTICATION_ERROR,
        # Rate limiting errors
        "rate_limit": LLMErrorCode.RATE_LIMIT_ERROR,
        "too_many_requests": LLMErrorCode.RATE_LIMIT_ERROR,
        "rate_limit_exceeded": LLMErrorCode.RATE_LIMIT_ERROR,
        "throttled": LLMErrorCode.RATE_LIMIT_ERROR,
        # Quota errors
        "quota": LLMErrorCode.QUOTA_EXCEEDED,
        "billing": LLMErrorCode.QUOTA_EXCEEDED,
        "insufficient_quota": LLMErrorCode.QUOTA_EXCEEDED,
        "quota_exceeded": LLMErrorCode.QUOTA_EXCEEDED,
        "resource_exhausted": LLMErrorCode.QUOTA_EXCEEDED,
        # Model errors
        "model_not_found": LLMErrorCode.MODEL_NOT_FOUND,
        "invalid_model": LLMErrorCode.MODEL_NOT_FOUND,
        "not_found": LLMErrorCode.MODEL_NOT_FOUND,
        "model_not_available": LLMErrorCode.MODEL_NOT_FOUND,
        # Timeout errors
        "timeout": LLMErrorCode.TIMEOUT_ERROR,
        "deadline_exceeded": LLMErrorCode.TIMEOUT_ERROR,
        "request_timeout": LLMErrorCode.TIMEOUT_ERROR,
        # Network errors
        "network": LLMErrorCode.NETWORK_ERROR,
        "connection": LLMErrorCode.NETWORK_ERROR,
        "unavailable": LLMErrorCode.NETWORK_ERROR,
        "service_unavailable": LLMErrorCode.NETWORK_ERROR,
        # Invalid request errors
        "invalid_argument": LLMErrorCode.INVALID_REQUEST,
        "bad_request": LLMErrorCode.INVALID_REQUEST,
        "validation_error": LLMErrorCode.INVALID_REQUEST,
        "invalid_request": LLMErrorCode.INVALID_REQUEST,
    }

    # Provider-specific error mappings
    PROVIDER_SPECIFIC_PATTERNS = {
        LLMProviderType.OPENAI: {
            "insufficient_quota": LLMErrorCode.QUOTA_EXCEEDED,
            "rate_limit_exceeded": LLMErrorCode.RATE_LIMIT_ERROR,
            "invalid_api_key": LLMErrorCode.AUTHENTICATION_ERROR,
            "model_not_found": LLMErrorCode.MODEL_NOT_FOUND,
        },
        LLMProviderType.ANTHROPIC: {
            "authentication_error": LLMErrorCode.AUTHENTICATION_ERROR,
            "rate_limit_error": LLMErrorCode.RATE_LIMIT_ERROR,
            "quota_exceeded": LLMErrorCode.QUOTA_EXCEEDED,
            "invalid_model": LLMErrorCode.MODEL_NOT_FOUND,
        },
        LLMProviderType.GOOGLE: {
            "permission_denied": LLMErrorCode.AUTHENTICATION_ERROR,
            "quota_exceeded": LLMErrorCode.QUOTA_EXCEEDED,
            "resource_exhausted": LLMErrorCode.QUOTA_EXCEEDED,
            "invalid_argument": LLMErrorCode.INVALID_REQUEST,
            "deadline_exceeded": LLMErrorCode.TIMEOUT_ERROR,
            "unavailable": LLMErrorCode.NETWORK_ERROR,
        },
    }

    @classmethod
    def handle_error(
        cls,
        error: Exception,
        provider: LLMProviderType,
        context: str = "",
        custom_patterns: dict[str, LLMErrorCode] | None = None,
    ) -> LLMError:
        """Handle LLM provider errors with standardized mapping.

        Args:
            error: The exception that occurred
            provider: The LLM provider type
            context: Additional context about where the error occurred
            custom_patterns: Provider-specific custom error patterns

        Returns:
            Standardized LLMError
        """
        error_message = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Start with common patterns
        error_code = cls._match_error_patterns(error_message, cls.COMMON_ERROR_PATTERNS)

        # Check provider-specific patterns
        if error_code == LLMErrorCode.PROVIDER_ERROR:
            provider_patterns = cls.PROVIDER_SPECIFIC_PATTERNS.get(provider, {})
            error_code = cls._match_error_patterns(error_message, provider_patterns)

        # Check custom patterns
        if error_code == LLMErrorCode.PROVIDER_ERROR and custom_patterns:
            error_code = cls._match_error_patterns(error_message, custom_patterns)

        # Try exception type name matching
        if error_code == LLMErrorCode.PROVIDER_ERROR:
            error_code = cls._match_error_patterns(
                error_type_name, cls.COMMON_ERROR_PATTERNS
            )

        # Default to provider error if no match found
        if error_code == LLMErrorCode.PROVIDER_ERROR:
            error_code = LLMErrorCode.PROVIDER_ERROR

        return LLMError(
            message=f"{context}: {str(error)}" if context else str(error),
            error_code=error_code,
            provider=provider.value,
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "context": context,
            },
        )

    @classmethod
    def _match_error_patterns(
        cls, text: str, patterns: dict[str, LLMErrorCode]
    ) -> LLMErrorCode:
        """Match error patterns against text.

        Args:
            text: Text to search for patterns
            patterns: Dictionary of patterns to error codes

        Returns:
            Matched error code or PROVIDER_ERROR if no match
        """
        for pattern, error_code in patterns.items():
            if pattern in text:
                return error_code
        return LLMErrorCode.PROVIDER_ERROR

    @classmethod
    def create_llm_error(
        cls,
        message: str,
        error_code: LLMErrorCode,
        provider: LLMProviderType,
        details: dict[str, Any] | None = None,
    ) -> LLMError:
        """Create a standardized LLMError.

        Args:
            message: Error message
            error_code: Standardized error code
            provider: LLM provider type
            details: Additional error details

        Returns:
            Standardized LLMError
        """
        return LLMError(
            message=message,
            error_code=error_code,
            provider=provider.value,
            details=details or {},
        )
