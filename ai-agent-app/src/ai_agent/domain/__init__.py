"""Domain layer for AI Agent application.

This module contains the core business entities, value objects, and domain logic.
"""

from .exceptions import (
    AIAgentException,
    AuthenticationException,
    AuthorizationException,
    CircuitBreakerOpenException,
    ExternalServiceException,
    RateLimitException,
    TimeoutException,
    ValidationException,
)
from .models import (
    Agent,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentStatus,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CreateMessageRequest,
    CreateSessionRequest,
    ErrorCode,
    ErrorDetail,
    ExternalService,
    ExternalServiceType,
    IdentifiedModel,
    MCPServer,
    Message,
    MessageRole,
    RetryConfig,
    ServiceHealth,
    Session,
    TimestampedModel,
    Tool,
)

__all__ = [
    # Models
    "Agent",
    "AgentExecutionRequest",
    "AgentExecutionResponse",
    "AgentStatus",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CreateMessageRequest",
    "CreateSessionRequest",
    "ErrorCode",
    "ErrorDetail",
    "ExternalService",
    "ExternalServiceType",
    "IdentifiedModel",
    "MCPServer",
    "Message",
    "MessageRole",
    "RetryConfig",
    "ServiceHealth",
    "Session",
    "TimestampedModel",
    "Tool",
    # Exceptions
    "AIAgentException",
    "AuthenticationException",
    "AuthorizationException",
    "CircuitBreakerOpenException",
    "ExternalServiceException",
    "RateLimitException",
    "TimeoutException",
    "ValidationException",
]
