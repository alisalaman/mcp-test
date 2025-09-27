"""Domain models for AI Agent application."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ExternalServiceType(str, Enum):
    """Types of external services."""

    LLM_PROVIDER = "llm_provider"
    MCP_SERVER = "mcp_server"
    DATABASE = "database"
    CACHE = "cache"
    SECRET_MANAGER = "secret_manager"
    MESSAGE_QUEUE = "message_queue"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Base Models
class TimestampedModel(BaseModel):
    """Base model with timestamps."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IdentifiedModel(TimestampedModel):
    """Base model with ID and timestamps."""

    id: UUID = Field(default_factory=uuid4)


# Domain Entities
class Message(IdentifiedModel):
    """Conversation message."""

    session_id: UUID
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    parent_message_id: UUID | None = None


class Session(IdentifiedModel):
    """User conversation session."""

    user_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Agent(IdentifiedModel):
    """AI agent configuration."""

    name: str
    description: str | None = None
    system_prompt: str | None = None
    llm_config: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(IdentifiedModel):
    """Tool available to agents."""

    name: str
    description: str
    tool_schema: dict[str, Any]
    mcp_server_id: UUID | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPServer(IdentifiedModel):
    """MCP server configuration."""

    name: str
    description: str | None = None
    endpoint: str
    authentication: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    health_check_url: str | None = None
    enabled: bool = True


class ExternalService(IdentifiedModel):
    """External service configuration."""

    name: str
    service_type: ExternalServiceType
    endpoint: str
    authentication: dict[str, Any] = Field(default_factory=dict)
    retry_config: dict[str, Any] = Field(default_factory=dict)
    circuit_breaker_config: dict[str, Any] = Field(default_factory=dict)
    health_check_url: str | None = None
    enabled: bool = True


# Error Handling Models
class ErrorCode(str, Enum):
    """Standardized error codes."""

    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    TIMEOUT_ERROR = "timeout_error"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retry_after: int | None = None
    correlation_id: str | None = None


# Request/Response Models
class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    user_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateMessageRequest(BaseModel):
    """Request to create a new message."""

    content: str
    role: MessageRole = MessageRole.USER
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionRequest(BaseModel):
    """Request for agent execution."""

    session_id: UUID
    message: str
    agent_id: UUID | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionResponse(BaseModel):
    """Response from agent execution."""

    session_id: UUID
    message_id: UUID
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: int


# Resilience Models
class RetryConfig(BaseModel):
    """Retry configuration for external services."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    retryable_errors: list[str] = Field(default_factory=list)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: list[str] = Field(default_factory=list)
    fallback_enabled: bool = True


class ServiceHealth(BaseModel):
    """Service health status."""

    service_name: str
    service_type: ExternalServiceType
    status: str
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
