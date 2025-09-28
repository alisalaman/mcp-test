"""MCP (Model Context Protocol) implementation."""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
import structlog

logger = structlog.get_logger()


class MCPMessageType(str, Enum):
    """MCP message types."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(str, Enum):
    """MCP protocol methods."""

    # Initialization
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"

    # Completion
    COMPLETION = "completion/complete"


class MCPErrorCode(str, Enum):
    """MCP error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000
    TIMEOUT = -32001
    CONNECTION_ERROR = -32002
    AUTHENTICATION_ERROR = -32003
    AUTHORIZATION_ERROR = -32004


@dataclass
class MCPMessage:
    """Base MCP message."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {"jsonrpc": self.jsonrpc}

        if self.id is not None:
            data["id"] = self.id

        if self.method is not None:
            data["method"] = self.method

        if self.params is not None:
            data["params"] = self.params

        if self.result is not None:
            data["result"] = self.result

        if self.error is not None:
            data["error"] = self.error

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPMessage":
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "MCPMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.method is None:
            raise ValueError("Method is required for MCP request")


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""

    def __post_init__(self) -> None:
        if self.id is None:
            raise ValueError("ID is required for MCP response")
        if self.result is None and self.error is None:
            raise ValueError("Either result or error is required for MCP response")


@dataclass
class MCPNotification(MCPMessage):
    """MCP notification message."""

    def __post_init__(self) -> None:
        if self.method is None:
            raise ValueError("Method is required for MCP notification")
        if self.id is not None:
            raise ValueError("ID should not be set for MCP notification")


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(
        self,
        code: MCPErrorCode,
        message: str,
        data: Any | None = None,
        request_id: str | int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.request_id = request_id

    def to_response(self) -> MCPResponse:
        """Convert to MCP response."""
        return MCPResponse(
            id=self.request_id,
            error={"code": self.code.value, "message": self.message, "data": self.data},
        )


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
class MCPResource:
    """MCP resource definition."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MCPPrompt:
    """MCP prompt definition."""

    name: str
    description: str
    arguments: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MCPServerCapabilities:
    """MCP server capabilities."""

    tools: bool = False
    resources: bool = False
    prompts: bool = False
    logging: bool = False
    completion: bool = False
    experimental: dict[str, Any] | None = None


@dataclass
class MCPClientCapabilities:
    """MCP client capabilities."""

    experimental: dict[str, Any] | None = None


class MCPHandler(ABC):
    """Abstract base class for MCP message handlers."""

    @abstractmethod
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request."""
        pass

    @abstractmethod
    async def handle_notification(self, notification: MCPNotification) -> None:
        """Handle an MCP notification."""
        pass


class MCPClient:
    """MCP protocol client implementation."""

    def __init__(self, client_name: str = "ai-agent", client_version: str = "1.0.0"):
        self.client_name = client_name
        self.client_version = client_version
        self.capabilities = MCPClientCapabilities()
        self.server_capabilities: MCPServerCapabilities | None = None
        self.handlers: dict[str, MCPHandler] = {}
        self.pending_requests: dict[str | int, asyncio.Future[MCPResponse]] = {}
        self._request_counter = 0
        self._lock = asyncio.Lock()

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> MCPResponse:
        """Send an MCP request and wait for response."""
        request_id = await self._get_next_request_id()
        request = MCPRequest(id=request_id, method=method, params=params or {})

        # Create future for response
        future: asyncio.Future[MCPResponse] = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            # Send the request (implemented by subclasses)
            await self._send_message(request)

            # Wait for response with timeout
            response: MCPResponse = await asyncio.wait_for(future, timeout=timeout)

            # Check for errors
            if response.error:
                error_code = MCPErrorCode(
                    response.error.get("code", MCPErrorCode.INTERNAL_ERROR)
                )
                raise MCPError(
                    code=error_code,
                    message=response.error.get("message", "Unknown error"),
                    data=response.error.get("data"),
                    request_id=request_id,
                )

            return response

        except TimeoutError as e:
            self.pending_requests.pop(request_id, None)
            raise MCPError(
                code=MCPErrorCode.TIMEOUT,
                message=f"Request {request_id} timed out",
                request_id=request_id,
            ) from e
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            if isinstance(e, MCPError):
                raise
            raise MCPError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message=f"Request failed: {str(e)}",
                request_id=request_id,
            ) from e

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send an MCP notification."""
        notification = MCPNotification(method=method, params=params or {})
        await self._send_message(notification)

    async def handle_message(self, message: MCPMessage) -> None:
        """Handle an incoming MCP message."""
        if isinstance(message, MCPRequest):
            await self._handle_request(message)
        elif isinstance(message, MCPResponse):
            await self._handle_response(message)
        elif isinstance(message, MCPNotification):
            await self._handle_notification(message)
        else:
            logger.warning(
                "Unknown message type received", message_type=type(message).__name__
            )

    async def _handle_request(self, request: MCPRequest) -> None:
        """Handle an incoming request."""
        try:
            # Find handler for the method
            handler = self.handlers.get(request.method or "")
            if not handler:
                error_response = MCPResponse(
                    id=request.id,
                    error={
                        "code": MCPErrorCode.METHOD_NOT_FOUND.value,
                        "message": f"Method '{request.method}' not found",
                    },
                )
                await self._send_message(error_response)
                return

            # Handle the request
            response = await handler.handle_request(request)
            response.id = request.id
            await self._send_message(response)

        except Exception as e:
            logger.error("Error handling request", method=request.method, error=str(e))
            error_response = MCPResponse(
                id=request.id,
                error={
                    "code": MCPErrorCode.INTERNAL_ERROR.value,
                    "message": f"Internal error: {str(e)}",
                },
            )
            await self._send_message(error_response)

    async def _handle_response(self, response: MCPResponse) -> None:
        """Handle an incoming response."""
        future = self.pending_requests.pop(response.id or "", None)
        if future and not future.done():
            future.set_result(response)
        else:
            logger.warning(
                "Received response for unknown request", request_id=response.id
            )

    async def _handle_notification(self, notification: MCPNotification) -> None:
        """Handle an incoming notification."""
        handler = self.handlers.get(notification.method or "")
        if handler:
            try:
                await handler.handle_notification(notification)
            except Exception as e:
                logger.error(
                    "Error handling notification",
                    method=notification.method,
                    error=str(e),
                )

    def register_handler(self, method: str, handler: MCPHandler) -> None:
        """Register a handler for a specific method."""
        self.handlers[method] = handler

    def unregister_handler(self, method: str) -> None:
        """Unregister a handler for a specific method."""
        self.handlers.pop(method, None)

    async def _get_next_request_id(self) -> str:
        """Get the next request ID."""
        async with self._lock:
            self._request_counter += 1
            return str(self._request_counter)

    @abstractmethod
    async def _send_message(self, message: MCPMessage) -> None:
        """Send a message to the server (implemented by subclasses)."""
        pass

    async def initialize(self) -> None:
        """Initialize the MCP connection."""
        try:
            # Send initialize request
            response = await self.send_request(
                method=MCPMethod.INITIALIZE,
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
                    "clientInfo": {
                        "name": self.client_name,
                        "version": self.client_version,
                    },
                },
            )

            # Store server capabilities
            if response.result and "capabilities" in response.result:
                capabilities_data = response.result["capabilities"]
                self.server_capabilities = MCPServerCapabilities(
                    tools=capabilities_data.get("tools", {}).get("listChanged", False),
                    resources=capabilities_data.get("resources", {}).get(
                        "listChanged", False
                    ),
                    prompts=capabilities_data.get("prompts", {}).get(
                        "listChanged", False
                    ),
                    logging=capabilities_data.get("logging", {}).get("level", False),
                    completion=capabilities_data.get("completion", {}).get(
                        "completionList", False
                    ),
                    experimental=capabilities_data.get("experimental"),
                )

            # Send initialized notification
            await self.send_notification(method=MCPMethod.INITIALIZED)

            logger.info(
                "MCP client initialized", server_capabilities=self.server_capabilities
            )

        except Exception as e:
            logger.error("Failed to initialize MCP client", error=str(e))
            raise

    async def list_tools(self) -> list[MCPTool]:
        """List available tools."""
        if not self.server_capabilities or not self.server_capabilities.tools:
            return []

        try:
            response = await self.send_request(method=MCPMethod.TOOLS_LIST)
            tools = []

            if response.result and "tools" in response.result:
                for tool_data in response.result["tools"]:
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        input_schema=tool_data["inputSchema"],
                        metadata=tool_data.get("metadata"),
                    )
                    tools.append(tool)

            return tools

        except Exception as e:
            logger.error("Failed to list tools", error=str(e))
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool."""
        if not self.server_capabilities or not self.server_capabilities.tools:
            raise MCPError(
                code=MCPErrorCode.METHOD_NOT_FOUND,
                message="Tools not supported by server",
            )

        try:
            response = await self.send_request(
                method=MCPMethod.TOOLS_CALL,
                params={"name": name, "arguments": arguments},
            )

            return response.result or {}

        except Exception as e:
            logger.error("Failed to call tool", tool=name, error=str(e))
            raise

    async def list_resources(self) -> list[MCPResource]:
        """List available resources."""
        if not self.server_capabilities or not self.server_capabilities.resources:
            return []

        try:
            response = await self.send_request(method=MCPMethod.RESOURCES_LIST)
            resources = []

            if response.result and "resources" in response.result:
                for resource_data in response.result["resources"]:
                    resource = MCPResource(
                        uri=resource_data["uri"],
                        name=resource_data["name"],
                        description=resource_data.get("description"),
                        mime_type=resource_data.get("mimeType"),
                        metadata=resource_data.get("metadata"),
                    )
                    resources.append(resource)

            return resources

        except Exception as e:
            logger.error("Failed to list resources", error=str(e))
            return []

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource."""
        if not self.server_capabilities or not self.server_capabilities.resources:
            raise MCPError(
                code=MCPErrorCode.METHOD_NOT_FOUND,
                message="Resources not supported by server",
            )

        try:
            response = await self.send_request(
                method=MCPMethod.RESOURCES_READ, params={"uri": uri}
            )

            return response.result or {}

        except Exception as e:
            logger.error("Failed to read resource", uri=uri, error=str(e))
            raise

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts."""
        if not self.server_capabilities or not self.server_capabilities.prompts:
            return []

        try:
            response = await self.send_request(method=MCPMethod.PROMPTS_LIST)
            prompts = []

            if response.result and "prompts" in response.result:
                for prompt_data in response.result["prompts"]:
                    prompt = MCPPrompt(
                        name=prompt_data["name"],
                        description=prompt_data["description"],
                        arguments=prompt_data.get("arguments"),
                        metadata=prompt_data.get("metadata"),
                    )
                    prompts.append(prompt)

            return prompts

        except Exception as e:
            logger.error("Failed to list prompts", error=str(e))
            return []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get a prompt."""
        if not self.server_capabilities or not self.server_capabilities.prompts:
            raise MCPError(
                code=MCPErrorCode.METHOD_NOT_FOUND,
                message="Prompts not supported by server",
            )

        try:
            params: dict[str, Any] = {"name": name}
            if arguments:
                params["arguments"] = arguments

            response = await self.send_request(
                method=MCPMethod.PROMPTS_GET, params=params
            )

            return response.result or {}

        except Exception as e:
            logger.error("Failed to get prompt", prompt=name, error=str(e))
            raise
