"""Tool execution framework with security and sandboxing."""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4
import structlog

from .tool_registry import RegisteredTool, ToolExecutionResult, ToolStatus

logger = structlog.get_logger()


class SecurityLevel(str, Enum):
    """Security levels for tool execution."""

    LOW = "low"  # No restrictions
    MEDIUM = "medium"  # Basic input validation
    HIGH = "high"  # Sandboxed execution
    CRITICAL = "critical"  # Maximum restrictions


class ExecutionEnvironment(str, Enum):
    """Execution environments."""

    HOST = "host"  # Execute on host system
    CONTAINER = "container"  # Execute in container
    VIRTUAL_MACHINE = "vm"  # Execute in VM
    SANDBOX = "sandbox"  # Execute in sandbox


@dataclass
class SecurityPolicy:
    """Security policy for tool execution."""

    level: SecurityLevel = SecurityLevel.MEDIUM
    environment: ExecutionEnvironment = ExecutionEnvironment.HOST
    allowed_operations: set[str] = field(default_factory=set)
    blocked_operations: set[str] = field(default_factory=set)
    max_execution_time: float = 30.0
    max_memory_mb: int = 512
    max_file_size_mb: int = 10
    allowed_file_extensions: set[str] = field(default_factory=set)
    blocked_file_extensions: set[str] = field(
        default_factory=lambda: {
            ".exe",
            ".bat",
            ".cmd",
            ".sh",
            ".ps1",
            ".py",
            ".js",
            ".php",
        }
    )
    network_access: bool = False
    file_system_access: bool = True
    environment_variables: dict[str, str] = field(default_factory=dict)
    custom_restrictions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for tool execution."""

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    session_id: str | None = None
    security_policy: SecurityPolicy = field(default_factory=SecurityPolicy)
    working_directory: str | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class ExecutionMetrics:
    """Metrics for tool execution."""

    execution_id: str
    tool_name: str
    start_time: float
    end_time: float | None = None
    success: bool = False
    error_message: str | None = None
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    files_created: int = 0
    files_modified: int = 0
    files_deleted: int = 0


class SecurityValidator:
    """Validates tool execution requests against security policies."""

    def __init__(self) -> None:
        self._dangerous_patterns = [
            r"rm\s+-rf",  # Dangerous file deletion
            r"sudo\s+",  # Privilege escalation
            r"chmod\s+777",  # Dangerous permissions
            r"wget\s+.*\|.*sh",  # Pipe to shell
            r"curl\s+.*\|.*sh",  # Pipe to shell
            r"eval\s*\(",  # Code evaluation
            r"exec\s*\(",  # Code execution
            r"system\s*\(",  # System calls
            r"shell_exec\s*\(",  # Shell execution
        ]
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._dangerous_patterns
        ]

    def validate_input(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> list[str]:
        """Validate tool input against security policy."""
        violations = []

        # Check security level
        if context.security_policy.level == SecurityLevel.CRITICAL:
            violations.extend(
                self._validate_critical_security(tool, arguments, context)
            )
        elif context.security_policy.level == SecurityLevel.HIGH:
            violations.extend(self._validate_high_security(tool, arguments, context))
        elif context.security_policy.level == SecurityLevel.MEDIUM:
            violations.extend(self._validate_medium_security(tool, arguments, context))

        return violations

    def _validate_critical_security(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> list[str]:
        """Validate for critical security level."""
        violations = []

        # Check for dangerous patterns in string arguments
        for key, value in arguments.items():
            if isinstance(value, str):
                for pattern in self._compiled_patterns:
                    if pattern.search(value):
                        violations.append(
                            f"Dangerous pattern detected in argument '{key}': {value}"
                        )

        # Check file operations
        if "file" in arguments or "path" in arguments:
            violations.append("File operations not allowed in critical security mode")

        # Check network operations
        if any(key in arguments for key in ["url", "endpoint", "host", "port"]):
            violations.append(
                "Network operations not allowed in critical security mode"
            )

        return violations

    def _validate_high_security(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> list[str]:
        """Validate for high security level."""
        violations = []

        # Check for dangerous patterns
        for key, value in arguments.items():
            if isinstance(value, str):
                for pattern in self._compiled_patterns:
                    if pattern.search(value):
                        violations.append(
                            f"Potentially dangerous pattern in argument '{key}': {value}"
                        )

        # Check file operations
        if "file" in arguments or "path" in arguments:
            if not context.security_policy.file_system_access:
                violations.append("File system access not allowed")

        # Check network operations
        if any(key in arguments for key in ["url", "endpoint", "host", "port"]):
            if not context.security_policy.network_access:
                violations.append("Network access not allowed")

        return violations

    def _validate_medium_security(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> list[str]:
        """Validate for medium security level."""
        violations = []

        # Basic input validation
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > 10000:  # 10KB limit
                violations.append(f"Argument '{key}' exceeds maximum length")

        return violations


class SandboxExecutor:
    """Executes tools in a sandboxed environment."""

    def __init__(self) -> None:
        self._active_executions: dict[str, asyncio.Task[None]] = {}
        self._execution_metrics: dict[str, ExecutionMetrics] = {}

    async def execute_tool(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute a tool in a sandboxed environment."""
        start_time = asyncio.get_event_loop().time()

        # Create execution metrics
        metrics = ExecutionMetrics(
            execution_id=context.execution_id,
            tool_name=tool.name,
            start_time=start_time,
        )
        self._execution_metrics[context.execution_id] = metrics

        try:
            # Choose execution method based on security policy
            if context.security_policy.environment == ExecutionEnvironment.CONTAINER:
                result = await self._execute_in_container(tool, arguments, context)
            elif (
                context.security_policy.environment
                == ExecutionEnvironment.VIRTUAL_MACHINE
            ):
                result = await self._execute_in_vm(tool, arguments, context)
            elif context.security_policy.environment == ExecutionEnvironment.SANDBOX:
                result = await self._execute_in_sandbox(tool, arguments, context)
            else:
                result = await self._execute_on_host(tool, arguments, context)

            # Update metrics
            end_time = asyncio.get_event_loop().time()
            metrics.end_time = end_time
            metrics.success = result.success
            metrics.error_message = result.error

            return result

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            metrics.end_time = end_time
            metrics.success = False
            metrics.error_message = str(e)

            logger.error("Tool execution failed", tool_name=tool.name, error=str(e))

            return ToolExecutionResult(
                success=False, error=str(e), execution_time=end_time - start_time
            )
        finally:
            # Clean up execution tracking
            self._active_executions.pop(context.execution_id, None)

    async def _execute_in_container(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute tool in a container."""
        # Implementation for container-based execution
        # This would use Docker or similar container runtime
        raise NotImplementedError("Container execution not yet implemented")

    async def _execute_in_vm(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute tool in a virtual machine."""
        # Implementation for VM-based execution
        raise NotImplementedError("VM execution not yet implemented")

    async def _execute_in_sandbox(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute tool in a sandbox."""
        # Implementation for sandbox-based execution
        # This would use a restricted execution environment
        raise NotImplementedError("Sandbox execution not yet implemented")

    async def _execute_on_host(
        self, tool: RegisteredTool, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute tool on the host system."""
        # This is a simplified implementation
        # In practice, this would delegate to the MCP connection manager
        start_time = asyncio.get_event_loop().time()

        try:
            # Simulate tool execution
            # In real implementation, this would call the MCP tool
            await asyncio.sleep(0.1)  # Simulate execution time

            result = {
                "status": "success",
                "output": f"Tool '{tool.name}' executed with arguments: {arguments}",
                "execution_id": context.execution_id,
            }

            execution_time = asyncio.get_event_loop().time() - start_time

            return ToolExecutionResult(
                success=True, result=result, execution_time=execution_time
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolExecutionResult(
                success=False, error=str(e), execution_time=execution_time
            )

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        if execution_id in self._active_executions:
            task = self._active_executions[execution_id]
            task.cancel()
            return True
        return False

    async def get_execution_metrics(self, execution_id: str) -> ExecutionMetrics | None:
        """Get metrics for an execution."""
        return self._execution_metrics.get(execution_id)


class ToolExecutor:
    """Main tool executor with security and monitoring."""

    def __init__(self) -> None:
        self.security_validator = SecurityValidator()
        self.sandbox_executor = SandboxExecutor()
        self._execution_history: list[ExecutionMetrics] = []
        self._max_history_size = 1000
        self._lock = asyncio.Lock()

    async def execute_tool(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ExecutionContext | None = None,
    ) -> ToolExecutionResult:
        """Execute a tool with security validation and monitoring."""
        if context is None:
            context = ExecutionContext()

        # Validate security
        violations = self.security_validator.validate_input(tool, arguments, context)
        if violations:
            return ToolExecutionResult(
                success=False, error=f"Security violations: {', '.join(violations)}"
            )

        # Check tool status
        if tool.status != ToolStatus.AVAILABLE:
            return ToolExecutionResult(
                success=False,
                error=f"Tool '{tool.name}' is not available (status: {tool.status})",
            )

        # Execute the tool
        result = await self.sandbox_executor.execute_tool(tool, arguments, context)

        # Update execution history
        async with self._lock:
            if context.execution_id in self.sandbox_executor._execution_metrics:
                metrics = self.sandbox_executor._execution_metrics[context.execution_id]
                self._execution_history.append(metrics)

                # Trim history if too large
                if len(self._execution_history) > self._max_history_size:
                    self._execution_history = self._execution_history[
                        -self._max_history_size :
                    ]

        return result

    async def get_execution_history(
        self,
        tool_name: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[ExecutionMetrics]:
        """Get execution history with filtering."""
        async with self._lock:
            history = self._execution_history.copy()

        # Apply filters
        if tool_name:
            history = [m for m in history if m.tool_name == tool_name]

        if user_id:
            # This would require user_id to be stored in metrics
            pass

        # Return most recent executions
        return sorted(history, key=lambda m: m.start_time, reverse=True)[:limit]

    async def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        async with self._lock:
            total_executions = len(self._execution_history)
            successful_executions = sum(1 for m in self._execution_history if m.success)
            failed_executions = total_executions - successful_executions

            if total_executions > 0:
                success_rate = successful_executions / total_executions
            else:
                success_rate = 0.0

            # Average execution time
            if self._execution_history:
                avg_execution_time = sum(
                    (m.end_time or m.start_time) - m.start_time
                    for m in self._execution_history
                ) / len(self._execution_history)
            else:
                avg_execution_time = 0.0

            # Tool usage statistics
            tool_usage: dict[str, int] = {}
            for metrics in self._execution_history:
                tool_name = metrics.tool_name
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "tool_usage": tool_usage,
                "most_used_tools": sorted(
                    tool_usage.items(), key=lambda x: x[1], reverse=True
                )[:10],
            }

    async def create_security_policy(
        self,
        level: SecurityLevel,
        environment: ExecutionEnvironment = ExecutionEnvironment.HOST,
        **kwargs: Any,
    ) -> SecurityPolicy:
        """Create a security policy with custom settings."""
        policy = SecurityPolicy(level=level, environment=environment)

        # Apply custom settings
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        return policy

    async def create_execution_context(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        security_policy: SecurityPolicy | None = None,
        **kwargs: Any,
    ) -> ExecutionContext:
        """Create an execution context with custom settings."""
        context = ExecutionContext(
            user_id=user_id,
            session_id=session_id,
            security_policy=security_policy or SecurityPolicy(),
        )

        # Apply custom settings
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)

        return context
