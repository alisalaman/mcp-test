"""Security middleware for FastAPI application."""

import asyncio
import time
from typing import Any
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from ..observability.logging import get_logger
from .auth import (
    get_auth_service,
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
)
from .authorization import (
    get_authz_service,
)

logger = get_logger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class CORSConfig:
    """CORS configuration."""

    origins: list[str] | None = None
    methods: list[str] | None = None
    headers: list[str] | None = None
    allow_credentials: bool = True
    max_age: int = 600

    def __post_init__(self) -> None:
        if self.origins is None:
            self.origins = ["http://localhost:3000", "http://localhost:8000"]
        if self.methods is None:
            self.methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.headers is None:
            self.headers = ["*"]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_minute: int = 100
    burst_size: int = 10
    window_size: int = 60  # seconds
    block_duration: int = 300  # seconds


@dataclass
class SecurityHeaders:
    """Security headers configuration."""

    hsts_max_age: int = 31536000  # 1 year
    content_type_nosniff: bool = True
    x_frame_options: str = "DENY"
    x_content_type_options: bool = True
    referrer_policy: str = "strict-origin-when-cross-origin"
    content_security_policy: str = "default-src 'self'"
    permissions_policy: str = "geolocation=(), microphone=(), camera=()"


class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware."""

    def __init__(
        self,
        app: Any,
        security_level: SecurityLevel = SecurityLevel.DEVELOPMENT,
        cors_config: CORSConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        security_headers: SecurityHeaders | None = None,
        enable_auth: bool = True,
        enable_rate_limiting: bool = True,
        enable_security_headers: bool = True,
    ):
        super().__init__(app)
        self.security_level = security_level
        self.cors_config = cors_config or CORSConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.security_headers = security_headers or SecurityHeaders()
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_security_headers = enable_security_headers

        # Rate limiting storage
        self._rate_limit_storage: dict[str, dict[str, Any]] = {}
        self._rate_limit_lock = asyncio.Lock()

        # Auth service
        self.auth_service = get_auth_service()
        self.authz_service = get_authz_service()

        self.logger = get_logger(__name__)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through security middleware."""
        start_time = time.time()

        try:
            # Add security headers
            if self.enable_security_headers:
                await self._add_security_headers(request)

            # Rate limiting
            if self.enable_rate_limiting and self.rate_limit_config.enabled:
                await self._check_rate_limit(request)

            # Authentication (for protected endpoints)
            if self.enable_auth and self._is_protected_endpoint(request):
                await self._authenticate_request(request)

            # Process request
            response = await call_next(request)

            # Add response security headers
            if self.enable_security_headers:
                self._add_response_security_headers(response)

            # Log security event
            duration = time.time() - start_time
            await self._log_security_event(request, response, duration)

            return response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            self.logger.error(f"Security middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error",
            )

    async def _add_security_headers(self, request: Request) -> None:
        """Add security headers to request."""
        # This would add security headers to the request context
        pass

    def _add_response_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        # HSTS
        if self.security_level == SecurityLevel.PRODUCTION:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.security_headers.hsts_max_age}"
            )

        # Content Type Options
        if self.security_headers.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # Frame Options
        response.headers["X-Frame-Options"] = self.security_headers.x_frame_options

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            self.security_headers.content_security_policy
        )

        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            self.security_headers.permissions_policy
        )

        # Referrer Policy
        response.headers["Referrer-Policy"] = self.security_headers.referrer_policy

    async def _check_rate_limit(self, request: Request) -> None:
        """Check rate limiting for request."""
        if not self.rate_limit_config.enabled:
            return

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        async with self._rate_limit_lock:
            # Clean up old entries
            await self._cleanup_rate_limit_storage(current_time)

            # Check rate limit
            if client_ip in self._rate_limit_storage:
                client_data = self._rate_limit_storage[client_ip]

                # Check if client is blocked
                if client_data.get("blocked_until", 0) > current_time:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Please try again later.",
                    )

                # Check request count
                window_start = current_time - self.rate_limit_config.window_size
                recent_requests = [
                    req_time
                    for req_time in client_data.get("requests", [])
                    if req_time > window_start
                ]

                if len(recent_requests) >= self.rate_limit_config.requests_per_minute:
                    # Block client
                    client_data["blocked_until"] = (
                        current_time + self.rate_limit_config.block_duration
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Please try again later.",
                    )

                # Add current request
                recent_requests.append(current_time)
                client_data["requests"] = recent_requests
            else:
                # New client
                self._rate_limit_storage[client_ip] = {
                    "requests": [current_time],
                    "blocked_until": 0,
                }

    async def _cleanup_rate_limit_storage(self, current_time: float) -> None:
        """Clean up old rate limit entries."""
        cutoff_time = current_time - (self.rate_limit_config.window_size * 2)

        for client_ip, client_data in list(self._rate_limit_storage.items()):
            # Remove old requests
            recent_requests = [
                req_time
                for req_time in client_data.get("requests", [])
                if req_time > cutoff_time
            ]

            if (
                not recent_requests
                and client_data.get("blocked_until", 0) < current_time
            ):
                # Remove inactive clients
                del self._rate_limit_storage[client_ip]
            else:
                client_data["requests"] = recent_requests

    def _is_protected_endpoint(self, request: Request) -> bool:
        """Check if endpoint requires authentication."""
        # Skip authentication for health checks, metrics, and docs
        skip_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/auth/register",
        ]

        return not any(request.url.path.startswith(path) for path in skip_paths)

    async def _authenticate_request(self, request: Request) -> None:
        """Authenticate the request."""
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
            )

        try:
            # Check if it's a Bearer token or API key
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user = self.auth_service.verify_access_token(token)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid JWT token",
                    )
                # At this point, user is guaranteed to be User, not None
                request.state.user = user
                request.state.auth_type = "jwt"
            elif auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]
                user = self.auth_service.verify_api_key(api_key)  # type: ignore[assignment]
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key",
                    )
                # At this point, user is guaranteed to be User, not None
                request.state.user = user
                request.state.auth_type = "api_key"
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization header format",
                )

        except (TokenExpiredError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    async def _log_security_event(
        self, request: Request, response: Response, duration: float
    ) -> None:
        """Log security event."""
        event_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "duration_ms": duration * 1000,
            "timestamp": time.time(),
        }

        # Add user info if authenticated
        if hasattr(request.state, "user") and request.state.user is not None:
            event_data["user_id"] = request.state.user.id
            event_data["auth_type"] = getattr(request.state, "auth_type", "unknown")

        # Log based on security level
        if self.security_level == SecurityLevel.PRODUCTION:
            if response.status_code >= 400:
                self.logger.warning("Security event", extra=event_data)
            else:
                self.logger.info("Security event", extra=event_data)
        else:
            self.logger.debug("Security event", extra=event_data)


class InputSanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)  # type: ignore[unreachable]

        # Remove null bytes and control characters
        sanitized = "".join(
            char for char in value if ord(char) >= 32 or char in "\t\n\r"
        )

        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    @staticmethod
    def sanitize_dict(
        data: dict[str, Any], max_string_length: int = 1000
    ) -> dict[str, Any]:
        """Sanitize dictionary input."""
        sanitized = {}

        for key, value in data.items():
            # Sanitize key
            clean_key = InputSanitizer.sanitize_string(str(key), 100)

            # Sanitize value
            if isinstance(value, str):
                clean_value: Any = InputSanitizer.sanitize_string(
                    value, max_string_length
                )
            elif isinstance(value, dict):
                clean_value = InputSanitizer.sanitize_dict(value, max_string_length)
            elif isinstance(value, list):
                clean_value = [
                    (
                        InputSanitizer.sanitize_string(str(item), max_string_length)
                        if isinstance(item, str)
                        else item
                    )
                    for item in value
                ]
            else:
                clean_value = value

            sanitized[clean_key] = clean_value

        return sanitized

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_password_strength(password: str) -> dict[str, Any]:
        """Validate password strength."""
        result: dict[str, Any] = {"is_valid": True, "score": 0, "issues": []}

        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not any(c.isupper() for c in password):
            result["issues"].append(
                "Password must contain at least one uppercase letter"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not any(c.islower() for c in password):
            result["issues"].append(
                "Password must contain at least one lowercase letter"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not any(c.isdigit() for c in password):
            result["issues"].append("Password must contain at least one digit")
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result["issues"].append(
                "Password must contain at least one special character"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        return result


class SecurityValidator:
    """Security validation utilities."""

    @staticmethod
    def validate_request_size(
        request: Request, max_size: int = 10 * 1024 * 1024
    ) -> bool:
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length:
            return int(content_length) <= max_size
        return True

    @staticmethod
    def validate_content_type(request: Request, allowed_types: list[str]) -> bool:
        """Validate content type."""
        content_type = request.headers.get("content-type", "")
        return any(allowed_type in content_type for allowed_type in allowed_types)

    @staticmethod
    def validate_origin(request: Request, allowed_origins: list[str]) -> bool:
        """Validate request origin."""
        origin = request.headers.get("origin")
        if not origin:
            return True  # No origin header, assume same-origin

        return origin in allowed_origins

    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection."""
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)",
        ]

        import re

        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS."""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]

        import re

        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False


def create_cors_middleware(cors_config: CORSConfig) -> dict[str, Any]:
    """Create CORS middleware configuration."""
    return {
        "allow_origins": cors_config.origins or [],
        "allow_credentials": cors_config.allow_credentials,
        "allow_methods": cors_config.methods or [],
        "allow_headers": cors_config.headers or [],
        "max_age": cors_config.max_age,
    }


def create_security_middleware(
    security_level: SecurityLevel = SecurityLevel.DEVELOPMENT,
    cors_config: CORSConfig | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    security_headers: SecurityHeaders | None = None,
) -> SecurityMiddleware:
    """Create security middleware with configuration."""
    return SecurityMiddleware(
        app=None,  # Will be set by FastAPI
        security_level=security_level,
        cors_config=cors_config,
        rate_limit_config=rate_limit_config,
        security_headers=security_headers,
    )
