"""Authentication service implementation."""

import os
import secrets
from datetime import datetime, timedelta, UTC
from typing import Any
from enum import Enum

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from ..observability.logging import get_logger
from .secure_storage import get_secure_storage, SecureStorage

logger = get_logger(__name__)


class TokenType(str, Enum):
    """Token types."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class UserRole(str, Enum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


class AuthenticationError(Exception):
    """Authentication error."""

    pass


class TokenExpiredError(AuthenticationError):
    """Token expired error."""

    pass


class InvalidTokenError(AuthenticationError):
    """Invalid token error."""

    pass


class UserNotFoundError(AuthenticationError):
    """User not found error."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid credentials error."""

    pass


class User(BaseModel):
    """User model."""

    id: str
    username: str
    email: str
    roles: list[UserRole] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_login: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Token(BaseModel):
    """Token model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: datetime


class APIKey(BaseModel):
    """API Key model."""

    id: str
    name: str
    key: str
    user_id: str
    permissions: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    last_used: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JWTManager:
    """JWT token management."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.logger = get_logger(__name__)

    def create_access_token(self, user: User) -> str:
        """Create access token for user."""
        now = datetime.now(UTC)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "type": TokenType.ACCESS.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_urlsafe(32),
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self.logger.info(f"Created access token for user: {user.username}")
        return str(token)

    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user."""
        now = datetime.now(UTC)
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user.id,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_urlsafe(32),
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self.logger.info(f"Created refresh token for user: {user.username}")
        return str(token)

    def create_tokens(self, user: User) -> Token:
        """Create both access and refresh tokens."""
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60,
            expires_at=datetime.now(UTC)
            + timedelta(minutes=self.access_token_expire_minutes),
        )

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if not isinstance(payload, dict):
                raise InvalidTokenError("Token payload is not a dictionary")
            return dict(payload)
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)

        if payload.get("type") != TokenType.REFRESH.value:
            raise InvalidTokenError("Invalid refresh token")

        # Create new user object from token payload
        user = User(
            id=payload["sub"],
            username=payload.get("username", ""),
            email=payload.get("email", ""),
            roles=[UserRole(role) for role in payload.get("roles", [])],
        )

        return self.create_access_token(user)

    def get_token_expiry(self, token: str) -> datetime | None:
        """Get token expiry time."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, tz=UTC)
        except jwt.InvalidTokenError:
            pass
        return None


class APIKeyManager:
    """API Key management."""

    def __init__(self, key_length: int = 32, key_prefix: str = "sk-"):
        self.key_length = key_length
        self.key_prefix = key_prefix
        self.logger = get_logger(__name__)
        self._keys: dict[str, APIKey] = {}  # In production, this would be in a database

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        key = secrets.token_urlsafe(self.key_length)
        return f"{self.key_prefix}{key}"

    def create_api_key(
        self,
        name: str,
        user_id: str,
        permissions: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> APIKey:
        """Create a new API key."""
        key_id = secrets.token_urlsafe(16)
        key = self.generate_api_key()

        api_key = APIKey(
            id=key_id,
            name=name,
            key=key,
            user_id=user_id,
            permissions=permissions or [],
            expires_at=expires_at,
        )

        self._keys[key] = api_key
        self.logger.info(f"Created API key: {name} for user: {user_id}")
        return api_key

    def get_api_key(self, key: str) -> APIKey | None:
        """Get API key by key value."""
        return self._keys.get(key)

    def validate_api_key(self, key: str) -> APIKey | None:
        """Validate API key."""
        api_key = self.get_api_key(key)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
            return None

        # Update last used
        api_key.last_used = datetime.now(UTC)

        return api_key

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        api_key = self.get_api_key(key)
        if api_key:
            api_key.is_active = False
            self.logger.info(f"Revoked API key: {api_key.name}")
            return True
        return False

    def list_user_api_keys(self, user_id: str) -> list[APIKey]:
        """List API keys for a user."""
        return [
            key
            for key in self._keys.values()
            if key.user_id == user_id and key.is_active
        ]

    def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys."""
        now = datetime.now(UTC)
        expired_keys = []

        for key, api_key in self._keys.items():
            if api_key.expires_at and api_key.expires_at < now:
                expired_keys.append(key)

        for key in expired_keys:
            del self._keys[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired API keys")

        return len(expired_keys)


class AuthenticationService:
    """Main authentication service."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        secure_storage: SecureStorage | None = None,
    ):
        self.jwt_manager = JWTManager(
            secret_key=secret_key,
            algorithm=algorithm,
            access_token_expire_minutes=access_token_expire_minutes,
            refresh_token_expire_days=refresh_token_expire_days,
        )
        self.api_key_manager = APIKeyManager()
        self.password_context = CryptContext(schemes=["argon2"], deprecated="auto")
        self.logger = get_logger(__name__)
        self.secure_storage = secure_storage or get_secure_storage()

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return str(self.password_context.hash(password))

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password."""
        return bool(self.password_context.verify(plain_password, hashed_password))

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: list[UserRole] | None = None,
    ) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = self.get_user_by_username(username)
        if existing_user:
            raise AuthenticationError(f"User '{username}' already exists") from None

        user_id = secrets.token_urlsafe(16)
        hashed_password = self.hash_password(password)

        user = User(
            id=user_id, username=username, email=email, roles=roles or [UserRole.USER]
        )

        # Store user data securely
        self.secure_storage.store(f"user:{user_id}", user.model_dump())
        self.secure_storage.store(
            f"credentials:{username}", {"password": hashed_password}
        )

        self.logger.info(f"Created user: {username}")
        return user

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        user_data = self.secure_storage.retrieve(f"user:{user_id}")
        if user_data:
            return User(**user_data)
        return None

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        # Search through all users to find by username
        for key in self.secure_storage.list_keys():
            if key.startswith("user:"):
                user_data = self.secure_storage.retrieve(key)
                if user_data and user_data.get("username") == username:
                    return User(**user_data)
        return None

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user with username and password."""
        user = self.get_user_by_username(username)
        if not user:
            return None

        if not user.is_active:
            return None

        credentials_data = self.secure_storage.retrieve(f"credentials:{username}")
        if not credentials_data:
            return None

        hashed_password = credentials_data.get("password")
        if not hashed_password:
            return None

        if not self.verify_password(password, hashed_password):
            return None

        # Update last login
        user.last_login = datetime.now(UTC)
        self.secure_storage.store(f"user:{user.id}", user.model_dump())

        self.logger.info(f"User authenticated: {username}")
        return user

    def login(self, username: str, password: str) -> Token:
        """Login user and return tokens."""
        user = self.authenticate_user(username, password)
        if not user:
            raise InvalidCredentialsError("Invalid username or password")

        return self.jwt_manager.create_tokens(user)

    def refresh_token(self, refresh_token: str) -> str:
        """Refresh access token."""
        return self.jwt_manager.refresh_access_token(refresh_token)

    def verify_access_token(self, token: str) -> User:
        """Verify access token and return user."""
        payload = self.jwt_manager.verify_token(token)

        if payload.get("type") != TokenType.ACCESS.value:
            raise InvalidTokenError("Invalid access token")

        user_id = payload["sub"]
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError("User not found")

        if not user.is_active:
            raise AuthenticationError("User is not active")

        return user

    def verify_api_key(self, api_key: str) -> User | None:
        """Verify API key and return user."""
        key = self.api_key_manager.validate_api_key(api_key)
        if not key:
            return None

        user = self.get_user(key.user_id)
        if not user or not user.is_active:
            return None

        return user

    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> APIKey:
        """Create API key for user."""
        user = self.get_user(user_id)
        if not user:
            raise UserNotFoundError("User not found")

        return self.api_key_manager.create_api_key(
            name=name, user_id=user_id, permissions=permissions, expires_at=expires_at
        )

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        return self.api_key_manager.revoke_api_key(api_key)

    def list_user_api_keys(self, user_id: str) -> list[APIKey]:
        """List API keys for user."""
        return self.api_key_manager.list_user_api_keys(user_id)

    def update_user_roles(self, user_id: str, roles: list[UserRole]) -> bool:
        """Update user roles."""
        user = self.get_user(user_id)
        if not user:
            return False

        user.roles = roles
        user.updated_at = datetime.now(UTC)

        # Update stored user data
        self.secure_storage.store(f"user:{user_id}", user.model_dump())

        self.logger.info(
            f"Updated roles for user {user.username}: {[r.value for r in roles]}"
        )
        return True

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user."""
        user = self.get_user(user_id)
        if not user:
            return False

        user.is_active = False
        user.updated_at = datetime.now(UTC)

        # Update stored user data
        self.secure_storage.store(f"user:{user_id}", user.model_dump())

        # Revoke all API keys
        api_keys = self.list_user_api_keys(user_id)
        for api_key in api_keys:
            self.revoke_api_key(api_key.key)

        self.logger.info(f"Deactivated user: {user.username}")
        return True

    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens (placeholder - in production, tokens would be stored in database)."""
        # This would clean up expired tokens from a database
        return 0

    def get_user_permissions(self, user: User) -> list[str]:
        """Get user permissions based on roles."""
        permissions = []

        for role in user.roles:
            if role == UserRole.ADMIN:
                permissions.extend(
                    [
                        "users:read",
                        "users:write",
                        "users:delete",
                        "agents:read",
                        "agents:write",
                        "agents:delete",
                        "sessions:read",
                        "sessions:write",
                        "sessions:delete",
                        "messages:read",
                        "messages:write",
                        "messages:delete",
                        "tools:read",
                        "tools:write",
                        "tools:delete",
                        "api_keys:read",
                        "api_keys:write",
                        "api_keys:delete",
                    ]
                )
            elif role == UserRole.USER:
                permissions.extend(
                    [
                        "agents:read",
                        "agents:write",
                        "sessions:read",
                        "sessions:write",
                        "messages:read",
                        "messages:write",
                        "tools:read",
                    ]
                )
            elif role == UserRole.READONLY:
                permissions.extend(
                    ["agents:read", "sessions:read", "messages:read", "tools:read"]
                )
            elif role == UserRole.SERVICE:
                permissions.extend(
                    [
                        "agents:read",
                        "agents:write",
                        "sessions:read",
                        "sessions:write",
                        "messages:read",
                        "messages:write",
                        "tools:read",
                    ]
                )

        return list(set(permissions))


# Global authentication service instance
_auth_service: AuthenticationService | None = None


def get_auth_service() -> AuthenticationService:
    """Get global authentication service instance."""
    global _auth_service
    if _auth_service is None:
        # Get secret key from environment variable
        secret_key = os.getenv("SECURITY_SECRET_KEY")
        if not secret_key:
            raise ValueError("SECURITY_SECRET_KEY environment variable is required")
        if len(secret_key) < 32:
            raise ValueError("SECURITY_SECRET_KEY must be at least 32 characters")
        _auth_service = AuthenticationService(secret_key=secret_key)
    return _auth_service


def setup_authentication(
    secret_key: str,
    algorithm: str = "HS256",
    access_token_expire_minutes: int = 30,
    refresh_token_expire_days: int = 7,
) -> AuthenticationService:
    """Setup global authentication service."""
    global _auth_service
    _auth_service = AuthenticationService(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_minutes=access_token_expire_minutes,
        refresh_token_expire_days=refresh_token_expire_days,
    )
    return _auth_service
