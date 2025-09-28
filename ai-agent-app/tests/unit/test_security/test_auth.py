"""Tests for authentication service."""

import pytest
from ai_agent.security.auth import (
    AuthenticationService,
    JWTManager,
    APIKeyManager,
    User,
    UserRole,
    TokenType,
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
    InvalidCredentialsError,
)


class TestJWTManager:
    """Test JWT token management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.secret_key = "test-secret-key-for-jwt-testing-only"
        self.jwt_manager = JWTManager(secret_key=self.secret_key)

    def test_create_access_token(self):
        """Test access token creation."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        token = self.jwt_manager.create_access_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        token = self.jwt_manager.create_refresh_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_valid(self):
        """Test valid token verification."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        token = self.jwt_manager.create_access_token(user)
        payload = self.jwt_manager.verify_token(token)

        assert payload["sub"] == "user123"
        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == TokenType.ACCESS.value

    def test_verify_token_invalid(self):
        """Test invalid token verification."""
        with pytest.raises(InvalidTokenError):
            self.jwt_manager.verify_token("invalid.token.here")

    def test_verify_token_expired(self):
        """Test expired token verification."""
        # Create a JWT manager with very short expiry
        short_jwt_manager = JWTManager(
            secret_key=self.secret_key,
            access_token_expire_minutes=0,  # Expires immediately
        )

        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        token = short_jwt_manager.create_access_token(user)

        # Wait a bit to ensure token is expired
        import time

        time.sleep(1)

        with pytest.raises(TokenExpiredError):
            short_jwt_manager.verify_token(token)

    def test_refresh_access_token(self):
        """Test access token refresh."""
        user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        refresh_token = self.jwt_manager.create_refresh_token(user)
        new_access_token = self.jwt_manager.refresh_access_token(refresh_token)

        assert isinstance(new_access_token, str)
        assert len(new_access_token) > 0

        # Verify the new token
        payload = self.jwt_manager.verify_token(new_access_token)
        assert payload["sub"] == "user123"
        assert payload["type"] == TokenType.ACCESS.value


class TestAPIKeyManager:
    """Test API key management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key_manager = APIKeyManager()

    def test_generate_api_key(self):
        """Test API key generation."""
        key = self.api_key_manager.generate_api_key()
        assert isinstance(key, str)
        assert key.startswith("sk-")
        assert len(key) > 32

    def test_create_api_key(self):
        """Test API key creation."""
        api_key = self.api_key_manager.create_api_key(
            name="test-key",
            user_id="user123",
            permissions=["read", "write"],
        )

        assert api_key.name == "test-key"
        assert api_key.user_id == "user123"
        assert api_key.permissions == ["read", "write"]
        assert api_key.is_active is True

    def test_validate_api_key(self):
        """Test API key validation."""
        api_key = self.api_key_manager.create_api_key(
            name="test-key",
            user_id="user123",
        )

        # Valid key
        validated_key = self.api_key_manager.validate_api_key(api_key.key)
        assert validated_key is not None
        assert validated_key.id == api_key.id

        # Invalid key
        invalid_key = self.api_key_manager.validate_api_key("invalid-key")
        assert invalid_key is None

    def test_revoke_api_key(self):
        """Test API key revocation."""
        api_key = self.api_key_manager.create_api_key(
            name="test-key",
            user_id="user123",
        )

        # Revoke the key
        result = self.api_key_manager.revoke_api_key(api_key.key)
        assert result is True

        # Key should no longer be valid
        validated_key = self.api_key_manager.validate_api_key(api_key.key)
        assert validated_key is None

    def test_list_user_api_keys(self):
        """Test listing user API keys."""
        # Create multiple keys for the same user
        key1 = self.api_key_manager.create_api_key("key1", "user123")
        key2 = self.api_key_manager.create_api_key("key2", "user123")
        key3 = self.api_key_manager.create_api_key("key3", "user456")

        # List keys for user123
        user_keys = self.api_key_manager.list_user_api_keys("user123")
        assert len(user_keys) == 2
        assert key1 in user_keys
        assert key2 in user_keys
        assert key3 not in user_keys


class TestAuthenticationService:
    """Test authentication service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.secret_key = "test-secret-key-for-auth-testing-only"
        self.auth_service = AuthenticationService(secret_key=self.secret_key)
        # Clear secure storage before each test
        self.auth_service.secure_storage._cache.clear()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear secure storage after each test
        self.auth_service.secure_storage._cache.clear()

    def test_hash_password(self):
        """Test password hashing."""
        password = "testpassword123"
        hashed = self.auth_service.hash_password(password)

        assert isinstance(hashed, str)
        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password(self):
        """Test password verification."""
        password = "testpassword123"
        hashed = self.auth_service.hash_password(password)

        # Correct password
        assert self.auth_service.verify_password(password, hashed) is True

        # Wrong password
        assert self.auth_service.verify_password("wrongpassword", hashed) is False

    def test_create_user(self):
        """Test user creation."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            roles=[UserRole.USER],
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert UserRole.USER in user.roles
        assert user.is_active is True

    def test_create_user_duplicate(self):
        """Test duplicate user creation."""
        self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        with pytest.raises(AuthenticationError):
            self.auth_service.create_user(
                username="testuser",
                email="test2@example.com",
                password="testpassword123",
            )

    def test_authenticate_user(self):
        """Test user authentication."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        # Correct credentials
        authenticated_user = self.auth_service.authenticate_user(
            "testuser", "testpassword123"
        )
        assert authenticated_user is not None
        assert authenticated_user.id == user.id

        # Wrong password
        authenticated_user = self.auth_service.authenticate_user(
            "testuser", "wrongpassword"
        )
        assert authenticated_user is None

        # Wrong username
        authenticated_user = self.auth_service.authenticate_user(
            "wronguser", "testpassword123"
        )
        assert authenticated_user is None

    def test_login(self):
        """Test user login."""
        self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        # Successful login
        tokens = self.auth_service.login("testuser", "testpassword123")
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"
        assert tokens.expires_in > 0

        # Failed login
        with pytest.raises(InvalidCredentialsError):
            self.auth_service.login("testuser", "wrongpassword")

    def test_verify_access_token(self):
        """Test access token verification."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        tokens = self.auth_service.login("testuser", "testpassword123")
        verified_user = self.auth_service.verify_access_token(tokens.access_token)

        assert verified_user.id == user.id
        assert verified_user.username == user.username

    def test_verify_access_token_invalid(self):
        """Test invalid access token verification."""
        with pytest.raises(InvalidTokenError):
            self.auth_service.verify_access_token("invalid.token.here")

    def test_create_api_key(self):
        """Test API key creation."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        api_key = self.auth_service.create_api_key(
            user_id=user.id,
            name="test-key",
            permissions=["read", "write"],
        )

        assert api_key.name == "test-key"
        assert api_key.user_id == user.id
        assert api_key.permissions == ["read", "write"]

    def test_verify_api_key(self):
        """Test API key verification."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        api_key = self.auth_service.create_api_key(
            user_id=user.id,
            name="test-key",
        )

        # Valid API key
        verified_user = self.auth_service.verify_api_key(api_key.key)
        assert verified_user is not None
        assert verified_user.id == user.id

        # Invalid API key
        verified_user = self.auth_service.verify_api_key("invalid-key")
        assert verified_user is None

    def test_get_user_permissions(self):
        """Test user permission retrieval."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            roles=[UserRole.ADMIN],
        )

        permissions = self.auth_service.get_user_permissions(user)
        assert "users:read" in permissions
        assert "users:write" in permissions
        assert "agents:read" in permissions

    def test_deactivate_user(self):
        """Test user deactivation."""
        user = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
        )

        # Create an API key for the user
        api_key = self.auth_service.create_api_key(
            user_id=user.id,
            name="test-key",
        )

        # Deactivate user
        result = self.auth_service.deactivate_user(user.id)
        assert result is True

        # User should be inactive
        deactivated_user = self.auth_service.get_user(user.id)
        assert deactivated_user.is_active is False

        # API key should be revoked
        verified_user = self.auth_service.verify_api_key(api_key.key)
        assert verified_user is None
