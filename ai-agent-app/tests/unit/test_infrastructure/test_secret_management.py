"""Tests for secret management infrastructure."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC
from ai_agent.infrastructure.secret_management.base import (
    SecretValue,
    SecretMetadata,
    SecretNotFoundError,
    SecretValidationError,
    SecretProviderType,
)
from ai_agent.infrastructure.secret_management.local_secrets import LocalSecretProvider
from ai_agent.infrastructure.secret_management.factory import (
    SecretManagerFactory,
    get_secret_manager,
)


class TestSecretValue:
    """Test SecretValue model."""

    def test_secret_value_creation(self):
        """Test SecretValue creation."""
        now = datetime.now(UTC)
        metadata = SecretMetadata(
            name="test_secret",
            provider="local",
            version="v1",
            created_at=now,
        )
        secret = SecretValue(
            value="secret_value",
            metadata=metadata,
            retrieved_at=now,
        )

        assert secret.metadata.name == "test_secret"
        assert secret.value == "secret_value"
        assert secret.metadata.version == "v1"
        assert secret.retrieved_at == now
        assert secret.metadata.provider == "local"

    def test_secret_value_serialization(self):
        """Test SecretValue serialization."""
        now = datetime.now(UTC)
        metadata = SecretMetadata(
            name="test_secret",
            provider="local",
            version="v1",
            created_at=now,
        )
        secret = SecretValue(
            value="secret_value",
            metadata=metadata,
            retrieved_at=now,
        )

        data = secret.model_dump()
        assert data["metadata"]["name"] == "test_secret"
        assert data["value"] == "secret_value"
        assert data["metadata"]["version"] == "v1"
        assert data["metadata"]["provider"] == "local"


class TestSecretMetadata:
    """Test SecretMetadata model."""

    def test_secret_metadata_creation(self):
        """Test SecretMetadata creation."""
        now = datetime.now(UTC)
        metadata = SecretMetadata(
            name="test_secret",
            provider="local",
            created_at=now,
            updated_at=now,
            version="v1",
            tags={"env": "test"},
            rotation_enabled=True,
            next_rotation=now,
            size_bytes=100,
        )

        assert metadata.name == "test_secret"
        assert metadata.provider == "local"
        assert metadata.created_at == now
        assert metadata.updated_at == now
        assert metadata.version == "v1"
        assert metadata.tags == {"env": "test"}
        assert metadata.rotation_enabled is True
        assert metadata.next_rotation == now
        assert metadata.size_bytes == 100


class TestLocalSecretProvider:
    """Test local secret provider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = LocalSecretProvider(
            config={
                "secrets_dir": "/tmp/test_secrets",
                "encryption_key": "bJ9OXfWDFTXxPr7GoPXRaOHbQdvTskJ30ztcp1M78K4=",
            }
        )

    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.unlink")
    @pytest.mark.asyncio
    async def test_connect(self, mock_unlink, mock_write_text):
        """Test provider connection."""
        # Should not raise an exception
        await self.provider.connect()

        mock_write_text.assert_called_once_with("test")
        mock_unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test provider disconnection."""
        # Should not raise an exception
        await self.provider.disconnect()

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    @pytest.mark.asyncio
    async def test_get_secret_success(self, mock_exists, mock_open):
        """Test successful secret retrieval."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = '{"value": "secret_value", "created_at": "2023-01-01T00:00:00Z", "updated_at": "2023-01-01T00:00:00Z", "version": "1", "tags": {}, "encrypted": false}'
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        secret = await self.provider.get_secret("test_secret")

        assert secret.metadata.name == "test_secret"
        assert secret.value == "secret_value"
        assert secret.metadata.provider == "local"

    @patch("pathlib.Path.exists")
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, mock_exists):
        """Test secret not found."""
        mock_exists.return_value = False

        with pytest.raises(SecretNotFoundError):
            await self.provider.get_secret("nonexistent_secret")

    @patch("builtins.open")
    @patch("os.chmod")
    @pytest.mark.asyncio
    async def test_set_secret_success(self, mock_chmod, mock_open):
        """Test successful secret storage."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        metadata = await self.provider.set_secret(
            "test_secret",
            "secret_value",
            description="Test secret",
            tags={"env": "test"},
        )

        assert metadata.name == "test_secret"
        assert metadata.provider == "local"
        assert "env" in metadata.tags
        assert metadata.tags["env"] == "test"
        mock_open.assert_called_once()
        mock_chmod.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_secret_success(self):
        """Test successful secret deletion."""
        with patch("pathlib.Path.unlink") as mock_unlink:
            result = await self.provider.delete_secret("test_secret")
            assert result is True
            mock_unlink.assert_called_once()

    @patch("os.environ", {})
    @pytest.mark.asyncio
    async def test_list_secrets(self):
        """Test secret listing."""
        with patch("pathlib.Path.glob") as mock_glob:
            mock_file = Mock()
            mock_file.name = "test_secret.json"
            mock_file.stem = "test_secret"
            mock_file.stat.return_value.st_size = 100
            mock_file.stat.return_value.st_mtime = 1234567890
            mock_glob.return_value = [mock_file]

            with patch("builtins.open") as mock_open:
                mock_file_obj = Mock()
                mock_file_obj.read.return_value = '{"name": "test_secret", "created_at": "2023-01-01T00:00:00Z", "updated_at": "2023-01-01T00:00:00Z", "version": "1", "tags": {}}'
                mock_open.return_value.__enter__.return_value = mock_file_obj
                mock_open.return_value.__exit__.return_value = None

                secrets = await self.provider.list_secrets()

                assert len(secrets) == 1
                assert secrets[0].name == "test_secret"

    @pytest.mark.asyncio
    async def test_rotate_secret(self):
        """Test secret rotation."""
        with patch.object(self.provider, "set_secret") as mock_set_secret:
            mock_metadata = SecretMetadata(
                name="test_secret",
                provider="local",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            mock_set_secret.return_value = mock_metadata

            result = await self.provider.rotate_secret("test_secret")
            assert result == mock_metadata
            mock_set_secret.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_secret_metadata(self):
        """Test secret metadata retrieval."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.read_text", return_value='{"name": "test_secret"}'
            ):
                metadata = await self.provider.get_secret_metadata("test_secret")
                assert metadata.name == "test_secret"
                assert metadata.provider == "local"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test healthy health check."""
        with patch.object(self.provider, "list_secrets", return_value=[]):
            health = await self.provider.health_check()
            assert health["status"] == "healthy"
            assert health["provider"] == "local"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test unhealthy health check."""
        with patch.object(
            self.provider, "list_secrets", side_effect=Exception("Test error")
        ):
            health = await self.provider.health_check()
            assert health["status"] == "unhealthy"
            assert "error" in health

    def test_validate_secret_name(self):
        """Test secret name validation."""
        # Valid name
        self.provider._validate_secret_name("valid_secret_name")

        # Empty name
        with pytest.raises(SecretValidationError):
            self.provider._validate_secret_name("")

        # Too long name
        with pytest.raises(SecretValidationError):
            self.provider._validate_secret_name("a" * 201)

    def test_validate_secret_value(self):
        """Test secret value validation."""
        # Valid value
        self.provider._validate_secret_value("valid_secret_value")

        # Empty value
        with pytest.raises(SecretValidationError):
            self.provider._validate_secret_value("")

        # Too large value
        with pytest.raises(SecretValidationError):
            self.provider._validate_secret_value("a" * 65537)

    @pytest.mark.asyncio
    async def test_get_secret_cached(self):
        """Test cached secret retrieval."""
        with patch.object(self.provider, "get_secret") as mock_get_secret:
            metadata = SecretMetadata(
                name="test_secret",
                provider="local",
                created_at=datetime.now(UTC),
            )
            mock_secret = SecretValue(
                value="secret_value",
                metadata=metadata,
                retrieved_at=datetime.now(UTC),
            )
            mock_get_secret.return_value = mock_secret

            # First call should cache the secret
            result1 = await self.provider.get_secret_cached("test_secret")
            assert result1 == mock_secret

            # Second call should return cached secret
            result2 = await self.provider.get_secret_cached("test_secret")
            assert result2 == mock_secret

            # get_secret should only be called once due to caching
            assert mock_get_secret.call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_cache(self):
        """Test cache invalidation."""
        with patch.object(self.provider, "get_secret") as mock_get_secret:
            metadata = SecretMetadata(
                name="test_secret",
                provider="local",
                created_at=datetime.now(UTC),
            )
            mock_secret = SecretValue(
                value="secret_value",
                metadata=metadata,
                retrieved_at=datetime.now(UTC),
            )
            mock_get_secret.return_value = mock_secret

            # Cache a secret
            await self.provider.get_secret_cached("test_secret")

            # Invalidate cache
            await self.provider.invalidate_cache("test_secret")

            # Next call should fetch from provider again
            await self.provider.get_secret_cached("test_secret")
            assert mock_get_secret.call_count == 2


class TestSecretManagerFactory:
    """Test secret manager factory."""

    def test_create_local_provider(self):
        """Test local provider creation."""
        config = {
            "secrets_dir": "/tmp/test_secrets",
            "encryption_key": "bJ9OXfWDFTXxPr7GoPXRaOHbQdvTskJ30ztcp1M78K4=",
        }
        factory = SecretManagerFactory()

        provider = factory.create_provider(SecretProviderType.LOCAL, config)
        assert isinstance(provider, LocalSecretProvider)

    def test_create_unsupported_provider(self):
        """Test unsupported provider creation."""
        config = {
            "secrets_dir": "/tmp/test_secrets",
        }
        factory = SecretManagerFactory()

        with pytest.raises(ValueError):
            factory.create_provider("unsupported", config)

    @pytest.mark.asyncio
    async def test_get_secret_manager_singleton(self):
        """Test secret manager singleton pattern."""
        # config = {  # Removed as it was unused
        #     "provider_type": "local",
        #     "secrets_dir": "/tmp/test_secrets",
        #     "encryption_key": "bJ9OXfWDFTXxPr7GoPXRaOHbQdvTskJ30ztcp1M78K4=",
        # }

        manager1 = await get_secret_manager()
        manager2 = await get_secret_manager()

        assert manager1 is manager2
