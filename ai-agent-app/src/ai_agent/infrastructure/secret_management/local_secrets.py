"""Local development secret management."""

import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from dotenv import load_dotenv

from .base import (
    SecretProvider,
    SecretProviderType,
    SecretError,
    SecretNotFoundError,
    SecretMetadata,
    SecretValue,
)


class LocalSecretProvider(SecretProvider):
    """Local development secret provider using environment variables and files."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(SecretProviderType.LOCAL, config)

        # Local configuration
        self.env_file = config.get("env_file", ".env")
        self.secrets_dir = Path(config.get("secrets_dir", "secrets"))
        self.encryption_key = config.get("encryption_key")
        self.auto_load_env = config.get("auto_load_env", True)

        # Create secrets directory if it doesn't exist
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        self._cipher = None
        if self.encryption_key:
            self._cipher = Fernet(self.encryption_key.encode())

        # Load environment variables if enabled
        if self.auto_load_env and os.path.exists(self.env_file):
            load_dotenv(self.env_file)

    async def connect(self) -> None:
        """Initialize local secret provider."""
        try:
            # Test directory access
            test_file = self.secrets_dir / ".test"
            test_file.write_text("test")
            test_file.unlink()

            self.logger.info(f"Connected to local secret provider: {self.secrets_dir}")

        except Exception as e:
            raise SecretError(
                message=f"Failed to connect to local secret provider: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_FAILED",
                details={"error": str(e)},
            )

    async def disconnect(self) -> None:
        """Close local secret provider."""
        self._cipher = None
        self.logger.info("Disconnected from local secret provider")

    async def get_secret(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Retrieve a secret from local storage."""
        self._validate_secret_name(secret_name)

        try:
            # Try environment variable first
            env_value = os.getenv(secret_name)
            if env_value:
                metadata = self._create_secret_metadata(secret_name=secret_name)

                secret_value_obj = SecretValue(value=env_value, metadata=metadata)

                self._log_secret_access("get", secret_name, True)
                return secret_value_obj

            # Try file-based secret
            secret_file = self.secrets_dir / f"{secret_name}.json"
            if secret_file.exists():
                with open(secret_file) as f:
                    secret_data = json.load(f)

                # Decrypt if encrypted
                secret_value = secret_data.get("value", "")
                if secret_data.get("encrypted", False) and self._cipher:
                    secret_value = self._cipher.decrypt(secret_value.encode()).decode()

                metadata = self._create_secret_metadata(
                    secret_name=secret_name,
                    created_at=datetime.fromisoformat(
                        secret_data.get("created_at", "")
                    ),
                    updated_at=datetime.fromisoformat(
                        secret_data.get("updated_at", "")
                    ),
                    version=secret_data.get("version"),
                    tags=secret_data.get("tags", {}),
                )

                secret_value_obj = SecretValue(value=secret_value, metadata=metadata)

                self._log_secret_access("get", secret_name, True)
                return secret_value_obj

            # Secret not found
            self._log_secret_access("get", secret_name, False, "Secret not found")
            raise SecretNotFoundError(secret_name, self.provider_type.value)

        except SecretNotFoundError:
            raise
        except Exception as e:
            self._log_secret_access("get", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to retrieve secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Store a secret in local storage."""
        self._validate_secret_name(secret_name)
        self._validate_secret_value(secret_value)

        try:
            # Prepare secret data
            secret_data = {
                "value": secret_value,
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "version": "1",
                "tags": tags or {},
                "description": description or "",
                "encrypted": False,
            }

            # Encrypt if cipher is available
            if self._cipher:
                secret_data["value"] = self._cipher.encrypt(
                    secret_value.encode()
                ).decode()
                secret_data["encrypted"] = True

            # Store in file
            secret_file = self.secrets_dir / f"{secret_name}.json"
            with open(secret_file, "w") as f:
                json.dump(secret_data, f, indent=2)

            # Set file permissions (readable only by owner)
            os.chmod(secret_file, 0o600)

            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=datetime.now(UTC),
                version="1",
                tags=tags or {},
            )

            self._log_secret_access("set", secret_name, True)
            return metadata

        except Exception as e:
            self._log_secret_access("set", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to store secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret from local storage."""
        self._validate_secret_name(secret_name)

        try:
            # Remove from environment (if it was set there)
            if secret_name in os.environ:
                del os.environ[secret_name]

            # Remove file
            secret_file = self.secrets_dir / f"{secret_name}.json"
            if secret_file.exists():
                secret_file.unlink()
                self._log_secret_access("delete", secret_name, True)
                return True
            else:
                self._log_secret_access(
                    "delete", secret_name, False, "Secret not found"
                )
                return False

        except Exception as e:
            self._log_secret_access("delete", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to delete secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List secrets in local storage."""
        try:
            secrets = []

            # List environment variables
            for key, _value in os.environ.items():
                if key.startswith(prefix or ""):
                    metadata = self._create_secret_metadata(secret_name=key)
                    secrets.append(metadata)

            # List file-based secrets
            for secret_file in self.secrets_dir.glob("*.json"):
                secret_name = secret_file.stem

                if prefix and not secret_name.startswith(prefix):
                    continue

                try:
                    with open(secret_file) as f:
                        secret_data = json.load(f)

                    metadata = self._create_secret_metadata(
                        secret_name=secret_name,
                        created_at=datetime.fromisoformat(
                            secret_data.get("created_at", "")
                        ),
                        updated_at=datetime.fromisoformat(
                            secret_data.get("updated_at", "")
                        ),
                        version=secret_data.get("version"),
                        tags=secret_data.get("tags", {}),
                    )
                    secrets.append(metadata)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read secret file {secret_file}: {e}"
                    )
                    continue

            self.logger.info(f"Listed {len(secrets)} secrets")
            return secrets

        except Exception as e:
            raise SecretError(
                message=f"Failed to list secrets: {str(e)}",
                provider=self.provider_type.value,
                error_code="LIST_FAILED",
                details={"error": str(e)},
            )

    async def rotate_secret(self, secret_name: str) -> SecretMetadata:
        """Rotate a secret in local storage."""
        self._validate_secret_name(secret_name)

        try:
            # Get current secret
            current_secret = await self.get_secret(secret_name)

            # Generate new secret value
            new_value = await self._generate_new_secret_value(
                secret_name, current_secret.value
            )

            # Update secret
            metadata = await self.set_secret(secret_name, new_value)

            self._log_secret_access("rotate", secret_name, True)
            return metadata

        except Exception as e:
            self._log_secret_access("rotate", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to rotate secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def get_secret_metadata(self, secret_name: str) -> SecretMetadata:
        """Get secret metadata without retrieving the value."""
        self._validate_secret_name(secret_name)

        try:
            # Check environment variable
            if secret_name in os.environ:
                return self._create_secret_metadata(secret_name=secret_name)

            # Check file-based secret
            secret_file = self.secrets_dir / f"{secret_name}.json"
            if secret_file.exists():
                with open(secret_file) as f:
                    secret_data = json.load(f)

                return self._create_secret_metadata(
                    secret_name=secret_name,
                    created_at=datetime.fromisoformat(
                        secret_data.get("created_at", "")
                    ),
                    updated_at=datetime.fromisoformat(
                        secret_data.get("updated_at", "")
                    ),
                    version=secret_data.get("version"),
                    tags=secret_data.get("tags", {}),
                )

            raise SecretNotFoundError(secret_name, self.provider_type.value)

        except SecretNotFoundError:
            raise
        except Exception as e:
            raise SecretError(
                message=f"Failed to get secret metadata: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def _generate_new_secret_value(
        self, secret_name: str, current_value: str
    ) -> str:
        """Generate a new secret value for rotation."""
        import secrets
        import string

        # Generate a random string of similar length
        length = len(current_value)
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def _create_secret_metadata(
        self,
        secret_name: str,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: str | None = None,
        tags: dict[str, str] | None = None,
        rotation_enabled: bool = False,
        next_rotation: datetime | None = None,
        size_bytes: int | None = None,
    ) -> SecretMetadata:
        """Create secret metadata object with local-specific fields."""
        metadata = super()._create_secret_metadata(
            secret_name=secret_name,
            created_at=created_at,
            updated_at=updated_at,
            version=version,
            tags=tags or {},
            rotation_enabled=rotation_enabled,
            next_rotation=next_rotation,
            size_bytes=size_bytes,
        )

        # Add local-specific metadata
        metadata.tags["source"] = "local"
        metadata.tags["provider"] = "local"

        return metadata

    async def export_secrets(self, output_file: str) -> bool:
        """Export all secrets to a file."""
        try:
            secrets_data = {}

            for secret_file in self.secrets_dir.glob("*.json"):
                secret_name = secret_file.stem
                with open(secret_file) as f:
                    secret_data = json.load(f)

                # Decrypt if encrypted
                if secret_data.get("encrypted", False) and self._cipher:
                    secret_data["value"] = self._cipher.decrypt(
                        secret_data["value"].encode()
                    ).decode()
                    secret_data["encrypted"] = False

                secrets_data[secret_name] = secret_data

            with open(output_file, "w") as f:
                json.dump(secrets_data, f, indent=2)

            os.chmod(output_file, 0o600)

            self.logger.info(f"Exported {len(secrets_data)} secrets to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export secrets: {e}")
            return False

    async def import_secrets(self, input_file: str) -> bool:
        """Import secrets from a file."""
        try:
            with open(input_file) as f:
                secrets_data = json.load(f)

            imported_count = 0
            for secret_name, secret_data in secrets_data.items():
                try:
                    # Encrypt if cipher is available
                    if self._cipher and not secret_data.get("encrypted", False):
                        secret_data["value"] = self._cipher.encrypt(
                            secret_data["value"].encode()
                        ).decode()
                        secret_data["encrypted"] = True

                    # Save secret
                    secret_file = self.secrets_dir / f"{secret_name}.json"
                    with open(secret_file, "w") as f:
                        json.dump(secret_data, f, indent=2)

                    os.chmod(secret_file, 0o600)
                    imported_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to import secret {secret_name}: {e}")
                    continue

            self.logger.info(f"Imported {imported_count} secrets from {input_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import secrets: {e}")
            return False

    async def generate_encryption_key(self) -> str:
        """Generate a new encryption key for local secrets."""
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        return key.decode()

    async def migrate_to_cloud(self, cloud_provider: SecretProvider) -> int:
        """Migrate local secrets to a cloud provider."""
        try:
            secrets = await self.list_secrets()
            migrated_count = 0

            for secret_metadata in secrets:
                try:
                    # Get secret value
                    secret_value = await self.get_secret(secret_metadata.name)

                    # Store in cloud provider
                    await cloud_provider.set_secret(
                        secret_name=secret_metadata.name,
                        secret_value=secret_value.value,
                        description=secret_metadata.tags.get("description", ""),
                        tags=secret_metadata.tags,
                    )

                    migrated_count += 1
                    self.logger.info(f"Migrated secret: {secret_metadata.name}")

                except Exception as e:
                    self.logger.warning(
                        f"Failed to migrate secret {secret_metadata.name}: {e}"
                    )
                    continue

            self.logger.info(
                f"Migrated {migrated_count} secrets to {cloud_provider.provider_type.value}"
            )
            return migrated_count

        except Exception as e:
            self.logger.error(f"Failed to migrate secrets: {e}")
            return 0
