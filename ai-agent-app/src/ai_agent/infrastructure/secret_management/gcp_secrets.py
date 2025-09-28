"""Google Cloud Secret Manager integration."""

import json
from datetime import datetime, UTC
from typing import Any

from google.cloud import secretmanager
from google.api_core import exceptions as gcp_exceptions
from google.oauth2 import service_account

from .base import (
    SecretProvider,
    SecretProviderType,
    SecretError,
    SecretNotFoundError,
    SecretAccessError,
    SecretMetadata,
    SecretValue,
)


class GCPSecretProvider(SecretProvider):
    """Google Cloud Secret Manager secret provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(SecretProviderType.GCP, config)

        # GCP configuration
        self.project_id = config.get("project_id")
        if not self.project_id:
            raise SecretError(
                message="GCP project ID is required",
                provider=self.provider_type.value,
                error_code="MISSING_CONFIG",
            )

        # Authentication configuration
        self.credentials_path = config.get("credentials_path")
        self.credentials_json = config.get("credentials_json")
        self.use_application_default_credentials = config.get(
            "use_application_default_credentials", True
        )

        # Secret Manager configuration
        self.location = config.get("location", "global")
        self.replication_policy = config.get("replication_policy", "automatic")

        # IAM configuration
        self.enable_iam = config.get("enable_iam", True)

        # Initialize client
        self._client: Any = None
        self._credentials: Any = None

    async def connect(self) -> None:
        """Initialize Google Cloud Secret Manager client."""
        try:
            # Create credentials
            if self.credentials_json:
                # Use JSON credentials
                credentials_info = json.loads(self.credentials_json)
                self._credentials = (
                    service_account.Credentials.from_service_account_info(
                        credentials_info
                    )
                )
            elif self.credentials_path:
                # Use credentials file
                self._credentials = (
                    service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                )
            elif self.use_application_default_credentials:
                # Use application default credentials
                self._credentials = None  # Will use default
            else:
                raise SecretError(
                    message="No authentication method specified",
                    provider=self.provider_type.value,
                    error_code="NO_AUTH_METHOD",
                )

            # Create Secret Manager client
            if self._credentials:
                self._client = secretmanager.SecretManagerServiceClient(
                    credentials=self._credentials
                )
            else:
                self._client = secretmanager.SecretManagerServiceClient()

            # Test connection
            await self._test_connection()

            self.logger.info(
                f"Connected to Google Cloud Secret Manager in project: {self.project_id}"
            )

        except Exception as e:
            raise SecretError(
                message=f"Failed to connect to Google Cloud Secret Manager: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_FAILED",
                details={"error": str(e)},
            )

    async def disconnect(self) -> None:
        """Close Google Cloud Secret Manager connection."""
        self._client = None
        self._credentials = None
        self.logger.info("Disconnected from Google Cloud Secret Manager")

    async def _test_connection(self) -> None:
        """Test Google Cloud Secret Manager connection."""
        try:
            # Test with a simple operation
            parent = f"projects/{self.project_id}"
            list(self._client.list_secrets(request={"parent": parent}))
        except gcp_exceptions.PermissionDenied:
            raise SecretAccessError(
                secret_name="test",
                provider=self.provider_type.value,
                reason="Insufficient permissions for Secret Manager",
            )
        except Exception as e:
            raise SecretError(
                message=f"Google Cloud Secret Manager connection test failed: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_TEST_FAILED",
                details={"error": str(e)},
            )

    async def get_secret(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Retrieve a secret from Google Cloud Secret Manager."""
        self._validate_secret_name(secret_name)

        try:
            # Build secret path
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"

            if version:
                version_path = f"{secret_path}/versions/{version}"
            else:
                # Get latest version
                version_path = f"{secret_path}/versions/latest"

            # Get secret
            response = self._client.access_secret_version(
                request={"name": version_path}
            )
            secret_value = response.payload.data.decode("UTF-8")

            # Get secret metadata
            secret_metadata = self._client.get_secret(request={"name": secret_path})

            # Create metadata
            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=secret_metadata.create_time,
                updated_at=secret_metadata.create_time,  # GCP doesn't track update time
                version=version or "latest",
                tags=dict(secret_metadata.labels) if secret_metadata.labels else {},
                rotation_enabled=bool(secret_metadata.rotation),
                next_rotation=(
                    secret_metadata.rotation.next_rotation_time
                    if secret_metadata.rotation
                    else None
                ),
                size_bytes=len(secret_value.encode("utf-8")),
            )

            secret_value_obj = SecretValue(value=secret_value, metadata=metadata)

            self._log_secret_access("get", secret_name, True)
            return secret_value_obj

        except gcp_exceptions.NotFound:
            self._log_secret_access("get", secret_name, False, "Secret not found")
            raise SecretNotFoundError(secret_name, self.provider_type.value)
        except gcp_exceptions.PermissionDenied as e:
            self._log_secret_access("get", secret_name, False, "Access denied")
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
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
        """Store a secret in Google Cloud Secret Manager."""
        self._validate_secret_name(secret_name)
        self._validate_secret_value(secret_value)

        try:
            # Build secret path

            # Check if secret exists
            try:
                await self.get_secret_metadata(secret_name)
                # Secret exists, add new version
                return await self._add_secret_version(secret_name, secret_value, tags)
            except SecretNotFoundError:
                # Secret doesn't exist, create it
                return await self._create_secret(
                    secret_name, secret_value, description, tags
                )

        except Exception as e:
            self._log_secret_access("set", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to store secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def _create_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Create a new secret."""
        parent = f"projects/{self.project_id}"

        # Create secret
        secret: dict[str, Any] = {
            "replication": {
                "automatic": (
                    {}
                    if self.replication_policy == "automatic"
                    else {"user_managed": {"replicas": [{"location": self.location}]}}
                )
            }
        }

        if description:
            secret["labels"] = {"description": description}

        if tags:
            if "labels" not in secret:
                secret["labels"] = {}
            secret["labels"].update(tags)

        response = self._client.create_secret(
            request={"parent": parent, "secret_id": secret_name, "secret": secret}
        )

        # Add secret version
        version_response = self._client.add_secret_version(
            request={
                "parent": response.name,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )

        metadata = self._create_secret_metadata(
            secret_name=secret_name,
            created_at=response.create_time,
            version=version_response.name.split("/")[-1],
            tags=dict(response.labels) if response.labels else {},
        )

        self._log_secret_access("create", secret_name, True)
        return metadata

    async def _add_secret_version(
        self, secret_name: str, secret_value: str, tags: dict[str, str] | None = None
    ) -> SecretMetadata:
        """Add a new version to an existing secret."""
        secret_path = f"projects/{self.project_id}/secrets/{secret_name}"

        # Add new version
        version_response = self._client.add_secret_version(
            request={
                "parent": secret_path,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )

        # Update labels if provided
        if tags:
            secret = self._client.get_secret(request={"name": secret_path})
            if secret.labels:
                secret.labels.update(tags)
            else:
                secret.labels = tags

            self._client.update_secret(request={"secret": secret})

        metadata = self._create_secret_metadata(
            secret_name=secret_name,
            created_at=version_response.create_time,
            version=version_response.name.split("/")[-1],
            tags=tags or {},
        )

        self._log_secret_access("add_version", secret_name, True)
        return metadata

    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret from Google Cloud Secret Manager."""
        self._validate_secret_name(secret_name)

        try:
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"

            if version:
                # Delete specific version
                version_path = f"{secret_path}/versions/{version}"
                self._client.destroy_secret_version(request={"name": version_path})
                self._log_secret_access("delete_version", secret_name, True)
            else:
                # Delete the entire secret
                self._client.delete_secret(request={"name": secret_path})
                self._log_secret_access("delete", secret_name, True)

            return True

        except gcp_exceptions.NotFound:
            self._log_secret_access("delete", secret_name, False, "Secret not found")
            return False
        except gcp_exceptions.PermissionDenied as e:
            self._log_secret_access("delete", secret_name, False, "Access denied")
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
        except Exception as e:
            self._log_secret_access("delete", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to delete secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List secrets in Google Cloud Secret Manager."""
        try:
            parent = f"projects/{self.project_id}"
            secrets = []

            for secret in self._client.list_secrets(request={"parent": parent}):
                secret_name = secret.name.split("/")[-1]

                # Filter by prefix if specified
                if prefix and not secret_name.startswith(prefix):
                    continue

                metadata = self._create_secret_metadata(
                    secret_name=secret_name,
                    created_at=secret.create_time,
                    updated_at=secret.create_time,  # GCP doesn't track update time
                    tags=dict(secret.labels) if secret.labels else {},
                    rotation_enabled=bool(secret.rotation),
                    next_rotation=(
                        secret.rotation.next_rotation_time if secret.rotation else None
                    ),
                )

                secrets.append(metadata)

            self.logger.info(f"Listed {len(secrets)} secrets")
            return secrets

        except gcp_exceptions.PermissionDenied as e:
            raise SecretAccessError(
                secret_name="list", provider=self.provider_type.value, reason=str(e)
            )
        except Exception as e:
            raise SecretError(
                message=f"Failed to list secrets: {str(e)}",
                provider=self.provider_type.value,
                error_code="LIST_FAILED",
                details={"error": str(e)},
            )

    async def rotate_secret(self, secret_name: str) -> SecretMetadata:
        """Rotate a secret in Google Cloud Secret Manager."""
        self._validate_secret_name(secret_name)

        try:
            # Get current secret
            current_secret = await self.get_secret(secret_name)

            # Generate new secret value
            new_value = await self._generate_new_secret_value(
                secret_name, current_secret.value
            )

            # Add new version
            metadata = await self._add_secret_version(secret_name, new_value)

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
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"
            secret = self._client.get_secret(request={"name": secret_path})

            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=secret.create_time,
                updated_at=secret.create_time,  # GCP doesn't track update time
                tags=dict(secret.labels) if secret.labels else {},
                rotation_enabled=bool(secret.rotation),
                next_rotation=(
                    secret.rotation.next_rotation_time if secret.rotation else None
                ),
            )

            return metadata

        except gcp_exceptions.NotFound:
            raise SecretNotFoundError(secret_name, self.provider_type.value)
        except gcp_exceptions.PermissionDenied as e:
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
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
        # This is a placeholder implementation
        # In a real scenario, you would implement proper secret generation logic
        import secrets
        import string

        # Generate a random string of similar length
        length = len(current_value)
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return "".join(secrets.choice(alphabet) for _ in range(length))

    async def enable_automatic_rotation(
        self, secret_name: str, rotation_period: str = "2592000s"  # 30 days
    ) -> bool:
        """Enable automatic rotation for a secret."""
        try:
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"

            # Get current secret
            secret = self._client.get_secret(request={"name": secret_path})

            # Update rotation settings
            secret.rotation = {
                "rotation_period": rotation_period,
                "next_rotation_time": {
                    "seconds": int(datetime.now(UTC).timestamp())
                    + int(rotation_period.rstrip("s"))
                },
            }

            self._client.update_secret(request={"secret": secret})

            self._log_secret_access("enable_rotation", secret_name, True)
            return True

        except Exception as e:
            self._log_secret_access("enable_rotation", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to enable rotation: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def disable_automatic_rotation(self, secret_name: str) -> bool:
        """Disable automatic rotation for a secret."""
        try:
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"

            # Get current secret
            secret = self._client.get_secret(request={"name": secret_path})

            # Remove rotation settings
            secret.rotation = None

            self._client.update_secret(request={"secret": secret})

            self._log_secret_access("disable_rotation", secret_name, True)
            return True

        except Exception as e:
            self._log_secret_access("disable_rotation", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to disable rotation: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def list_secret_versions(self, secret_name: str) -> list[dict[str, Any]]:
        """List all versions of a secret."""
        try:
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}"
            versions = []

            for version in self._client.list_secret_versions(
                request={"parent": secret_path}
            ):
                versions.append(
                    {
                        "name": version.name,
                        "version": version.name.split("/")[-1],
                        "state": version.state.name,
                        "create_time": version.create_time,
                        "destroy_time": version.destroy_time,
                    }
                )

            return versions

        except Exception as e:
            raise SecretError(
                message=f"Failed to list secret versions: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )
