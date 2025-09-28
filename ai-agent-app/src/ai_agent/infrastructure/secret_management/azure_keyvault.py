"""Azure Key Vault integration."""

from typing import Any

from azure.identity import (
    DefaultAzureCredential,
    ClientSecretCredential,
    ManagedIdentityCredential,
)
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import (
    ResourceNotFoundError,
    ClientAuthenticationError,
    HttpResponseError,
)

from .base import (
    SecretProvider,
    SecretProviderType,
    SecretError,
    SecretNotFoundError,
    SecretAccessError,
    SecretMetadata,
    SecretValue,
)


class AzureKeyVaultProvider(SecretProvider):
    """Azure Key Vault secret provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(SecretProviderType.AZURE, config)

        # Azure configuration
        vault_url = config.get("vault_url")
        if not vault_url:
            raise SecretError(
                message="Azure Key Vault URL is required",
                provider=self.provider_type.value,
                error_code="MISSING_CONFIG",
            )
        self.vault_url: str = vault_url

        # Authentication configuration
        self.tenant_id = config.get("tenant_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.use_managed_identity = config.get("use_managed_identity", False)

        # Key Vault configuration
        self.api_version = config.get("api_version", "7.4")
        self.retry_policy = config.get("retry_policy", {})

        # RBAC configuration
        self.enable_rbac = config.get("enable_rbac", True)

        # Initialize client
        self._client: Any = None
        self._credential: Any = None

    async def connect(self) -> None:
        """Initialize Azure Key Vault client."""
        try:
            # Create credential
            if self.use_managed_identity:
                self._credential = ManagedIdentityCredential()
            elif self.tenant_id and self.client_id and self.client_secret:
                self._credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
            else:
                # Use default credential chain
                self._credential = DefaultAzureCredential()

            # Create Key Vault client
            self._client = SecretClient(
                vault_url=self.vault_url,
                credential=self._credential,
                api_version=self.api_version,
            )

            # Test connection
            await self._test_connection()

            self.logger.info(f"Connected to Azure Key Vault: {self.vault_url}")

        except ClientAuthenticationError as e:
            raise SecretAccessError(
                secret_name="test",
                provider=self.provider_type.value,
                reason=f"Authentication failed: {str(e)}",
            )
        except Exception as e:
            raise SecretError(
                message=f"Failed to connect to Azure Key Vault: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_FAILED",
                details={"error": str(e)},
            )

    async def disconnect(self) -> None:
        """Close Azure Key Vault connection."""
        self._client = None
        self._credential = None
        self.logger.info("Disconnected from Azure Key Vault")

    async def _test_connection(self) -> None:
        """Test Azure Key Vault connection."""
        try:
            # Test with a simple operation
            list(self._client.list_properties_of_secrets(max_page_size=1))
        except ClientAuthenticationError:
            raise SecretAccessError(
                secret_name="test",
                provider=self.provider_type.value,
                reason="Insufficient permissions for Key Vault",
            )
        except Exception as e:
            raise SecretError(
                message=f"Azure Key Vault connection test failed: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_TEST_FAILED",
                details={"error": str(e)},
            )

    async def get_secret(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Retrieve a secret from Azure Key Vault."""
        self._validate_secret_name(secret_name)

        try:
            # Get secret
            if version:
                secret = self._client.get_secret(name=secret_name, version=version)
            else:
                secret = self._client.get_secret(name=secret_name)

            # Create metadata
            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=secret.properties.created_on,
                updated_at=secret.properties.updated_on,
                version=secret.properties.version,
                tags=secret.properties.tags or {},
                rotation_enabled=secret.properties.enabled,
                size_bytes=len(secret.value.encode("utf-8")),
            )

            secret_value_obj = SecretValue(value=secret.value, metadata=metadata)

            self._log_secret_access("get", secret_name, True)
            return secret_value_obj

        except ResourceNotFoundError:
            self._log_secret_access("get", secret_name, False, "Secret not found")
            raise SecretNotFoundError(secret_name, self.provider_type.value)
        except ClientAuthenticationError as e:
            self._log_secret_access("get", secret_name, False, "Access denied")
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
        except HttpResponseError as e:
            self._log_secret_access("get", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to retrieve secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                error_code="HTTP_ERROR",
                details={"error": str(e), "status_code": e.status_code},
            )

    async def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Store a secret in Azure Key Vault."""
        self._validate_secret_name(secret_name)
        self._validate_secret_value(secret_value)

        try:
            # Set secret
            secret = self._client.set_secret(
                name=secret_name,
                value=secret_value,
                tags=tags or {},
                content_type="text/plain",
            )

            # Update description if provided
            if description:
                # Azure Key Vault doesn't have a direct description field
                # We can store it in tags
                if not tags:
                    tags = {}
                tags["description"] = description

                # Update secret with tags
                secret = self._client.update_secret_properties(
                    name=secret_name, version=secret.properties.version, tags=tags
                )

            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=secret.properties.created_on,
                updated_at=secret.properties.updated_on,
                version=secret.properties.version,
                tags=secret.properties.tags or {},
            )

            self._log_secret_access("set", secret_name, True)
            return metadata

        except ClientAuthenticationError as e:
            self._log_secret_access("set", secret_name, False, "Access denied")
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
        except HttpResponseError as e:
            self._log_secret_access("set", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to store secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                error_code="HTTP_ERROR",
                details={"error": str(e), "status_code": e.status_code},
            )

    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret from Azure Key Vault."""
        self._validate_secret_name(secret_name)

        try:
            if version:
                # Azure Key Vault doesn't support version-specific deletion
                # We can only disable the specific version
                self._client.update_secret_properties(
                    name=secret_name, version=version, enabled=False
                )
                self._log_secret_access("disable_version", secret_name, True)
            else:
                # Delete the secret (soft delete)
                self._client.begin_delete_secret(name=secret_name)
                self._log_secret_access("delete", secret_name, True)

            return True

        except ResourceNotFoundError:
            self._log_secret_access("delete", secret_name, False, "Secret not found")
            return False
        except ClientAuthenticationError as e:
            self._log_secret_access("delete", secret_name, False, "Access denied")
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
        except HttpResponseError as e:
            self._log_secret_access("delete", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to delete secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                error_code="HTTP_ERROR",
                details={"error": str(e), "status_code": e.status_code},
            )

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List secrets in Azure Key Vault."""
        try:
            secrets = []

            for secret_properties in self._client.list_properties_of_secrets():
                secret_name = secret_properties.name

                # Filter by prefix if specified
                if prefix and not secret_name.startswith(prefix):
                    continue

                metadata = self._create_secret_metadata(
                    secret_name=secret_name,
                    created_at=secret_properties.created_on,
                    updated_at=secret_properties.updated_on,
                    version=secret_properties.version,
                    tags=secret_properties.tags or {},
                    rotation_enabled=secret_properties.enabled,
                )

                secrets.append(metadata)

            self.logger.info(f"Listed {len(secrets)} secrets")
            return secrets

        except ClientAuthenticationError as e:
            raise SecretAccessError(
                secret_name="list", provider=self.provider_type.value, reason=str(e)
            )
        except HttpResponseError as e:
            raise SecretError(
                message=f"Failed to list secrets: {str(e)}",
                provider=self.provider_type.value,
                error_code="HTTP_ERROR",
                details={"error": str(e), "status_code": e.status_code},
            )

    async def rotate_secret(self, secret_name: str) -> SecretMetadata:
        """Rotate a secret in Azure Key Vault."""
        self._validate_secret_name(secret_name)

        try:
            # Get current secret
            current_secret = await self.get_secret(secret_name)

            # Generate new secret value (this would typically call a rotation function)
            new_value = await self._generate_new_secret_value(
                secret_name, current_secret.value
            )

            # Store new version
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
            secret_properties = self._client.get_secret_properties(name=secret_name)

            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=secret_properties.created_on,
                updated_at=secret_properties.updated_on,
                version=secret_properties.version,
                tags=secret_properties.tags or {},
                rotation_enabled=secret_properties.enabled,
            )

            return metadata

        except ResourceNotFoundError:
            raise SecretNotFoundError(secret_name, self.provider_type.value)
        except ClientAuthenticationError as e:
            raise SecretAccessError(secret_name, self.provider_type.value, str(e))
        except HttpResponseError as e:
            raise SecretError(
                message=f"Failed to get secret metadata: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                error_code="HTTP_ERROR",
                details={"error": str(e), "status_code": e.status_code},
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

    async def enable_secret(self, secret_name: str) -> bool:
        """Enable a secret in Azure Key Vault."""
        try:
            self._client.update_secret_properties(name=secret_name, enabled=True)
            self._log_secret_access("enable", secret_name, True)
            return True
        except Exception as e:
            self._log_secret_access("enable", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to enable secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def disable_secret(self, secret_name: str) -> bool:
        """Disable a secret in Azure Key Vault."""
        try:
            self._client.update_secret_properties(name=secret_name, enabled=False)
            self._log_secret_access("disable", secret_name, True)
            return True
        except Exception as e:
            self._log_secret_access("disable", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to disable secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def purge_secret(self, secret_name: str) -> bool:
        """Permanently delete a secret from Azure Key Vault."""
        try:
            self._client.purge_deleted_secret(name=secret_name)
            self._log_secret_access("purge", secret_name, True)
            return True
        except Exception as e:
            self._log_secret_access("purge", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to purge secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )

    async def recover_secret(self, secret_name: str) -> bool:
        """Recover a soft-deleted secret in Azure Key Vault."""
        try:
            self._client.begin_recover_deleted_secret(name=secret_name)
            self._log_secret_access("recover", secret_name, True)
            return True
        except Exception as e:
            self._log_secret_access("recover", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to recover secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                details={"error": str(e)},
            )
