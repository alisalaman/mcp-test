"""AWS Secrets Manager integration."""

from datetime import datetime, UTC
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from .base import (
    SecretProvider,
    SecretProviderType,
    SecretError,
    SecretNotFoundError,
    SecretAccessError,
    SecretMetadata,
    SecretValue,
)


class AWSSecretProvider(SecretProvider):
    """AWS Secrets Manager secret provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(SecretProviderType.AWS, config)

        # AWS configuration
        self.region = config.get("region", "us-east-1")
        self.access_key_id = config.get("access_key_id")
        self.secret_access_key = config.get("secret_access_key")
        self.session_token = config.get("session_token")
        self.role_arn = config.get("role_arn")
        self.external_id = config.get("external_id")

        # KMS configuration
        self.kms_key_id = config.get("kms_key_id")
        self.encryption_context = config.get("encryption_context", {})

        # Rotation configuration
        self.rotation_lambda_arn = config.get("rotation_lambda_arn")
        self.rotation_days = config.get("rotation_days", 30)

        # Boto3 configuration
        Config(
            retries={"max_attempts": config.get("max_retries", 3), "mode": "adaptive"},
            region_name=self.region,
        )

        # Initialize clients
        self._secrets_client: Any = None
        self._kms_client: Any = None
        self._sts_client: Any = None

    async def connect(self) -> None:
        """Initialize AWS clients."""
        try:
            # Create session
            session_kwargs = {
                "region_name": self.region,
                "config": Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    region_name=self.region,
                ),
            }

            if self.access_key_id and self.secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": self.access_key_id,
                        "aws_secret_access_key": self.secret_access_key,
                    }
                )
                if self.session_token:
                    session_kwargs["aws_session_token"] = self.session_token

            session = boto3.Session(**session_kwargs)

            # Assume role if specified
            if self.role_arn:
                sts_client = session.client("sts")
                assume_role_kwargs = {
                    "RoleArn": self.role_arn,
                    "RoleSessionName": "ai-agent-secret-manager",
                }
                if self.external_id:
                    assume_role_kwargs["ExternalId"] = self.external_id

                response = sts_client.assume_role(**assume_role_kwargs)
                credentials = response["Credentials"]

                session = boto3.Session(
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"],
                    region_name=self.region,
                )

            # Initialize clients
            self._secrets_client = session.client("secretsmanager")
            self._kms_client = session.client("kms")
            self._sts_client = session.client("sts")

            # Test connection
            await self._test_connection()

            self.logger.info(
                f"Connected to AWS Secrets Manager in region: {self.region}"
            )

        except NoCredentialsError as e:
            raise SecretError(
                message="AWS credentials not found",
                provider=self.provider_type.value,
                error_code="NO_CREDENTIALS",
                details={"error": str(e)},
            )
        except Exception as e:
            raise SecretError(
                message=f"Failed to connect to AWS Secrets Manager: {str(e)}",
                provider=self.provider_type.value,
                error_code="CONNECTION_FAILED",
                details={"error": str(e)},
            )

    async def disconnect(self) -> None:
        """Close AWS connections."""
        self._secrets_client = None
        self._kms_client = None
        self._sts_client = None
        self.logger.info("Disconnected from AWS Secrets Manager")

    async def _test_connection(self) -> None:
        """Test AWS connection."""
        try:
            # Test with a simple API call
            self._secrets_client.list_secrets(MaxResults=1)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "AccessDenied":
                raise SecretAccessError(
                    secret_name="test",
                    provider=self.provider_type.value,
                    reason="Insufficient permissions for Secrets Manager",
                )
            else:
                raise SecretError(
                    message=f"AWS connection test failed: {str(e)}",
                    provider=self.provider_type.value,
                    error_code=error_code,
                    details={"error": str(e)},
                )

    async def get_secret(
        self, secret_name: str, version: str | None = None
    ) -> SecretValue:
        """Retrieve a secret from AWS Secrets Manager."""
        self._validate_secret_name(secret_name)

        try:
            # Prepare request parameters
            request_params = {"SecretId": secret_name}
            if version:
                request_params["VersionId"] = version

            # Get secret value
            response = self._secrets_client.get_secret_value(**request_params)

            # Extract secret value
            secret_value = response.get("SecretString")
            if not secret_value:
                # Try binary secret
                secret_binary = response.get("SecretBinary")
                if secret_binary:
                    secret_value = secret_binary.decode("utf-8")
                else:
                    raise SecretNotFoundError(secret_name, self.provider_type.value)

            # Create metadata
            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=response.get("CreatedDate"),
                version=response.get("VersionId"),
                tags=self._extract_tags_from_arn(response.get("ARN", "")),
                rotation_enabled=response.get("RotationEnabled", False),
                next_rotation=response.get("NextRotationDate"),
                size_bytes=len(secret_value.encode("utf-8")),
            )

            secret_value_obj = SecretValue(value=secret_value, metadata=metadata)

            self._log_secret_access("get", secret_name, True)
            return secret_value_obj

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                self._log_secret_access("get", secret_name, False, "Secret not found")
                raise SecretNotFoundError(secret_name, self.provider_type.value)
            elif error_code == "AccessDeniedException":
                self._log_secret_access("get", secret_name, False, "Access denied")
                raise SecretAccessError(
                    secret_name, self.provider_type.value, "Access denied"
                )
            else:
                self._log_secret_access("get", secret_name, False, str(e))
                raise SecretError(
                    message=f"Failed to retrieve secret: {str(e)}",
                    provider=self.provider_type.value,
                    secret_name=secret_name,
                    error_code=error_code,
                    details={"error": str(e)},
                )

    async def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Store a secret in AWS Secrets Manager."""
        self._validate_secret_name(secret_name)
        self._validate_secret_value(secret_value)

        try:
            # Check if secret exists
            try:
                await self.get_secret_metadata(secret_name)
                # Secret exists, update it
                return await self._update_secret(
                    secret_name, secret_value, description, tags
                )
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
        request_params: dict[str, Any] = {
            "Name": secret_name,
            "SecretString": secret_value,
            "Description": description
            or f"Secret created by AI Agent at {datetime.now(UTC).isoformat()}",
        }

        # Add KMS key if specified
        if self.kms_key_id:
            request_params["KmsKeyId"] = self.kms_key_id

        # Add tags
        if tags:
            request_params["Tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]

        response = self._secrets_client.create_secret(**request_params)

        metadata = self._create_secret_metadata(
            secret_name=secret_name,
            created_at=datetime.now(UTC),
            version=response.get("VersionId"),
            tags=tags or {},
        )

        self._log_secret_access("create", secret_name, True)
        return metadata

    async def _update_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> SecretMetadata:
        """Update an existing secret."""
        # Update secret value
        request_params = {"SecretId": secret_name, "SecretString": secret_value}

        response = self._secrets_client.update_secret(**request_params)

        # Update description if provided
        if description:
            self._secrets_client.update_secret_description(
                SecretId=secret_name, Description=description
            )

        # Update tags if provided
        if tags:
            # Remove existing tags
            self._secrets_client.untag_resource(
                SecretId=secret_name, TagKeys=list(tags.keys())
            )
            # Add new tags
            self._secrets_client.tag_resource(
                SecretId=secret_name,
                Tags=[{"Key": k, "Value": v} for k, v in tags.items()],
            )

        metadata = self._create_secret_metadata(
            secret_name=secret_name,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            version=response.get("VersionId"),
            tags=tags or {},
        )

        self._log_secret_access("update", secret_name, True)
        return metadata

    async def delete_secret(self, secret_name: str, version: str | None = None) -> bool:
        """Delete a secret from AWS Secrets Manager."""
        self._validate_secret_name(secret_name)

        try:
            if version:
                # Delete specific version (not supported by AWS Secrets Manager)
                raise SecretError(
                    message="Version-specific deletion not supported by AWS Secrets Manager",
                    provider=self.provider_type.value,
                    secret_name=secret_name,
                    error_code="UNSUPPORTED_OPERATION",
                )

            # Schedule deletion (AWS doesn't allow immediate deletion)
            self._secrets_client.delete_secret(
                SecretId=secret_name, ForceDeleteWithoutRecovery=True
            )

            self._log_secret_access("delete", secret_name, True)
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                self._log_secret_access(
                    "delete", secret_name, False, "Secret not found"
                )
                return False
            else:
                self._log_secret_access("delete", secret_name, False, str(e))
                raise SecretError(
                    message=f"Failed to delete secret: {str(e)}",
                    provider=self.provider_type.value,
                    secret_name=secret_name,
                    error_code=error_code,
                    details={"error": str(e)},
                )

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List secrets in AWS Secrets Manager."""
        try:
            secrets = []
            paginator = self._secrets_client.get_paginator("list_secrets")

            for page in paginator.paginate():
                for secret in page["SecretList"]:
                    secret_name = secret["Name"]

                    # Filter by prefix if specified
                    if prefix and not secret_name.startswith(prefix):
                        continue

                    metadata = self._create_secret_metadata(
                        secret_name=secret_name,
                        created_at=secret.get("CreatedDate"),
                        updated_at=secret.get("LastChangedDate"),
                        version=secret.get("VersionId"),
                        tags=self._extract_tags_from_arn(secret.get("ARN", "")),
                        rotation_enabled=secret.get("RotationEnabled", False),
                        next_rotation=secret.get("NextRotationDate"),
                        size_bytes=secret.get("Size", 0),
                    )

                    secrets.append(metadata)

            self.logger.info(f"Listed {len(secrets)} secrets")
            return secrets

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise SecretError(
                message=f"Failed to list secrets: {str(e)}",
                provider=self.provider_type.value,
                error_code=error_code,
                details={"error": str(e)},
            )

    async def rotate_secret(self, secret_name: str) -> SecretMetadata:
        """Rotate a secret in AWS Secrets Manager."""
        self._validate_secret_name(secret_name)

        try:
            if not self.rotation_lambda_arn:
                raise SecretError(
                    message="Rotation not configured - no Lambda ARN provided",
                    provider=self.provider_type.value,
                    secret_name=secret_name,
                    error_code="ROTATION_NOT_CONFIGURED",
                )

            # Configure rotation
            self._secrets_client.rotate_secret(
                SecretId=secret_name,
                RotationLambdaARN=self.rotation_lambda_arn,
                RotationRules={"AutomaticallyAfterDays": self.rotation_days},
            )

            # Get updated metadata
            metadata = await self.get_secret_metadata(secret_name)

            self._log_secret_access("rotate", secret_name, True)
            return metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            self._log_secret_access("rotate", secret_name, False, str(e))
            raise SecretError(
                message=f"Failed to rotate secret: {str(e)}",
                provider=self.provider_type.value,
                secret_name=secret_name,
                error_code=error_code,
                details={"error": str(e)},
            )

    async def get_secret_metadata(self, secret_name: str) -> SecretMetadata:
        """Get secret metadata without retrieving the value."""
        self._validate_secret_name(secret_name)

        try:
            response = self._secrets_client.describe_secret(SecretId=secret_name)

            metadata = self._create_secret_metadata(
                secret_name=secret_name,
                created_at=response.get("CreatedDate"),
                updated_at=response.get("LastChangedDate"),
                version=response.get("VersionIdsToStages", {}).get("AWSCURRENT"),
                tags=self._extract_tags_from_arn(response.get("ARN", "")),
                rotation_enabled=response.get("RotationEnabled", False),
                next_rotation=response.get("NextRotationDate"),
                size_bytes=response.get("Size", 0),
            )

            return metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                raise SecretNotFoundError(secret_name, self.provider_type.value)
            else:
                raise SecretError(
                    message=f"Failed to get secret metadata: {str(e)}",
                    provider=self.provider_type.value,
                    secret_name=secret_name,
                    error_code=error_code,
                    details={"error": str(e)},
                )

    def _extract_tags_from_arn(self, arn: str) -> dict[str, str]:
        """Extract tags from secret ARN (simplified implementation)."""
        # In a real implementation, you would use the describe_secret API
        # to get the actual tags. This is a placeholder.
        return {}

    async def encrypt_secret(self, secret_value: str, key_id: str | None = None) -> str:
        """Encrypt secret value using AWS KMS."""
        try:
            kms_key_id = key_id or self.kms_key_id
            if not kms_key_id:
                raise SecretError(
                    message="No KMS key specified for encryption",
                    provider=self.provider_type.value,
                    error_code="NO_KMS_KEY",
                )

            response = self._kms_client.encrypt(
                KeyId=kms_key_id,
                Plaintext=secret_value,
                EncryptionContext=self.encryption_context,
            )

            return str(response["CiphertextBlob"].decode("utf-8"))

        except ClientError as e:
            raise SecretError(
                message=f"Failed to encrypt secret: {str(e)}",
                provider=self.provider_type.value,
                error_code="ENCRYPTION_FAILED",
                details={"error": str(e)},
            )

    async def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt secret value using AWS KMS."""
        try:
            response = self._kms_client.decrypt(
                CiphertextBlob=encrypted_value.encode("utf-8"),
                EncryptionContext=self.encryption_context,
            )

            return str(response["Plaintext"].decode("utf-8"))

        except ClientError as e:
            raise SecretError(
                message=f"Failed to decrypt secret: {str(e)}",
                provider=self.provider_type.value,
                error_code="DECRYPTION_FAILED",
                details={"error": str(e)},
            )
