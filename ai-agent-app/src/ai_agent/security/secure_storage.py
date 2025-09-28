"""Secure storage implementation for sensitive data."""

import json
import os
from datetime import datetime, UTC
from typing import Any
from uuid import UUID, uuid4

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

from ..observability.logging import get_logger

logger = get_logger(__name__)


class SecureStorageError(Exception):
    """Secure storage error."""

    pass


class EncryptedData(BaseModel):
    """Encrypted data container."""

    id: UUID = Field(default_factory=uuid4)
    encrypted_data: bytes
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecureStorage:
    """Secure storage for sensitive data using encryption."""

    def __init__(self, encryption_key: bytes | None = None):
        """Initialize secure storage.

        Args:
            encryption_key: Encryption key. If None, will generate or load from env.
        """
        self.logger = get_logger(__name__)

        if encryption_key is None:
            encryption_key = self._get_or_create_encryption_key()

        self.cipher = Fernet(encryption_key)
        self._cache: dict[str, EncryptedData] = {}

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key."""
        # In production, this should come from a secure key management system
        key_env = os.getenv("ENCRYPTION_KEY")
        if key_env:
            return key_env.encode()

        # For development, generate a key and store it
        key_file = os.path.join(os.getcwd(), ".encryption_key")
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()
        with open(key_file, "w") as f:
            f.write(key.decode())
        os.chmod(key_file, 0o600)  # Restrict permissions

        self.logger.warning(
            "Generated new encryption key for development. "
            "In production, use a proper key management system."
        )
        return key

    def encrypt_data(self, data: dict[str, Any]) -> bytes:
        """Encrypt data."""
        try:
            json_data = json.dumps(data, default=str).encode()
            return self.cipher.encrypt(json_data)
        except Exception as e:
            self.logger.error("Failed to encrypt data", error=str(e))
            raise SecureStorageError(f"Encryption failed: {e}") from e

    def decrypt_data(self, encrypted_data: bytes) -> dict[str, Any]:
        """Decrypt data."""
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_bytes.decode())
            if not isinstance(data, dict):
                raise SecureStorageError("Decrypted data is not a dictionary")
            return data
        except Exception as e:
            self.logger.error("Failed to decrypt data", error=str(e))
            raise SecureStorageError(f"Decryption failed: {e}") from e

    def store(self, key: str, data: dict[str, Any]) -> None:
        """Store encrypted data."""
        try:
            encrypted_data = self.encrypt_data(data)
            encrypted_record = EncryptedData(
                encrypted_data=encrypted_data, metadata={"key": key}
            )
            self._cache[key] = encrypted_record
            self.logger.debug("Stored encrypted data", key=key)
        except Exception as e:
            self.logger.error("Failed to store data", key=key, error=str(e))
            raise SecureStorageError(f"Storage failed: {e}") from e

    def retrieve(self, key: str) -> dict[str, Any] | None:
        """Retrieve and decrypt data."""
        try:
            if key not in self._cache:
                return None

            encrypted_record = self._cache[key]
            return self.decrypt_data(encrypted_record.encrypted_data)
        except Exception as e:
            self.logger.error("Failed to retrieve data", key=key, error=str(e))
            return None

    def delete(self, key: str) -> bool:
        """Delete stored data."""
        try:
            if key in self._cache:
                del self._cache[key]
                self.logger.debug("Deleted encrypted data", key=key)
                return True
            return False
        except Exception as e:
            self.logger.error("Failed to delete data", key=key, error=str(e))
            return False

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        return list(self._cache.keys())

    def clear(self) -> None:
        """Clear all stored data."""
        self._cache.clear()
        self.logger.info("Cleared all encrypted data")

    def health_check(self) -> dict[str, Any]:
        """Check storage health."""
        try:
            # Test encryption/decryption
            test_data = {"test": "data", "timestamp": datetime.now(UTC).isoformat()}
            encrypted = self.encrypt_data(test_data)
            decrypted = self.decrypt_data(encrypted)

            if decrypted == test_data:
                return {
                    "status": "healthy",
                    "message": "Encryption/decryption working correctly",
                    "stored_items": len(self._cache),
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Encryption/decryption test failed",
                }
        except Exception as e:
            return {"status": "unhealthy", "message": f"Health check failed: {e}"}


# Global secure storage instance
_secure_storage: SecureStorage | None = None


def get_secure_storage() -> SecureStorage:
    """Get global secure storage instance."""
    global _secure_storage
    if _secure_storage is None:
        _secure_storage = SecureStorage()
    return _secure_storage


def setup_secure_storage(encryption_key: bytes | None = None) -> SecureStorage:
    """Setup global secure storage."""
    global _secure_storage
    _secure_storage = SecureStorage(encryption_key)
    return _secure_storage
