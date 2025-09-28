"""Encryption and data protection utilities."""

import base64
import hashlib
import secrets
from typing import Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from ..observability.logging import get_logger

logger = get_logger(__name__)


class EncryptionError(Exception):
    """Encryption error."""

    pass


class DecryptionError(Exception):
    """Decryption error."""

    pass


class EncryptionService:
    """Main encryption service."""

    def __init__(self, master_key: str | None = None):
        self.master_key = master_key or self._generate_master_key()
        self.logger = get_logger(__name__)

        # Initialize Fernet with master key
        key = self._derive_key(self.master_key)
        self.cipher = Fernet(key)

    def _generate_master_key(self) -> str:
        """Generate a new master key."""
        return Fernet.generate_key().decode()

    def _derive_key(self, password: str, salt: bytes | None = None) -> bytes:
        """Derive encryption key from password."""
        if salt is None:
            salt = b"ai_agent_salt"  # In production, use random salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a string."""
        try:
            encrypted_bytes = self.cipher.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt string: {e}")

    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt a string."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt string: {e}")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt a dictionary."""
        import json

        try:
            json_str = json.dumps(data, default=str)
            return self.encrypt_string(json_str)
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt dictionary: {e}")

    def decrypt_dict(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt a dictionary."""
        import json

        try:
            json_str = self.decrypt_string(encrypted_data)
            result = json.loads(json_str)
            if not isinstance(result, dict):
                raise DecryptionError("Decrypted data is not a dictionary")
            return result
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt dictionary: {e}")

    def hash_password(self, password: str, salt: str | None = None) -> str:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)

        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
            backend=default_backend(),
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode())).decode()
        return f"{salt}:{key}"

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, key = hashed_password.split(":", 1)
            expected_key = self.hash_password(password, salt).split(":", 1)[1]
            return secrets.compare_digest(key, expected_key)
        except Exception:
            return False

    def generate_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    def generate_api_key(self, prefix: str = "sk-") -> str:
        """Generate a secure API key."""
        return f"{prefix}{secrets.token_urlsafe(32)}"

    def create_hmac(self, message: str, key: str | None = None) -> str:
        """Create HMAC for message authentication."""
        if key is None:
            key = self.master_key

        hmac_obj = hashlib.sha256()
        hmac_obj.update(key.encode())
        hmac_obj.update(message.encode())
        return hmac_obj.hexdigest()

    def verify_hmac(self, message: str, hmac: str, key: str | None = None) -> bool:
        """Verify HMAC for message authentication."""
        expected_hmac = self.create_hmac(message, key)
        return secrets.compare_digest(hmac, expected_hmac)


class AsymmetricEncryption:
    """Asymmetric encryption utilities."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def generate_key_pair(self) -> tuple[bytes, bytes]:
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    def encrypt_with_public_key(self, message: str, public_key_pem: bytes) -> str:
        """Encrypt message with public key."""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            # Only RSA keys support encryption
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise ValueError("Only RSA public keys are supported for encryption")

            encrypted = public_key.encrypt(
                message.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt with public key: {e}")

    def decrypt_with_private_key(
        self, encrypted_message: str, private_key_pem: bytes
    ) -> str:
        """Decrypt message with private key."""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None, backend=default_backend()
            )

            # Only RSA keys support decryption
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise ValueError("Only RSA private keys are supported for decryption")

            encrypted_bytes = base64.urlsafe_b64decode(encrypted_message.encode())
            decrypted = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            return decrypted.decode()
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt with private key: {e}")


class DataMasking:
    """Data masking utilities for PII protection."""

    @staticmethod
    def mask_email(email: str, visible_chars: int = 2) -> str:
        """Mask email address."""
        if "@" not in email:
            return email

        local, domain = email.split("@", 1)

        if len(local) <= visible_chars:
            masked_local = "*" * len(local)
        else:
            masked_local = local[:visible_chars] + "*" * (len(local) - visible_chars)

        return f"{masked_local}@{domain}"

    @staticmethod
    def mask_phone(phone: str, visible_digits: int = 4) -> str:
        """Mask phone number."""
        # Remove non-digit characters
        digits = "".join(filter(str.isdigit, phone))

        if len(digits) <= visible_digits:
            return "*" * len(phone)

        # Keep last visible_digits
        masked_digits = "*" * (len(digits) - visible_digits) + digits[-visible_digits:]

        # Restore original format
        result = phone
        digit_index = 0
        for i, char in enumerate(phone):
            if char.isdigit():
                result = result[:i] + masked_digits[digit_index] + result[i + 1 :]
                digit_index += 1

        return result

    @staticmethod
    def mask_credit_card(card_number: str, visible_digits: int = 4) -> str:
        """Mask credit card number."""
        digits = "".join(filter(str.isdigit, card_number))

        if len(digits) <= visible_digits:
            return "*" * len(card_number)

        masked_digits = "*" * (len(digits) - visible_digits) + digits[-visible_digits:]

        # Restore original format
        result = card_number
        digit_index = 0
        for i, char in enumerate(card_number):
            if char.isdigit():
                result = result[:i] + masked_digits[digit_index] + result[i + 1 :]
                digit_index += 1

        return result

    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """Mask Social Security Number."""
        digits = "".join(filter(str.isdigit, ssn))

        if len(digits) != 9:
            return "*" * len(ssn)

        # Format: XXX-XX-XXXX, show only last 4 digits
        "XXX-XX-" + digits[-4:]

        # Restore original format
        result = ssn
        digit_index = 0
        for i, char in enumerate(ssn):
            if char.isdigit():
                if digit_index < 5:
                    result = result[:i] + "X" + result[i + 1 :]
                else:
                    result = result[:i] + digits[digit_index] + result[i + 1 :]
                digit_index += 1

        return result

    @staticmethod
    def mask_ip_address(ip: str) -> str:
        """Mask IP address."""
        parts = ip.split(".")
        if len(parts) != 4:
            return "*" * len(ip)

        # Show only first octet
        return f"{parts[0]}.xxx.xxx.xxx"


class SecureStorage:
    """Secure storage utilities."""

    def __init__(self, encryption_service: EncryptionService):
        self.encryption_service = encryption_service
        self.logger = get_logger(__name__)

    def store_sensitive_data(self, key: str, data: dict[str, Any]) -> str:
        """Store sensitive data encrypted."""
        try:
            encrypted_data = self.encryption_service.encrypt_dict(data)

            # In production, this would store to a secure database
            # For now, just return the encrypted data
            self.logger.info(f"Stored sensitive data for key: {key}")
            return encrypted_data
        except Exception as e:
            self.logger.error(f"Failed to store sensitive data: {e}")
            raise

    def retrieve_sensitive_data(self, key: str, encrypted_data: str) -> dict[str, Any]:
        """Retrieve and decrypt sensitive data."""
        try:
            data = self.encryption_service.decrypt_dict(encrypted_data)
            self.logger.info(f"Retrieved sensitive data for key: {key}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to retrieve sensitive data: {e}")
            raise

    def delete_sensitive_data(self, key: str) -> bool:
        """Delete sensitive data."""
        try:
            # In production, this would delete from secure database
            self.logger.info(f"Deleted sensitive data for key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete sensitive data: {e}")
            return False


# Global encryption service instance
_encryption_service: EncryptionService | None = None


def get_encryption_service() -> EncryptionService:
    """Get global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def setup_encryption(master_key: str | None = None) -> EncryptionService:
    """Setup global encryption service."""
    global _encryption_service
    _encryption_service = EncryptionService(master_key)
    return _encryption_service
