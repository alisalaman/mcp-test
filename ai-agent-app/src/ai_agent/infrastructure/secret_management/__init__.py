"""Secret management infrastructure for multi-cloud secret providers."""

from .base import SecretProvider, SecretError, SecretNotFoundError
from .factory import SecretManagerFactory, get_secret_manager
from .aws_secrets import AWSSecretProvider
from .azure_keyvault import AzureKeyVaultProvider
from .gcp_secrets import GCPSecretProvider
from .local_secrets import LocalSecretProvider

__all__ = [
    "SecretProvider",
    "SecretError",
    "SecretNotFoundError",
    "SecretManagerFactory",
    "get_secret_manager",
    "AWSSecretProvider",
    "AzureKeyVaultProvider",
    "GCPSecretProvider",
    "LocalSecretProvider",
]
