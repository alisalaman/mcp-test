"""Database infrastructure layer."""

from .base import (
    BaseRepository,
    ConnectionError,
    DuplicateError,
    NotFoundError,
    Repository,
    RepositoryError,
    ValidationError,
)
from .factory import (
    RepositoryFactory,
    RepositoryHealthMonitor,
    cleanup_repository,
    get_repository,
    setup_repository,
)
from .memory import InMemoryRepository
from .postgresql import PostgreSQLRepository
from .redis import RedisRepository

__all__ = [
    # Base interfaces and exceptions
    "Repository",
    "BaseRepository",
    "RepositoryError",
    "NotFoundError",
    "DuplicateError",
    "ConnectionError",
    "ValidationError",
    # Repository implementations
    "InMemoryRepository",
    "RedisRepository",
    "PostgreSQLRepository",
    # Factory and DI
    "RepositoryFactory",
    "get_repository",
    "setup_repository",
    "cleanup_repository",
    "RepositoryHealthMonitor",
]
