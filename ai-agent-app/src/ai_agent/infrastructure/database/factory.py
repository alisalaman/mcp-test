"""
Repository factory pattern for creating storage backend instances.

This factory provides dependency injection support and automatic backend
selection based on configuration settings.
"""

import logging

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ai_agent.config.settings import ApplicationSettings
from ai_agent.infrastructure.database.base import Repository
from ai_agent.infrastructure.database.memory import InMemoryRepository
from ai_agent.infrastructure.database.postgresql import PostgreSQLRepository
from ai_agent.infrastructure.database.redis import RedisRepository


class RepositoryFactory:
    """Factory for creating repository instances based on configuration."""

    _instances: dict[str, Repository] = {}

    @staticmethod
    def create_repository(settings: ApplicationSettings) -> Repository:
        """
        Create repository based on configuration.

        Priority order:
        1. PostgreSQL (if use_database=True)
        2. Redis (if use_redis=True)
        3. In-Memory (default/fallback)
        """
        # Determine backend type
        if settings.use_database:
            try:
                backend_type = "postgresql"
                repository = PostgreSQLRepository(settings.database)
            except ImportError as e:
                logger.warning(
                    f"PostgreSQL not available: {e}, falling back to in-memory"
                )
                backend_type = "memory"
                repository = InMemoryRepository()
        elif settings.use_redis:
            try:
                backend_type = "redis"
                repository = RedisRepository(settings.redis)
            except ImportError as e:
                logger.warning(f"Redis not available: {e}, falling back to in-memory")
                backend_type = "memory"
                repository = InMemoryRepository()
        else:
            backend_type = "memory"
            repository = InMemoryRepository()

        logger.info(
            "Created repository instance",
            backend_type=backend_type,
            environment=settings.environment.value,
        )

        return repository

    @staticmethod
    def get_or_create_repository(settings: ApplicationSettings) -> Repository:
        """
        Get existing repository instance or create new one.

        This ensures singleton behavior for repository instances.
        """
        # Create cache key based on settings
        if settings.use_database:
            cache_key = f"postgresql_{settings.database.host}_{settings.database.port}_{settings.database.name}"
        elif settings.use_redis:
            cache_key = (
                f"redis_{settings.redis.host}_{settings.redis.port}_{settings.redis.db}"
            )
        else:
            cache_key = "memory"

        # Return existing instance if available
        if cache_key in RepositoryFactory._instances:
            return RepositoryFactory._instances[cache_key]

        # Create new instance
        repository = RepositoryFactory.create_repository(settings)
        RepositoryFactory._instances[cache_key] = repository

        return repository

    @staticmethod
    async def initialize_repository(repository: Repository) -> Repository:
        """
        Initialize repository connection and perform health check.
        """
        try:
            await repository.connect()

            if await repository.health_check():
                logger.info("Repository connection established successfully")
                return repository
            else:
                logger.error("Repository health check failed")
                raise RuntimeError("Repository health check failed")

        except Exception as e:
            logger.error("Failed to initialize repository", error=str(e))
            raise

    @staticmethod
    async def cleanup_repositories() -> None:
        """Clean up all repository instances."""
        for repository in RepositoryFactory._instances.values():
            try:
                await repository.disconnect()
            except Exception as e:
                logger.warning("Error during repository cleanup", error=str(e))

        RepositoryFactory._instances.clear()
        logger.info("All repository instances cleaned up")


# Dependency injection helpers
_repository_instance: Repository | None = None


async def get_repository(settings: ApplicationSettings) -> Repository:
    """
    Dependency injection function for FastAPI.

    This function can be used with FastAPI's Depends() to inject
    the repository into route handlers.
    """
    global _repository_instance

    if _repository_instance is None:
        _repository_instance = RepositoryFactory.get_or_create_repository(settings)
        await RepositoryFactory.initialize_repository(_repository_instance)

    return _repository_instance


async def setup_repository(settings: ApplicationSettings) -> Repository:
    """
    Setup repository for application startup.

    This should be called during application initialization.
    """
    repository = RepositoryFactory.create_repository(settings)
    await RepositoryFactory.initialize_repository(repository)

    global _repository_instance
    _repository_instance = repository

    logger.info("Repository setup completed")
    return repository


async def cleanup_repository() -> None:
    """
    Cleanup repository for application shutdown.

    This should be called during application shutdown.
    """
    global _repository_instance

    if _repository_instance:
        try:
            await _repository_instance.disconnect()
            logger.info("Repository disconnected successfully")
        except Exception as e:
            logger.warning("Error during repository disconnection", error=str(e))
        finally:
            _repository_instance = None

    await RepositoryFactory.cleanup_repositories()


# Repository health monitoring
class RepositoryHealthMonitor:
    """Monitor repository health and provide fallback strategies."""

    def __init__(
        self,
        primary_repository: Repository,
        fallback_repository: Repository | None = None,
    ):
        self.primary_repository = primary_repository
        self.fallback_repository = fallback_repository
        self._using_fallback = False

    async def get_healthy_repository(self) -> Repository:
        """Get a healthy repository instance, falling back if necessary."""
        # Check primary repository health
        if await self.primary_repository.health_check():
            if self._using_fallback:
                logger.info("Primary repository recovered, switching back")
                self._using_fallback = False
            return self.primary_repository

        # Use fallback if available
        if self.fallback_repository:
            if await self.fallback_repository.health_check():
                if not self._using_fallback:
                    logger.warning(
                        "Primary repository unhealthy, switching to fallback"
                    )
                    self._using_fallback = True
                return self.fallback_repository

        # No healthy repository available
        logger.error("No healthy repository available")
        raise RuntimeError("No healthy repository available")

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using fallback repository."""
        return self._using_fallback


# Factory configuration validation
def validate_repository_config(settings: ApplicationSettings) -> bool:
    """Validate repository configuration settings."""
    errors = []

    # Check database configuration if enabled
    if settings.use_database:
        if not settings.database.host:
            errors.append("Database host is required when use_database=True")
        if not settings.database.name:
            errors.append("Database name is required when use_database=True")
        if not settings.database.user:
            errors.append("Database user is required when use_database=True")

    # Check Redis configuration if enabled
    if settings.use_redis:
        if not settings.redis.host:
            errors.append("Redis host is required when use_redis=True")
        if settings.redis.port <= 0:
            errors.append("Redis port must be positive when use_redis=True")

    # Warn about memory-only configuration in production
    if (
        settings.is_production
        and settings.use_memory
        and not settings.use_database
        and not settings.use_redis
    ):
        errors.append("Production environment should not use memory-only storage")

    if errors:
        for error in errors:
            logger.error("Repository configuration error", error=error)
        return False

    return True
