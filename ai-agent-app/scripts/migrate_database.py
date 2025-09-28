#!/usr/bin/env python3
"""
Database migration script for AI Agent application.

This script handles database schema creation and migration management
for PostgreSQL databases.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg
import structlog

from ai_agent.config.settings import get_settings
from ai_agent.infrastructure.database.postgresql import PostgreSQLRepository

logger = structlog.get_logger()


class DatabaseMigrator:
    """Handles database schema migrations."""

    def __init__(self, settings):
        self.settings = settings
        self.migrations_dir = (
            Path(__file__).parent.parent
            / "src"
            / "ai_agent"
            / "infrastructure"
            / "database"
            / "migrations"
        )

    async def create_database_if_not_exists(self) -> bool:
        """Create database if it doesn't exist."""
        try:
            # Connect to default postgres database first
            conn = await asyncpg.connect(
                host=self.settings.database.host,
                port=self.settings.database.port,
                user=self.settings.database.user,
                password=self.settings.database.password,
                database="postgres",  # Connect to default database
            )

            # Check if target database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.settings.database.name,
            )

            if not db_exists:
                # Create database
                await conn.execute(f'CREATE DATABASE "{self.settings.database.name}"')
                logger.info(f"Created database: {self.settings.database.name}")
                await conn.close()
                return True
            else:
                logger.info(f"Database already exists: {self.settings.database.name}")
                await conn.close()
                return False

        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise

    async def get_migration_files(self) -> list[Path]:
        """Get all migration files in order."""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return []

        migration_files = sorted(
            [f for f in self.migrations_dir.glob("*.sql") if f.is_file()]
        )

        return migration_files

    async def create_migrations_table(self, conn: asyncpg.Connection):
        """Create migrations tracking table."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    async def get_applied_migrations(self, conn: asyncpg.Connection) -> list[str]:
        """Get list of already applied migrations."""
        try:
            rows = await conn.fetch(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            return [row["version"] for row in rows]
        except asyncpg.UndefinedTableError:
            # Table doesn't exist yet
            return []

    async def apply_migration(self, conn: asyncpg.Connection, migration_file: Path):
        """Apply a single migration file."""
        logger.info(f"Applying migration: {migration_file.name}")

        # Read migration content
        migration_sql = migration_file.read_text()

        async with conn.transaction():
            # Execute migration
            await conn.execute(migration_sql)

            # Record migration as applied
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration_file.stem,
            )

        logger.info(f"Successfully applied migration: {migration_file.name}")

    async def run_migrations(self):
        """Run all pending migrations."""
        logger.info("Starting database migrations")

        # Create database if needed
        await self.create_database_if_not_exists()

        # Connect to target database
        repository = PostgreSQLRepository(self.settings.database)
        await repository.connect()

        try:
            async with repository.get_connection() as conn:
                # Create migrations table
                await self.create_migrations_table(conn)

                # Get migration files and applied migrations
                migration_files = await self.get_migration_files()
                applied_migrations = await self.get_applied_migrations(conn)

                # Apply pending migrations
                pending_migrations = [
                    f for f in migration_files if f.stem not in applied_migrations
                ]

                if not pending_migrations:
                    logger.info("No pending migrations to apply")
                    return

                logger.info(f"Found {len(pending_migrations)} pending migration(s)")

                for migration_file in pending_migrations:
                    await self.apply_migration(conn, migration_file)

                logger.info("All migrations completed successfully")

        finally:
            await repository.disconnect()

    async def reset_database(self):
        """Reset database by dropping and recreating it."""
        logger.warning("Resetting database - ALL DATA WILL BE LOST!")

        try:
            # Connect to default postgres database
            conn = await asyncpg.connect(
                host=self.settings.database.host,
                port=self.settings.database.port,
                user=self.settings.database.user,
                password=self.settings.database.password,
                database="postgres",
            )

            # Terminate connections to target database
            await conn.execute(
                f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{self.settings.database.name}' AND pid <> pg_backend_pid()
            """
            )

            # Drop database if exists
            await conn.execute(
                f'DROP DATABASE IF EXISTS "{self.settings.database.name}"'
            )
            logger.info(f"Dropped database: {self.settings.database.name}")

            await conn.close()

            # Run migrations to recreate schema
            await self.run_migrations()

        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise

    async def check_connection(self):
        """Check database connection."""
        try:
            repository = PostgreSQLRepository(self.settings.database)
            await repository.connect()

            if await repository.health_check():
                logger.info("Database connection successful")
            else:
                logger.error("Database health check failed")
                return False

            await repository.disconnect()
            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False


async def main():
    """Main migration script."""
    import argparse

    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "command",
        choices=["migrate", "reset", "check"],
        help="Migration command to run",
    )
    parser.add_argument(
        "--environment",
        default="development",
        help="Environment to use (default: development)",
    )

    args = parser.parse_args()

    # Set environment
    import os

    os.environ["ENVIRONMENT"] = args.environment

    # Get settings
    settings = get_settings()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    migrator = DatabaseMigrator(settings)

    try:
        if args.command == "migrate":
            await migrator.run_migrations()
        elif args.command == "reset":
            await migrator.reset_database()
        elif args.command == "check":
            success = await migrator.check_connection()
            sys.exit(0 if success else 1)

        logger.info(f"Migration command '{args.command}' completed successfully")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
