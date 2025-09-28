#!/usr/bin/env python3
"""
Phase 2 Infrastructure Layer Demo

This script demonstrates the configuration management system and repository
pattern implementations from Phase 2 of the AI Agent implementation plan.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_agent.config.settings import ConfigurationValidator, get_settings
from ai_agent.domain.models import Message, MessageRole, Session
from ai_agent.infrastructure.database import (
    RepositoryFactory,
    cleanup_repository,
    setup_repository,
)


async def demo_configuration_system():
    """Demonstrate the configuration management system."""
    print("=" * 60)
    print("PHASE 2.1: Configuration Management System Demo")
    print("=" * 60)

    # Get settings based on environment
    settings = get_settings()

    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"App name: {settings.app_name}")
    print(f"Server: {settings.host}:{settings.port}")
    print()

    # Show storage backend selection
    print("Storage Backend Configuration:")
    print(f"  Use Database: {settings.use_database}")
    print(f"  Use Redis: {settings.use_redis}")
    print(f"  Use Memory: {settings.use_memory}")
    print()

    # Show component configurations
    print("Component Configurations:")
    print(f"  Database URL: {settings.database.url}")
    print(f"  Redis URL: {settings.redis.url}")
    print(f"  Log Level: {settings.observability.log_level}")
    print(f"  Metrics Enabled: {settings.observability.metrics_enabled}")
    print()

    # Show feature flags
    print("Feature Flags:")
    print(f"  Circuit Breakers: {settings.features.enable_circuit_breakers}")
    print(f"  WebSockets: {settings.features.enable_websockets}")
    print(f"  Debug Endpoints: {settings.features.enable_debug_endpoints}")
    print()

    # Validate configuration
    print("Configuration Validation:")
    errors = ConfigurationValidator.validate_settings(settings)
    if errors:
        print("  Validation errors found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  ✓ Configuration is valid")
    print()


async def demo_repository_pattern():
    """Demonstrate the repository pattern implementations."""
    print("=" * 60)
    print("PHASE 2.2: Repository Pattern Demo")
    print("=" * 60)

    # Get settings and create repository
    settings = get_settings()

    # Determine which repository will be used
    if settings.use_database:
        repo_type = "PostgreSQL"
    elif settings.use_redis:
        repo_type = "Redis"
    else:
        repo_type = "In-Memory"

    print(f"Using {repo_type} repository")
    print()

    try:
        # Setup repository
        repository = await setup_repository(settings)
        print("✓ Repository connected successfully")

        # Health check
        healthy = await repository.health_check()
        print(f"✓ Repository health check: {'passed' if healthy else 'failed'}")
        print()

        # Demo session operations
        print("Session Operations Demo:")

        # Create a test session
        session = Session(
            id=uuid4(),
            user_id="demo_user",
            title="Phase 2 Demo Session",
            metadata={"demo": True, "phase": 2},
        )

        created_session = await repository.create_session(session)
        print(f"  ✓ Created session: {created_session.id}")

        # Get the session back
        retrieved_session = await repository.get_session(session.id)
        print(f"  ✓ Retrieved session: {retrieved_session.title}")

        # List sessions
        sessions = await repository.list_sessions(user_id="demo_user", limit=10)
        print(f"  ✓ Found {len(sessions)} session(s) for user")
        print()

        # Demo message operations
        print("Message Operations Demo:")

        # Create test messages
        messages_data = [
            {"role": MessageRole.USER, "content": "Hello, this is a test message"},
            {
                "role": MessageRole.ASSISTANT,
                "content": "Hello! I'm demonstrating the repository pattern.",
            },
            {
                "role": MessageRole.USER,
                "content": "Can you show me how sessions and messages work?",
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "Sure! This demo shows CRUD operations for sessions and messages.",
            },
        ]

        created_messages = []
        for msg_data in messages_data:
            message = Message(
                id=uuid4(),
                session_id=session.id,
                role=msg_data["role"],
                content=msg_data["content"],
                metadata={"demo": True},
            )
            created_message = await repository.create_message(message)
            created_messages.append(created_message)
            role_str = (
                created_message.role.value
                if hasattr(created_message.role, "value")
                else str(created_message.role)
            )
            print(f"  ✓ Created message: {role_str}")

        # Get messages for session
        session_messages = await repository.get_messages_by_session(
            session.id, limit=10
        )
        print(f"  ✓ Retrieved {len(session_messages)} message(s) for session")

        # Show conversation
        print("\n  Conversation:")
        for msg in session_messages:
            role_icon = "👤" if msg.role == MessageRole.USER else "🤖"
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            print(f"    {role_icon} {role_str}: {msg.content[:50]}...")
        print()

        # Update session title
        retrieved_session.title = "Updated Phase 2 Demo Session"
        updated_session = await repository.update_session(retrieved_session)
        print(f"  ✓ Updated session title: {updated_session.title}")
        print()

        # Demo repository factory
        print("Repository Factory Demo:")
        factory_repo = RepositoryFactory.create_repository(settings)
        print(f"  ✓ Factory created {type(factory_repo).__name__}")

        # Test different configurations
        print(f"  ✓ Memory repository available: {settings.use_memory}")
        print(f"  ✓ Redis repository available: {settings.use_redis}")
        print(f"  ✓ PostgreSQL repository available: {settings.use_database}")
        print()

        # Cleanup demo data
        print("Cleanup:")
        for message in created_messages:
            await repository.delete_message(message.id)
        print(f"  ✓ Deleted {len(created_messages)} test messages")

        await repository.delete_session(session.id)
        print("  ✓ Deleted test session")
        print()

    except Exception as e:
        print(f"❌ Repository demo failed: {e}")
        return False

    finally:
        # Cleanup repository
        await cleanup_repository()
        print("✓ Repository cleanup completed")
        print()

    return True


async def main():
    """Run Phase 2 infrastructure demo."""
    print("AI Agent Application - Phase 2 Infrastructure Layer Demo")
    print("This demo showcases the configuration management and repository pattern")
    print("implementations from Phase 2 of the development plan.")
    print()

    try:
        # Demo configuration system
        await demo_configuration_system()

        # Demo repository pattern
        success = await demo_repository_pattern()

        if success:
            print("🎉 Phase 2 Infrastructure Layer Demo completed successfully!")
            print()
            print("What was demonstrated:")
            print("  ✓ Environment-specific configuration management")
            print("  ✓ Configuration validation and factory pattern")
            print("  ✓ Multi-backend repository pattern (Memory/Redis/PostgreSQL)")
            print("  ✓ Complete CRUD operations for sessions and messages")
            print("  ✓ Repository factory with dependency injection")
            print("  ✓ Health checking and connection management")
        else:
            print("❌ Demo encountered some issues")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
