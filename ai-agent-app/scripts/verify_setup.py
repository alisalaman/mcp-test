#!/usr/bin/env python3
"""Verification script for Phase 1 implementation."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def verify_imports():
    """Verify that all domain models and exceptions can be imported."""
    try:
        # Test domain model imports
        # Test exception imports
        from ai_agent.domain.exceptions import (
            AIAgentException,
            AuthenticationException,
            AuthorizationException,
            CircuitBreakerOpenException,
            ExternalServiceException,
            RateLimitException,
            TimeoutException,
            ValidationException,
        )
        from ai_agent.domain.models import (
            Agent,
            AgentStatus,
            CircuitBreakerConfig,
            ErrorCode,
            ErrorDetail,
            ExternalService,
            ExternalServiceType,
            MCPServer,
            Message,
            MessageRole,
            RetryConfig,
            Session,
            Tool,
        )

        # Verify imports by checking they exist (this uses the imports)
        imports_to_verify = [
            AIAgentException,
            AuthenticationException,
            AuthorizationException,
            CircuitBreakerOpenException,
            ExternalServiceException,
            RateLimitException,
            TimeoutException,
            ValidationException,
            Agent,
            AgentStatus,
            CircuitBreakerConfig,
            ErrorCode,
            ErrorDetail,
            ExternalService,
            ExternalServiceType,
            MCPServer,
            Message,
            MessageRole,
            RetryConfig,
            Session,
            Tool,
        ]

        for import_item in imports_to_verify:
            assert import_item is not None

        print("✅ All domain models and exceptions imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def verify_pydantic_models():
    """Verify that Pydantic models are properly configured."""
    try:
        from ai_agent.domain.models import Agent, Message, Session

        # Test Agent model
        agent = Agent(name="test-agent", description="Test agent")
        assert agent.id is not None
        assert agent.name == "test-agent"
        assert agent.status == "idle"

        # Test Session model
        session = Session()
        assert session.id is not None
        assert session.message_count == 0

        # Test Message model
        message = Message(session_id=session.id, role="user", content="Hello world")
        assert message.session_id == session.id
        assert message.role == "user"

        print("✅ Pydantic model validation working correctly")
        return True

    except Exception as e:
        print(f"❌ Pydantic model error: {e}")
        return False


def verify_exceptions():
    """Verify that exception hierarchy is working."""
    try:
        from ai_agent.domain.exceptions import ErrorCode, ValidationException

        # Test exception creation
        exc = ValidationException("Test validation error", "test_field", "test_value")
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.details["field"] == "test_field"

        print("✅ Exception hierarchy working correctly")
        return True

    except Exception as e:
        print(f"❌ Exception error: {e}")
        return False


def verify_fastapi_app():
    """Verify that FastAPI app can be created."""
    try:
        from ai_agent.main import app

        assert app is not None
        assert app.title == "AI Agent Application"
        assert app.version == "0.1.0"

        print("✅ FastAPI application created successfully")
        return True

    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("🔍 Phase 1 Implementation Verification")
    print("=" * 40)

    tests = [
        verify_imports,
        verify_pydantic_models,
        verify_exceptions,
        verify_fastapi_app,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Phase 1 implementation verified successfully!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
