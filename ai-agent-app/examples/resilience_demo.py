"""Demonstration of resilience patterns implementation.

This example shows how to use the comprehensive resilience patterns
including retry logic, circuit breakers, health checking, fallback
strategies, and rate limiting.
"""

import asyncio
import time
from typing import Any

from ai_agent.resilience import (
    CircuitBreakerManager,
    FallbackManager,
    HealthMonitorConfig,
    RateLimitManager,
    ResilienceManager,
    RetryManager,
    ServiceHealthMonitor,
)
from ai_agent.resilience.circuit_breaker import CircuitBreakerSettings
from ai_agent.resilience.fallback import FallbackConfig
from ai_agent.resilience.health import DatabaseHealthChecker, HTTPHealthChecker
from ai_agent.resilience.rate_limiting import RateLimitConfig
from ai_agent.resilience.retry import RetrySettings


async def simulate_llm_call(prompt: str) -> str:
    """Simulate an LLM API call with occasional failures."""
    await asyncio.sleep(0.1)  # Simulate network delay

    # Simulate 20% failure rate
    if time.time() % 5 < 1:
        raise Exception("LLM API temporarily unavailable")

    return f"LLM Response to: {prompt}"


async def simulate_database_query(query: str) -> dict[str, Any]:
    """Simulate a database query with occasional failures."""
    await asyncio.sleep(0.05)  # Simulate database delay

    # Simulate 15% failure rate
    if time.time() % 7 < 1:
        raise Exception("Database connection lost")

    return {"query": query, "results": ["result1", "result2", "result3"]}


async def simulate_mcp_tool_call(
    tool_name: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Simulate an MCP tool call with occasional failures."""
    await asyncio.sleep(0.2)  # Simulate MCP server delay

    # Simulate 10% failure rate
    if time.time() % 10 < 1:
        raise Exception("MCP server timeout")

    return {"tool": tool_name, "params": params, "result": "success"}


async def simulate_secret_retrieval(secret_name: str) -> str:
    """Simulate secret retrieval with occasional failures."""
    await asyncio.sleep(0.03)  # Simulate secret manager delay

    # Simulate 5% failure rate
    if time.time() % 20 < 1:
        raise Exception("Secret manager unavailable")

    return f"secret_value_for_{secret_name}"


async def demo_retry_mechanisms():
    """Demonstrate retry mechanisms."""
    print("\n=== Retry Mechanisms Demo ===")

    # Initialize retry manager
    retry_settings = RetrySettings()
    retry_manager = RetryManager(retry_settings)

    # Get LLM retry decorator
    llm_retry = retry_manager.get_llm_retry_decorator()

    # Apply retry decorator to LLM function
    @llm_retry
    async def resilient_llm_call(prompt: str) -> str:
        return await simulate_llm_call(prompt)

    # Test with multiple calls
    for i in range(5):
        try:
            result = await resilient_llm_call(f"Test prompt {i}")
            print(f"✅ LLM call {i+1}: {result}")
        except Exception as e:
            print(f"❌ LLM call {i+1} failed: {e}")


async def demo_circuit_breakers():
    """Demonstrate circuit breaker patterns."""
    print("\n=== Circuit Breaker Demo ===")

    # Initialize circuit breaker manager
    circuit_settings = CircuitBreakerSettings()
    circuit_manager = CircuitBreakerManager(circuit_settings)

    # Get circuit breaker for database
    db_breaker = circuit_manager.get_breaker("database")

    # Simulate database calls with circuit breaker
    for i in range(10):
        try:
            result = await db_breaker.call(
                simulate_database_query, f"SELECT * FROM table_{i}"
            )
            print(f"✅ Database call {i+1}: {result['query']}")
        except Exception as e:
            print(f"❌ Database call {i+1} failed: {e}")
            print(f"   Circuit state: {db_breaker.get_state_info()['state']}")


async def demo_fallback_strategies():
    """Demonstrate fallback strategies."""
    print("\n=== Fallback Strategies Demo ===")

    # Initialize fallback manager
    fallback_config = FallbackConfig()
    fallback_manager = FallbackManager(fallback_config)

    # Create custom fallback for MCP
    from ai_agent.resilience.fallback.strategies import DefaultValueFallbackStrategy

    default_fallback = DefaultValueFallbackStrategy(
        fallback_config, {"error": "MCP unavailable"}
    )
    fallback_manager.create_custom_handler("mcp", [default_fallback])

    # Test MCP calls with fallback
    for i in range(5):
        try:
            result = await fallback_manager.handle_service_call(
                "mcp", simulate_mcp_tool_call, f"tool_{i}", {"param": f"value_{i}"}
            )
            print(f"✅ MCP call {i+1}: {result}")
        except Exception as e:
            print(f"❌ MCP call {i+1} failed: {e}")


async def demo_rate_limiting():
    """Demonstrate rate limiting."""
    print("\n=== Rate Limiting Demo ===")

    # Initialize rate limit manager
    rate_manager = RateLimitManager()

    # Add rate limiter for LLM (5 requests per minute)
    llm_config = RateLimitConfig(requests_per_minute=5, strategy="token_bucket")
    rate_manager.add_limiter("llm", llm_config)

    # Test rate limiting
    for i in range(8):
        try:
            result = await rate_manager.consume("llm", "user_123")
            if result.allowed:
                print(
                    f"✅ Rate limit check {i+1}: Allowed (remaining: {result.remaining})"
                )
            else:
                print(
                    f"⏳ Rate limit check {i+1}: Rate limited (retry after: {result.retry_after}s)"
                )
        except Exception as e:
            print(f"❌ Rate limit check {i+1} failed: {e}")


async def demo_health_checking():
    """Demonstrate health checking."""
    print("\n=== Health Checking Demo ===")

    # Create health checkers
    db_health_checker = DatabaseHealthChecker(
        "database", lambda: simulate_database_query("SELECT 1")
    )

    http_health_checker = HTTPHealthChecker(
        "api_service", "https://httpbin.org/status/200"
    )

    # Initialize health monitor
    health_config = HealthMonitorConfig(check_interval=5.0, timeout=3.0)
    health_monitor = ServiceHealthMonitor(health_config)

    # Add health checkers
    health_monitor.add_checker("database", db_health_checker)
    health_monitor.add_checker("api_service", http_health_checker)

    # Start monitoring
    await health_monitor.start_monitoring()

    # Let it run for a bit
    await asyncio.sleep(10)

    # Get health status
    health_status = health_monitor.get_all_health()
    print("Health Status:")
    for service, status in health_status.items():
        print(
            f"  {service}: {status['status']} (success rate: {status['success_rate']:.2%})"
        )

    # Stop monitoring
    await health_monitor.stop_monitoring()


async def demo_integrated_resilience():
    """Demonstrate integrated resilience patterns."""
    print("\n=== Integrated Resilience Demo ===")

    # Initialize all resilience components
    retry_settings = RetrySettings()
    retry_manager = RetryManager(retry_settings)

    circuit_settings = CircuitBreakerSettings()
    circuit_manager = CircuitBreakerManager(circuit_settings)

    fallback_config = FallbackConfig()
    fallback_manager = FallbackManager(fallback_config)

    rate_manager = RateLimitManager()
    llm_config = RateLimitConfig(requests_per_minute=10, strategy="sliding_window")
    rate_manager.add_limiter("llm", llm_config)

    # Create resilience manager
    resilience_manager = ResilienceManager(
        retry_manager=retry_manager,
        circuit_breaker_manager=circuit_manager,
        fallback_manager=fallback_manager,
        rate_limit_manager=rate_manager,
    )

    # Get integrated decorators
    llm_decorator = resilience_manager.get_llm_decorator()
    db_decorator = resilience_manager.get_database_decorator()

    # Apply integrated resilience
    @llm_decorator
    async def resilient_llm(prompt: str) -> str:
        return await simulate_llm_call(prompt)

    @db_decorator
    async def resilient_db(query: str) -> dict[str, Any]:
        return await simulate_database_query(query)

    # Test integrated resilience
    print("Testing integrated LLM resilience:")
    for i in range(3):
        try:
            result = await resilient_llm(f"Integrated test {i}")
            print(f"  ✅ LLM {i+1}: {result}")
        except Exception as e:
            print(f"  ❌ LLM {i+1} failed: {e}")

    print("\nTesting integrated database resilience:")
    for i in range(3):
        try:
            result = await resilient_db(f"SELECT * FROM test_{i}")
            print(f"  ✅ DB {i+1}: {result['query']}")
        except Exception as e:
            print(f"  ❌ DB {i+1} failed: {e}")

    # Show statistics
    stats = resilience_manager.get_stats()
    print(f"\nResilience Statistics: {stats}")


async def main():
    """Run all resilience pattern demonstrations."""
    print("🚀 AI Agent Resilience Patterns Demonstration")
    print("=" * 50)

    try:
        await demo_retry_mechanisms()
        await demo_circuit_breakers()
        await demo_fallback_strategies()
        await demo_rate_limiting()
        await demo_health_checking()
        await demo_integrated_resilience()

        print("\n✅ All demonstrations completed successfully!")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
