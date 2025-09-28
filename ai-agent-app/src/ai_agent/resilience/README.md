# Resilience Layer

This module provides comprehensive resilience patterns for external service integration, ensuring production-ready reliability and fault tolerance.

## Overview

The resilience layer implements the following patterns:

- **Retry Logic**: Exponential backoff with jitter for transient failures
- **Circuit Breakers**: Prevent cascading failures and enable quick recovery
- **Health Checking**: Monitor service health with configurable intervals
- **Fallback Strategies**: Graceful degradation when services are unavailable
- **Rate Limiting**: Protect external services from being overwhelmed

## Quick Start

```python
from ai_agent.resilience import (
    RetryManager,
    CircuitBreakerManager,
    FallbackManager,
    RateLimitManager,
    ResilienceManager,
    llm_resilient,
    database_resilient,
    mcp_resilient,
    secret_resilient,
)

# Initialize resilience components
retry_manager = RetryManager(RetrySettings())
circuit_manager = CircuitBreakerManager(CircuitBreakerSettings())
fallback_manager = FallbackManager(FallbackConfig())
rate_manager = RateLimitManager()

# Create integrated resilience manager
resilience_manager = ResilienceManager(
    retry_manager=retry_manager,
    circuit_breaker_manager=circuit_manager,
    fallback_manager=fallback_manager,
    rate_limit_manager=rate_manager
)

# Use service-specific decorators
@resilience_manager.get_llm_decorator()
async def call_llm_api(prompt: str) -> str:
    # Your LLM API call here
    pass

@resilience_manager.get_database_decorator()
async def query_database(query: str) -> dict:
    # Your database query here
    pass
```

## Components

### 1. Retry Management

Implements sophisticated retry mechanisms with exponential backoff and jitter.

```python
from ai_agent.resilience.retry import RetryManager, RetrySettings

# Configure retry settings
retry_settings = RetrySettings()
retry_manager = RetryManager(retry_settings)

# Get service-specific retry decorators
llm_retry = retry_manager.get_llm_retry_decorator()
db_retry = retry_manager.get_database_retry_decorator()

# Apply to functions
@llm_retry
async def call_llm(prompt: str) -> str:
    # LLM API call with automatic retries
    pass
```

**Features:**
- Service-specific retry configurations
- Exponential backoff with jitter
- Configurable retryable exceptions
- Comprehensive logging with correlation IDs

### 2. Circuit Breakers

Prevents cascading failures by opening circuits when services are unhealthy.

```python
from ai_agent.resilience.circuit_breaker import CircuitBreakerManager, CircuitBreakerSettings

# Initialize circuit breaker manager
circuit_manager = CircuitBreakerManager(CircuitBreakerSettings())

# Get circuit breaker for a service
breaker = circuit_manager.get_breaker("llm")

# Use circuit breaker protection
result = await breaker.call(your_function, *args, **kwargs)
```

**States:**
- **Closed**: Normal operation, counting failures
- **Open**: Blocking requests, waiting for recovery timeout
- **Half-Open**: Testing with limited requests

### 3. Health Checking

Monitors service health with configurable intervals and dependency validation.

```python
from ai_agent.resilience.health import ServiceHealthMonitor, DatabaseHealthChecker

# Create health checkers
db_checker = DatabaseHealthChecker("database", get_db_connection)
http_checker = HTTPHealthChecker("api", "https://api.example.com/health")

# Initialize health monitor
monitor = ServiceHealthMonitor(HealthMonitorConfig())
monitor.add_checker("database", db_checker)
monitor.add_checker("api", http_checker)

# Start monitoring
await monitor.start_monitoring()

# Get health status
health_status = monitor.get_all_health()
```

### 4. Fallback Strategies

Provides graceful degradation when services are unavailable.

```python
from ai_agent.resilience.fallback import FallbackManager, FallbackConfig

# Initialize fallback manager
fallback_manager = FallbackManager(FallbackConfig())

# Create custom fallback strategies
from ai_agent.resilience.fallback.strategies import (
    DefaultValueFallbackStrategy,
    CachedFallbackStrategy
)

# Add fallback strategies
default_fallback = DefaultValueFallbackStrategy(config, "Service unavailable")
cache_fallback = CachedFallbackStrategy(config)

# Create custom handler
handler = fallback_manager.create_custom_handler(
    "llm",
    [default_fallback, cache_fallback],
    default_value="I'm sorry, I'm currently unable to help."
)

# Use fallback protection
result = await fallback_manager.handle_service_call(
    "llm", your_function, *args, **kwargs
)
```

### 5. Rate Limiting

Protects external services from being overwhelmed.

```python
from ai_agent.resilience.rate_limiting import RateLimitManager, RateLimitConfig

# Initialize rate limit manager
rate_manager = RateLimitManager()

# Add rate limiters for different services
llm_config = RateLimitConfig(
    requests_per_minute=60,
    strategy="token_bucket"
)
rate_manager.add_limiter("llm", llm_config)

# Check rate limits
result = await rate_manager.consume("llm", "user_123")
if not result.allowed:
    print(f"Rate limited. Retry after {result.retry_after} seconds")
```

**Strategies:**
- **Token Bucket**: Smooth rate limiting with burst capacity
- **Sliding Window**: Precise rate limiting over time windows
- **Fixed Window**: Simple rate limiting with fixed time periods

## Integration Decorators

Use service-specific decorators for easy integration:

```python
from ai_agent.resilience import (
    llm_resilient,
    database_resilient,
    mcp_resilient,
    secret_resilient,
)

@llm_resilient()
async def call_openai_api(prompt: str) -> str:
    # Your OpenAI API call
    pass

@database_resilient()
async def query_postgres(query: str) -> list:
    # Your database query
    pass

@mcp_resilient()
async def call_mcp_tool(tool_name: str, params: dict) -> dict:
    # Your MCP tool call
    pass

@secret_resilient()
async def get_secret(secret_name: str) -> str:
    # Your secret retrieval
    pass
```

## Configuration

### Environment Variables

```bash
# Retry Configuration
RETRY_LLM_MAX_ATTEMPTS=3
RETRY_LLM_BASE_DELAY=1.0
RETRY_LLM_MAX_DELAY=60.0

# Circuit Breaker Configuration
CIRCUIT_LLM_FAILURE_THRESHOLD=5
CIRCUIT_LLM_RECOVERY_TIMEOUT=60.0

# Rate Limiting Configuration
RATE_LIMIT_LLM_REQUESTS_PER_MINUTE=60
RATE_LIMIT_LLM_STRATEGY=token_bucket

# Health Check Configuration
HEALTH_CHECK_INTERVAL=30.0
HEALTH_CHECK_TIMEOUT=5.0
```

### Programmatic Configuration

```python
from ai_agent.resilience.retry import RetrySettings
from ai_agent.resilience.circuit_breaker import CircuitBreakerSettings

# Custom retry settings
retry_settings = RetrySettings(
    llm_max_attempts=5,
    llm_base_delay=2.0,
    llm_max_delay=120.0
)

# Custom circuit breaker settings
circuit_settings = CircuitBreakerSettings(
    llm_failure_threshold=3,
    llm_recovery_timeout=30.0
)
```

## Monitoring and Observability

### Health Endpoints

```python
from ai_agent.resilience.health import HealthEndpoint

# Create health endpoint
health_endpoint = HealthEndpoint(monitor, app_version="1.0.0")

# Get overall health
health = await health_endpoint.get_health()
print(f"Overall status: {health.status}")

# Get service-specific health
service_health = await health_endpoint.get_service_health("database")
print(f"Database status: {service_health.status}")
```

### Metrics and Statistics

```python
# Get circuit breaker statistics
breaker_stats = circuit_manager.get_global_stats()
print(f"Open circuits: {breaker_stats['open_breakers']}")

# Get retry statistics
retry_stats = retry_manager.get_stats()
print(f"Cached decorators: {retry_stats['cached_decorators']}")

# Get fallback statistics
fallback_stats = fallback_manager.get_stats()
print(f"Total handlers: {fallback_stats['total_handlers']}")
```

## Error Handling

The resilience layer provides specific exceptions for different failure scenarios:

```python
from ai_agent.resilience.exceptions import (
    CircuitBreakerOpenException,
    RateLimitExceededException,
    HealthCheckFailedException,
    FallbackFailedException,
    RetryExhaustedException,
)

try:
    result = await resilient_service_call()
except CircuitBreakerOpenException as e:
    print(f"Circuit breaker is open: {e.service_name}")
except RateLimitExceededException as e:
    print(f"Rate limit exceeded: {e.retry_after}s")
except FallbackFailedException as e:
    print(f"All fallback strategies failed: {e.service_name}")
```

## Best Practices

1. **Configure appropriately**: Set retry limits and timeouts based on service characteristics
2. **Monitor health**: Use health checks to detect issues early
3. **Implement fallbacks**: Always have a fallback strategy for critical services
4. **Rate limit appropriately**: Set rate limits to protect both your service and external services
5. **Log comprehensively**: Use correlation IDs for tracing across resilience patterns
6. **Test failure scenarios**: Use chaos engineering to test resilience patterns

## Examples

See `examples/resilience_demo.py` for comprehensive examples of all resilience patterns in action.

## Architecture

The resilience layer follows these design principles:

- **Composable**: Each pattern can be used independently or combined
- **Configurable**: All patterns support environment-based configuration
- **Observable**: Comprehensive logging and metrics for monitoring
- **Testable**: Easy to mock and test individual components
- **Production-ready**: Built for enterprise-scale reliability
