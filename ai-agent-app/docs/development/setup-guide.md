# Phase 2 Setup Guide

This guide walks you through setting up the Phase 2 Infrastructure Layer components.

## Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- Optional: PostgreSQL and Redis for full functionality

## 1. Install Dependencies

### Core Dependencies (Required)

For Phase 2, you need the core dependencies:

```bash
cd ai-agent-app

# Install core dependencies
uv sync

# Or with pip
pip install -e .
```

### Phase 2 Specific Dependencies (Optional)

For full Phase 2 functionality (PostgreSQL + Redis):

```bash
# Install Phase 2 specific dependencies
uv add asyncpg redis[hiredis] structlog

# Or with pip
pip install asyncpg redis[hiredis] structlog
```

### All Dependencies (Recommended)

For complete functionality:

```bash
# Install all dependencies including future phases
uv sync --extra full --extra dev

# Or with pip
pip install -e ".[full,dev]"
```

## 2. Environment Configuration

### Create Environment File

Copy the environment template:

```bash
# Copy environment template
cp env-templates/env.example .env

# Edit with your settings
nano .env  # or your preferred editor
```

### Basic Configuration

For development with in-memory storage (no external dependencies):

```bash
# .env
ENVIRONMENT=development
DEBUG=true
USE_MEMORY=true
USE_DATABASE=false
USE_REDIS=false

# Security (change this!)
SECURITY_SECRET_KEY=dev-secret-key-change-in-production-32chars
```

### Docker Setup (Recommended)

For PostgreSQL and Redis with Docker:

```bash
# Start services with Docker
python scripts/setup_docker.py setup

# Or with management tools (pgAdmin, Redis Commander)
python scripts/setup_docker.py setup --with-tools

# Check service status
python scripts/setup_docker.py status

# The script automatically updates .env with Docker settings
```

### Manual Docker Commands

```bash
# Start services
docker compose up -d

# Start with management tools
docker compose --profile tools up -d

# Check status
docker compose ps

# View logs
docker compose logs postgres
docker compose logs redis

# Stop services
docker compose down
```

### Manual Installation (Alternative)

If you prefer to install services directly:

```bash
# PostgreSQL (macOS with Homebrew)
brew install postgresql@15
brew services start postgresql@15
createdb ai_agent

# Redis (macOS with Homebrew)
brew install redis
brew services start redis

# Update .env manually
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_agent
DB_USER=your_username
DB_PASSWORD=your_password
USE_DATABASE=true

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
USE_REDIS=true
```

## 3. Database Setup (If Using PostgreSQL)

### Run Migrations

```bash
# Check database connection
python scripts/migrate_database.py check

# Run migrations to create schema
python scripts/migrate_database.py migrate

# Optional: Reset database (destructive!)
python scripts/migrate_database.py reset
```

## 4. Verify Setup

### Run Phase 2 Demo

Test your setup with the comprehensive demo:

```bash
python examples/phase2_demo.py
```

Expected output:
```
AI Agent Application - Phase 2 Infrastructure Layer Demo
========================================
PHASE 2.1: Configuration Management System Demo
========================================
Environment: development
Debug mode: True
...
✓ Configuration is valid

========================================
PHASE 2.2: Repository Pattern Demo
========================================
Using In-Memory repository
✓ Repository connected successfully
✓ Repository health check: passed
...
🎉 Phase 2 Infrastructure Layer Demo completed successfully!
```

### Test Individual Components

#### Configuration System
```python
from ai_agent.config.settings import get_settings

settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Storage: DB={settings.use_database}, Redis={settings.use_redis}, Memory={settings.use_memory}")
```

#### Repository System
```python
import asyncio
from ai_agent.infrastructure.database import setup_repository
from ai_agent.config.settings import get_settings

async def test_repo():
    settings = get_settings()
    repo = await setup_repository(settings)
    healthy = await repo.health_check()
    print(f"Repository healthy: {healthy}")

asyncio.run(test_repo())
```

## 5. Configuration Options

### Storage Backend Selection

The system automatically selects storage backend based on configuration:

| Priority | Backend | Configuration | Use Case |
|----------|---------|---------------|----------|
| 1 | PostgreSQL | `USE_DATABASE=true` | Production persistence |
| 2 | Redis | `USE_REDIS=true` | Session caching |
| 3 | In-Memory | Default fallback | Development/testing |

### Environment-Specific Settings

| Environment | File | Description |
|-------------|------|-------------|
| `development` | `env-templates/env.development` | Local development |
| `testing` | Built-in | Automated testing |
| `staging` | `env-templates/env.production` | Pre-production |
| `production` | `env-templates/env.production` | Production deployment |

## 6. Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the right directory
cd ai-agent-app

# Install in development mode
uv sync
# or
pip install -e .
```

#### Database Connection Issues
```bash
# Check PostgreSQL is running
brew services list | grep postgresql

# Test connection manually
psql -h localhost -U your_username -d ai_agent

# Check configuration
python -c "from ai_agent.config.settings import get_settings; print(get_settings().database.url)"
```

#### Redis Connection Issues
```bash
# Check Redis is running
brew services list | grep redis

# Test connection manually
redis-cli ping

# Check configuration
python -c "from ai_agent.config.settings import get_settings; print(get_settings().redis.url)"
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/migrate_database.py
chmod +x examples/phase2_demo.py
```

### Validation Errors

The configuration system validates settings on startup:

```python
from ai_agent.config.settings import get_settings, ConfigurationValidator

settings = get_settings()
errors = ConfigurationValidator.validate_settings(settings)
if errors:
    for error in errors:
        print(f"❌ {error}")
```

## 7. Development Workflow

### Recommended Setup for Development

1. **Use in-memory storage** for fast iteration:
   ```bash
   ENVIRONMENT=development
   USE_MEMORY=true
   ```

2. **Enable debug features**:
   ```bash
   DEBUG=true
   FEATURE_ENABLE_DEBUG_ENDPOINTS=true
   OBSERVABILITY_LOG_LEVEL=DEBUG
   ```

3. **Use PostgreSQL for integration testing**:
   ```bash
   ENVIRONMENT=testing
   USE_DATABASE=true
   ```

### Testing Different Backends

You can test different storage backends by changing environment variables:

```bash
# Test in-memory (fastest)
ENVIRONMENT=development python examples/phase2_demo.py

# Test with Redis
USE_REDIS=true ENVIRONMENT=development python examples/phase2_demo.py

# Test with PostgreSQL
USE_DATABASE=true ENVIRONMENT=development python examples/phase2_demo.py
```

## 8. Next Steps

Once Phase 2 is set up and working:

1. **Verify all tests pass**: Run the demo script successfully
2. **Check configuration validation**: No validation errors
3. **Test storage backends**: At least in-memory working
4. **Ready for Phase 3**: Resilience layer implementation

### Optional Enhancements

- Set up PostgreSQL for persistent storage
- Configure Redis for session caching
- Set up monitoring with structured logging
- Configure feature flags for different environments

## Quick Start Summary

For the fastest setup to get Phase 2 working:

```bash
# 1. Install dependencies
cd ai-agent-app
uv sync

# 2. Create basic environment
echo "ENVIRONMENT=development
DEBUG=true
USE_MEMORY=true
SECURITY_SECRET_KEY=dev-secret-key-change-in-production-32chars" > .env

# 3. Test setup
python examples/phase2_demo.py
```

That's it! You now have a working Phase 2 infrastructure layer.
