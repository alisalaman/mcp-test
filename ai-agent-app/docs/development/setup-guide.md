# Phase 2 Setup Guide

This guide walks you through setting up the Phase 2 Infrastructure Layer components.

## Using This Repository as a Template

This repository is designed to be used as a template for creating new AI agent backend projects. Here are the recommended approaches:

### Method 1: GitHub Template (Recommended)

If this repository is on GitHub:

1. **Make it a template repository** (if you own the repo):
   - Go to Settings → General
   - Check "Template repository"
   - Save changes

2. **Create new project**:
   - Click "Use this template" button
   - Choose "Create a new repository"
   - Name your new project
   - Clone the new repository

### Method 2: Manual Clone and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url> my-new-ai-project
   cd my-new-ai-project
   ```

2. **Clean up for new project**:
   ```bash
   # Remove old git history and initialize new repository
   rm -rf .git
   git init
   git add .
   git commit -m "Initial commit: AI agent backend template"

   # Add your remote repository
   git remote add origin <your-new-repo-url>
   git push -u origin main
   ```

3. **Customize project information**:
   - Update `README.md` with your project name and description
   - Modify `pyproject.toml` with your project metadata
   - Update environment files as needed
   - Customize API endpoints and business logic for your use case

### Method 3: Using GitHub CLI

```bash
# Create a new repository from template
gh repo create my-new-ai-project --template ai-agent-backend-starter

# Clone your new repository
git clone https://github.com/yourusername/my-new-ai-project
cd my-new-ai-project
```

### Template Customization Checklist

After creating your new project from this template:

- [ ] Update project name and description in README files
- [ ] Modify `pyproject.toml` with your project metadata
- [ ] Configure environment variables for your use case
- [ ] Update API endpoints to match your requirements
- [ ] Customize AI agent behaviors and workflows
- [ ] Configure external service integrations
- [ ] Set up your preferred database and caching solutions
- [ ] Update security settings and authentication
- [ ] Configure monitoring and observability
- [ ] Set up CI/CD pipelines for your deployment

## Prerequisites

- Python 3.13+ (required - check with `python3 --version`)
- UV package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Optional: PostgreSQL and Redis for full functionality

## 0. Python Environment Setup

### Install UV Package Manager

If you don't have UV installed:

```bash
# Install UV (fastest Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Create Virtual Environment

```bash
cd ai-agent-app

# Create virtual environment with Python 3.13+
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

## 1. Install Dependencies

### Core Dependencies (Required)

For Phase 2, you need the core dependencies:

```bash
# Install core dependencies (after activating venv)
uv sync

# Or with pip (not recommended)
pip install -e .
```

### Phase 2 Specific Dependencies (Optional)

For full Phase 2 functionality (PostgreSQL + Redis):

```bash
# Install Phase 2 specific dependencies
uv add asyncpg redis[hiredis] structlog
```

### All Dependencies (Recommended)

For complete functionality:

```bash
# Install all dependencies including future phases
uv sync --extra full --extra dev
```

## 2. Environment Configuration

### Create Environment File

Copy the environment template:

```bash
# Copy development environment template
cp env-templates/env.development .env

# Edit with your settings
nano .env  # or your preferred editor
```

### Basic Configuration

For development with in-memory storage (no external dependencies):

```bash
# .env (matches env-templates/env.development)
# Application Settings
ENVIRONMENT=development
DEBUG=true
APP_NAME="AI Agent Application"
HOST=0.0.0.0
PORT=8000

# Storage Settings
USE_MEMORY=true
USE_DATABASE=false
USE_REDIS=false

# Security Settings (auto-generated if not provided)
SECURITY_SECRET_KEY=dev-secret-key-change-in-production-32chars

# Observability Settings
OBSERVABILITY_LOG_LEVEL=DEBUG
OBSERVABILITY_TRACING_SAMPLE_RATE=1.0
FEATURE_ENABLE_DEBUG_ENDPOINTS=true

# Relaxed rate limits for development
RATE_LIMIT_API_DEFAULT_LIMIT=1000
RATE_LIMIT_LLM_REQUESTS_PER_MINUTE=100
```

**Note**: The `env.development` template is already configured for development with in-memory storage. For production or when using PostgreSQL/Redis, you'll need to add additional configuration variables.

### Environment Template Options

The project provides three environment templates:

| Template | Purpose | Storage | Use Case |
|----------|---------|---------|----------|
| `env.development` | Local development | In-memory only | Fast development, no external dependencies |
| `env.example` | Complete configuration | PostgreSQL + Redis | Full functionality with all features |
| `env.production` | Production deployment | PostgreSQL + Redis | Production environment |

**For development**: Use `env.development` (simple, in-memory)
**For full features**: Use `env.example` and configure database/Redis
**For production**: Use `env.production` with proper secrets

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

# Activate virtual environment
source .venv/bin/activate

# Install in development mode
uv sync
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
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set up Python environment
cd ai-agent-app
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv sync

# 4. Create basic environment (or copy the template)
cp env-templates/env.development .env

# 5. Test setup
python examples/phase2_demo.py
```

That's it! You now have a working Phase 2 infrastructure layer.

## Template Usage Quick Reference

### For New Projects from Template

If you're setting up a new project created from this template:

1. **Start with the template setup** (see "Using This Repository as a Template" section above)
2. **Follow the Quick Start Summary** below
3. **Customize for your specific use case** using the Template Customization Checklist

### Quick Start Summary

For the fastest setup to get Phase 2 working:

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set up Python environment
cd ai-agent-app
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv sync

# 4. Create basic environment (or copy the template)
cp env-templates/env.development .env

# 5. Test setup
python examples/phase2_demo.py
```

### Template Features Overview

This template provides:

✅ **Production-ready architecture** with layered design
✅ **Comprehensive error handling** and resilience patterns
✅ **Real-time capabilities** with WebSocket support
✅ **Multiple storage backends** (PostgreSQL, Redis, in-memory)
✅ **Observability** with logging, metrics, and tracing
✅ **Security** with authentication and authorization
✅ **Testing framework** with unit, integration, and e2e tests
✅ **Docker support** for easy deployment
✅ **API documentation** with automatic OpenAPI generation

### Next Steps After Template Setup

1. **Verify the demo works**: `python examples/phase2_demo.py`
2. **Customize the application** for your specific needs
3. **Set up external services** (PostgreSQL, Redis, LLM providers)
4. **Configure production settings** for deployment
5. **Add your business logic** and domain-specific features
