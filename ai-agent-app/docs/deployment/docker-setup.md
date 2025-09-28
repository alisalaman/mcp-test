# Docker Setup for AI Agent Application

This guide explains how to use Docker containers for PostgreSQL and Redis services in the AI Agent application.

## Quick Start

```bash
# Start all services
python scripts/setup_docker.py setup

# Start with management tools (pgAdmin, Redis Commander)
python scripts/setup_docker.py setup --with-tools

# Check service status
python scripts/setup_docker.py status

# Stop services
python scripts/setup_docker.py stop
```

## Services Included

### Core Services
- **PostgreSQL 15** - Primary database for persistent storage
- **Redis 7** - Session state and caching

### Management Tools (Optional)
- **pgAdmin** - PostgreSQL database management (http://localhost:8080)
- **Redis Commander** - Redis management interface (http://localhost:8081)

## Docker Compose Files

### `docker-compose.yml` (Base)
- Default configuration for all environments
- PostgreSQL on port 5432
- Redis on port 6379

### `docker-compose.dev.yml` (Development)
- PostgreSQL on port 5433 (avoids conflicts)
- Redis on port 6380 (avoids conflicts)
- Development-optimized settings

### `docker-compose.prod.yml` (Production)
- No exposed ports (security)
- Production-optimized PostgreSQL settings
- Memory limits and persistence policies

## Usage Examples

### Development Environment
```bash
# Start development services
python scripts/setup_docker.py setup --environment development

# Or manually
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Environment
```bash
# Start production services
python scripts/setup_docker.py setup --environment production

# Or manually
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### With Management Tools
```bash
# Start with pgAdmin and Redis Commander
python scripts/setup_docker.py setup --with-tools

# Or manually
docker compose --profile tools up -d
```

## Service Management

### Check Status
```bash
# Using script
python scripts/setup_docker.py status

# Or manually
docker compose ps
```

### View Logs
```bash
# Using script
python scripts/setup_docker.py logs postgres
python scripts/setup_docker.py logs redis

# Or manually
docker compose logs postgres
docker compose logs redis
```

### Restart Services
```bash
# Using script
python scripts/setup_docker.py restart

# Or manually
docker compose restart
```

### Stop Services
```bash
# Using script
python scripts/setup_docker.py stop

# Or manually
docker compose down
```

## Database Migrations

After starting PostgreSQL:

```bash
# Run migrations
python scripts/migrate_database.py migrate

# Check database connection
python scripts/migrate_database.py check
```

## Configuration

The Docker setup script automatically updates your `.env` file with the correct service endpoints:

### Development
```bash
DB_HOST=localhost
DB_PORT=5433
DB_NAME=ai_agent_dev
DB_USER=postgres
DB_PASSWORD=dev_password
USE_DATABASE=true

REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0
USE_REDIS=true
```

### Production
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_agent_prod
DB_USER=ai_agent_user
DB_PASSWORD=secure_password_change_me
USE_DATABASE=true

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
USE_REDIS=true
```

## Data Persistence

### Volumes
- `postgres_data` - PostgreSQL data directory
- `redis_data` - Redis data directory

### Backup
```bash
# Backup PostgreSQL
docker compose exec postgres pg_dump -U postgres ai_agent > backup.sql

# Backup Redis
docker compose exec redis redis-cli --rdb /data/dump.rdb
```

### Restore
```bash
# Restore PostgreSQL
docker compose exec -T postgres psql -U postgres ai_agent < backup.sql

# Restore Redis
docker compose exec redis redis-cli --rdb /data/dump.rdb
```

## Troubleshooting

### Services Won't Start
```bash
# Check Docker is running
docker --version
docker compose version

# Check for port conflicts
lsof -i :5432
lsof -i :6379

# View detailed logs
docker compose logs --tail 100 postgres
docker compose logs --tail 100 redis
```

### Connection Issues
```bash
# Test PostgreSQL connection
docker compose exec postgres psql -U postgres -d ai_agent -c "SELECT 1;"

# Test Redis connection
docker compose exec redis redis-cli ping
```

### Reset Everything
```bash
# Stop and remove all containers and volumes
docker compose down -v

# Remove all data (DESTRUCTIVE!)
docker volume prune -f

# Start fresh
python scripts/setup_docker.py setup
```

## Security Notes

### Development
- Ports are exposed for easy access
- Default passwords are used
- Management tools are available

### Production
- No ports exposed externally
- Strong passwords required
- No management tools by default
- Consider using Docker secrets for sensitive data

## Performance Tuning

### PostgreSQL
The production configuration includes:
- `shared_buffers=256MB`
- `effective_cache_size=1GB`
- `max_connections=200`
- `checkpoint_completion_target=0.9`

### Redis
The production configuration includes:
- `maxmemory=512mb`
- `maxmemory-policy=allkeys-lru`
- Automatic persistence with `save` directives

## Integration with Phase 2

The Docker setup integrates seamlessly with the Phase 2 infrastructure:

1. **Automatic Detection**: The repository factory detects Docker services
2. **Configuration Updates**: `.env` is automatically updated
3. **Health Checks**: Built-in health monitoring
4. **Migration Support**: Database migrations work automatically

## Next Steps

After Docker setup:
1. Run `python examples/phase2_demo.py` to test everything
2. Use `python scripts/migrate_database.py migrate` for database setup
3. Ready for Phase 3 development with persistent storage!
