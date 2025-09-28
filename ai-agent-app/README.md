# AI Agent Application

A production-ready AI agent application built with Python 3.12+, LangGraph, MCP servers, and FastAPI. The system is designed for enterprise scalability, resilience, and maintainability with sophisticated external service integration patterns.

## 🚀 Features

- **Enterprise-Ready**: Built for production with comprehensive error handling, monitoring, and observability
- **Type-Safe**: Strong typing with Pydantic v2 models throughout the application
- **Resilient**: Built-in retry mechanisms, circuit breakers, and graceful degradation
- **Real-Time**: WebSocket support with event-driven architecture
- **Configurable**: Runtime behavior controlled via environment-specific configurations
- **Observable**: Comprehensive logging, metrics, and distributed tracing
- **API Complete**: Full CRUD operations with automatic OpenAPI documentation

## 🏗️ Architecture

The application follows a layered architecture with clear separation of concerns:

- **Presentation Layer**: REST API v1 with FastAPI
- **Application Layer**: Core business logic and workflow orchestration
- **Domain Layer**: Business entities and rules
- **Infrastructure Layer**: External service integrations
- **Resilience Layer**: Circuit breakers, retries, and failure handling
- **Observability Layer**: Logging, monitoring, and metrics

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Web Framework** | FastAPI | 0.104+ | High performance, automatic OpenAPI docs |
| **Python Version** | Python | 3.12+ | Latest language features, improved performance |
| **Package Manager** | UV | 0.1.35+ | Fastest Python package installer |
| **Workflow Engine** | LangGraph | 0.0.40+ | Graph-based workflow orchestration |
| **Validation** | Pydantic | 2.5+ | Runtime type validation, settings management |
| **ASGI Server** | Uvicorn | 0.24+ | High-performance ASGI implementation |

### External Services Support

- **LLM Providers**: OpenAI, Anthropic, Google Generative AI
- **Databases**: PostgreSQL (with asyncpg), Redis
- **Secret Management**: AWS Secrets Manager, Azure Key Vault, GCP Secret Manager
- **Monitoring**: Prometheus, OpenTelemetry, Structured logging

## 📦 Installation

### Prerequisites

- Python 3.12+
- UV package manager
- Docker Desktop (recommended for PostgreSQL and Redis)
- PostgreSQL (optional, for persistent storage)
- Redis (optional, for caching)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-agent-app
   ```

2. **Install dependencies**:
   ```bash
   make install-dev
   ```

3. **Set up services (Docker recommended)**:
   ```bash
   # Start PostgreSQL and Redis with Docker
   python scripts/setup_docker.py setup

   # Or set up manually
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application**:
   ```bash
   make dev
   ```

The application will be available at `http://localhost:8000`.

## 🔧 Development

### Available Make Commands

```bash
# Environment setup
make install          # Install production dependencies
make install-dev      # Install development dependencies
make clean           # Clean up build artifacts

# Code quality
make lint            # Run linting
make format          # Format code
make type-check      # Run type checking
make quality         # Run all code quality checks

# Testing
make test            # Run all tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests only
make test-e2e        # Run end-to-end tests only
make test-coverage   # Run tests with coverage report

# Development server
make run             # Run the application
make dev             # Run in development mode with reload

# Database
make migrate         # Run database migrations
make seed-data       # Seed database with sample data

# Documentation
make docs-build      # Build documentation
make docs-serve      # Serve documentation locally
```

### Project Structure

```
ai-agent-app/
├── src/ai_agent/           # Main application code
│   ├── api/               # REST API layer
│   ├── core/              # Core business logic
│   ├── domain/            # Domain models and exceptions
│   ├── infrastructure/    # External integrations
│   ├── resilience/        # Resilience patterns
│   ├── observability/     # Monitoring and logging
│   └── config/            # Configuration management
├── tests/                 # Test suite
├── docker/                # Docker configurations
├── k8s/                   # Kubernetes manifests
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## ⚙️ Configuration

The application supports multiple environments with environment-specific configuration:

- **Development**: Memory-based storage, debug logging, relaxed rate limits
- **Production**: Database storage, optimized logging, strict rate limits
- **Testing**: Isolated test environment with mocked services

### Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# Application
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Database
DB_HOST=localhost
DB_NAME=ai_agent
DB_USER=postgres
DB_PASSWORD=password

# LLM Providers
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Security
SECURITY_SECRET_KEY=your-32-character-secret-key
```

## 🔄 API Endpoints

The application provides RESTful APIs for:

- **Sessions**: Create and manage conversation sessions
- **Messages**: Send and retrieve messages
- **Agents**: Configure and manage AI agents
- **Tools**: Register and manage agent tools
- **MCP Servers**: Configure MCP server connections
- **Health**: Application health and readiness checks

API documentation is automatically generated and available at `/docs` when running the application.

## 🛡️ Security

- **Authentication**: JWT-based authentication with configurable expiration
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive validation with Pydantic models
- **Secret Management**: Support for external secret managers
- **CORS**: Configurable CORS policies

## 📊 Monitoring & Observability

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Metrics**: Prometheus-compatible metrics collection
- **Distributed Tracing**: OpenTelemetry integration
- **Health Checks**: Comprehensive health and readiness endpoints
- **Error Tracking**: Structured error reporting with context

## 🚢 Deployment

### Docker

```bash
# Start services with Docker
python scripts/setup_docker.py setup

# Build production image
make docker-build

# Run with Docker Compose
make docker-dev
```

See [Docker Setup Guide](docs/deployment/docker-setup.md) for detailed Docker configuration.

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 🧪 Testing

The application includes comprehensive testing:

- **Unit Tests**: Fast, isolated tests for business logic
- **Integration Tests**: Tests with real external services
- **End-to-End Tests**: Full workflow testing
- **Resilience Tests**: Chaos engineering and failure injection
- **Load Tests**: Performance and scalability testing

## 🤝 Contributing

1. Follow the development workflow: `make full`
2. Ensure all tests pass: `make test`
3. Maintain code coverage above 80%
4. Follow the established code style and architecture patterns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:

1. Check the documentation in the `docs/` directory:
   - [Development Setup Guide](docs/development/setup-guide.md) for getting started
   - [Docker Setup Guide](docs/deployment/docker-setup.md) for containerized services
   - [Architecture Documentation](docs/architecture/) for technical details
2. Review existing issues and discussions
3. Create a new issue with detailed information
4. Follow the contributing guidelines

---

Built with ❤️ for production-ready AI agent applications.
