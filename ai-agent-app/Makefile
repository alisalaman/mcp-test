.PHONY: help install install-dev clean lint format type-check test test-unit test-integration test-e2e test-coverage run dev build docker-build docker-run migrate seed-data pre-commit-install pre-commit-run

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync --dev
	$(MAKE) pre-commit-install

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Code quality
lint: ## Run linting
	uv run ruff check src tests
	uv run ruff format --check src tests

format: ## Format code
	uv run black src tests
	uv run isort src tests
	uv run ruff format src tests

type-check: ## Run type checking
	uv run mypy src tests

quality: lint type-check ## Run all code quality checks

# Testing
test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration -v

test-e2e: ## Run end-to-end tests only
	uv run pytest tests/e2e -v

test-coverage: ## Run tests with coverage report
	uv run pytest --cov=src/ai_agent --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	uv run pytest --watch

# Development server
run: ## Run the application
	uv run uvicorn src.ai_agent.main:app --host 0.0.0.0 --port 8000

dev: ## Run the application in development mode
	uv run uvicorn src.ai_agent.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Build
build: ## Build the package
	uv build

# Docker
docker-build: ## Build Docker image
	docker build -t ai-agent-app -f docker/Dockerfile.prod .

docker-build-dev: ## Build development Docker image
	docker build -t ai-agent-app:dev -f docker/Dockerfile.dev .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env ai-agent-app

docker-dev: ## Run development Docker container
	docker-compose -f docker/docker-compose.yml up

# Database
migrate: ## Run database migrations
	uv run python scripts/migrate.py

seed-data: ## Seed database with sample data
	uv run python scripts/seed_data.py

# Development tools
pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

# Documentation
docs-build: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

# Security
security-check: ## Run security checks
	uv run bandit -r src/
	uv run safety check

# CI/CD helpers
ci-install: ## Install dependencies for CI
	uv sync --dev --frozen

ci-test: quality test-coverage ## Run all CI checks

ci-build: clean build ## Build for CI

# Health check
health: ## Check application health
	uv run python scripts/health_check.py

# Load testing
load-test: ## Run load tests
	uv run locust -f tests/load/locustfile.py --host=http://localhost:8000

# Chaos testing
chaos-test: ## Run chaos engineering tests
	uv run chaos run tests/resilience/chaos_experiments.json

# All checks (used in CI)
check-all: clean install-dev quality test-coverage ## Run all checks

# Quick development workflow
quick: format lint test-unit ## Quick development checks

# Full development workflow
full: clean install-dev format lint type-check test-coverage ## Full development workflow
