#!/usr/bin/env python3
"""
Phase 2 Setup Script

This script automates the setup process for Phase 2 Infrastructure Layer,
including dependency installation, environment configuration, and validation.
"""

import shutil
import subprocess
import sys
from pathlib import Path


# Colors for output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_step(message: str):
    """Print a step message."""
    print(f"{Colors.BLUE}{Colors.BOLD}==> {message}{Colors.END}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def run_command(
    cmd: list[str], check: bool = True, capture: bool = False
) -> str | None:
    """Run a command and return output if requested."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {' '.join(cmd)}")
            if capture and e.stderr:
                print_error(f"Error: {e.stderr.strip()}")
            raise
        return None


def check_python_version():
    """Check if Python version is 3.12+."""
    print_step("Checking Python version")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print_error(f"Python 3.12+ required, found {version.major}.{version.minor}")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_uv_available() -> bool:
    """Check if UV is available."""
    try:
        run_command(["uv", "--version"], capture=True)
        print_success("UV package manager available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("UV not available, will use pip")
        return False


def install_dependencies(use_uv: bool, full_install: bool = False):
    """Install project dependencies."""
    print_step("Installing dependencies")

    if use_uv:
        if full_install:
            run_command(["uv", "sync", "--extra", "full", "--extra", "dev"])
            print_success("Installed all dependencies with UV")
        else:
            run_command(["uv", "sync"])
            print_success("Installed core dependencies with UV")
    else:
        if full_install:
            run_command([sys.executable, "-m", "pip", "install", "-e", ".[full,dev]"])
            print_success("Installed all dependencies with pip")
        else:
            run_command([sys.executable, "-m", "pip", "install", "-e", "."])
            print_success("Installed core dependencies with pip")


def create_environment_file():
    """Create .env file from template."""
    print_step("Setting up environment configuration")

    env_file = Path(".env")
    template_file = Path("env-templates/env.example")

    if env_file.exists():
        print_warning(".env file already exists, skipping creation")
        return

    if not template_file.exists():
        print_warning("Environment template not found, creating basic .env")
        basic_env = """# AI Agent Application Configuration
ENVIRONMENT=development
DEBUG=true

# Storage Configuration (Phase 2)
USE_MEMORY=true
USE_DATABASE=false
USE_REDIS=false

# Security Configuration
SECURITY_SECRET_KEY=dev-secret-key-change-in-production-32chars
SECURITY_CORS_ORIGINS=["http://localhost:3000"]

# Observability
OBSERVABILITY_LOG_LEVEL=DEBUG
OBSERVABILITY_TRACING_SAMPLE_RATE=1.0

# Feature Flags
FEATURE_ENABLE_DEBUG_ENDPOINTS=true
FEATURE_ENABLE_CIRCUIT_BREAKERS=true
FEATURE_ENABLE_WEBSOCKETS=true
"""
        env_file.write_text(basic_env)
    else:
        shutil.copy(template_file, env_file)

    print_success("Created .env file")


def check_optional_services():
    """Check for optional services (PostgreSQL, Redis)."""
    print_step("Checking optional services")

    services = {}

    # Check Docker
    docker_available = False
    try:
        run_command(["docker", "--version"], capture=True, check=False)
        docker_available = True
        print_success("Docker available")
    except FileNotFoundError:
        print_warning(
            "Docker not found - install Docker Desktop for easy service management"
        )

    # Check PostgreSQL (local installation)
    try:
        result = run_command(["psql", "--version"], capture=True, check=False)
        if result:
            services["postgresql_local"] = True
            print_success("PostgreSQL (local) available")
        else:
            services["postgresql_local"] = False
    except FileNotFoundError:
        services["postgresql_local"] = False

    # Check Redis (local installation)
    try:
        result = run_command(["redis-cli", "--version"], capture=True, check=False)
        if result:
            services["redis_local"] = True
            print_success("Redis (local) available")
        else:
            services["redis_local"] = False
    except FileNotFoundError:
        services["redis_local"] = False

    # Check Docker services
    if docker_available:
        try:
            result = run_command(
                ["docker", "compose", "ps", "--format", "json"],
                capture=True,
                check=False,
            )
            if result and "postgres" in result:
                services["postgresql_docker"] = True
                print_success("PostgreSQL (Docker) available")
            else:
                services["postgresql_docker"] = False
        except Exception:
            services["postgresql_docker"] = False

        try:
            result = run_command(
                ["docker", "compose", "ps", "--format", "json"],
                capture=True,
                check=False,
            )
            if result and "redis" in result:
                services["redis_docker"] = True
                print_success("Redis (Docker) available")
            else:
                services["redis_docker"] = False
        except Exception:
            services["redis_docker"] = False

    # Provide recommendations
    if not services["postgresql_local"] and not services["postgresql_docker"]:
        if docker_available:
            print_warning(
                "PostgreSQL not found - start with: python scripts/setup_docker.py setup"
            )
        else:
            print_warning(
                "PostgreSQL not found - install with: brew install postgresql@15 or use Docker"
            )

    if not services["redis_local"] and not services["redis_docker"]:
        if docker_available:
            print_warning(
                "Redis not found - start with: python scripts/setup_docker.py setup"
            )
        else:
            print_warning(
                "Redis not found - install with: brew install redis or use Docker"
            )

    return services


def test_configuration():
    """Test the configuration system."""
    print_step("Testing configuration system")

    try:
        # Add current directory to Python path
        sys.path.insert(0, str(Path("src").absolute()))

        from ai_agent.config.settings import ConfigurationValidator, get_settings

        settings = get_settings()
        print_success(f"Configuration loaded for environment: {settings.environment}")

        # Validate configuration
        errors = ConfigurationValidator.validate_settings(settings)
        if errors:
            print_warning("Configuration validation warnings:")
            for error in errors:
                print_warning(f"  - {error}")
        else:
            print_success("Configuration validation passed")

        return True

    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False


def test_repository():
    """Test the repository system."""
    print_step("Testing repository system")

    try:
        import asyncio

        # Ensure we can import the modules
        sys.path.insert(0, str(Path("src").absolute()))

        # Import after path adjustment
        from ai_agent.config.settings import get_settings
        from ai_agent.infrastructure.database import (
            cleanup_repository,
            setup_repository,
        )

        async def test_repo():
            settings = get_settings()

            # Determine backend
            if settings.use_database:
                backend = "PostgreSQL"
            elif settings.use_redis:
                backend = "Redis"
            else:
                backend = "In-Memory"

            print_success(f"Using {backend} repository backend")

            try:
                repo = await setup_repository(settings)
                healthy = await repo.health_check()

                if healthy:
                    print_success("Repository health check passed")
                else:
                    print_warning("Repository health check failed")

                await cleanup_repository()
                return healthy

            except Exception as e:
                print_error(f"Repository test failed: {e}")
                return False

        return asyncio.run(test_repo())

    except Exception as e:
        print_error(f"Repository test setup failed: {e}")
        return False


def run_demo():
    """Run the Phase 2 demo."""
    print_step("Running Phase 2 demo")

    try:
        run_command([sys.executable, "examples/phase2_demo.py"])
        print_success("Phase 2 demo completed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Phase 2 demo failed")
        return False


def main():
    """Main setup function."""
    print(f"{Colors.BOLD}AI Agent Application - Phase 2 Setup{Colors.END}")
    print("This script will set up the Phase 2 Infrastructure Layer")
    print()

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print_error("Please run this script from the ai-agent-app directory")
        sys.exit(1)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 setup script")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install all dependencies (including future phases)",
    )
    parser.add_argument(
        "--skip-demo", action="store_true", help="Skip running the demo"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip configuration and repository tests",
    )

    args = parser.parse_args()

    success = True

    try:
        # Step 1: Check Python version
        if not check_python_version():
            sys.exit(1)

        # Step 2: Check UV availability
        use_uv = check_uv_available()

        # Step 3: Install dependencies
        install_dependencies(use_uv, args.full)

        # Step 4: Create environment file
        create_environment_file()

        # Step 5: Check optional services
        services = check_optional_services()

        # Step 6: Test configuration (if not skipped)
        if not args.skip_tests:
            if not test_configuration():
                success = False

            # Step 7: Test repository (if not skipped)
            if not test_repository():
                success = False

        # Step 8: Run demo (if not skipped)
        if not args.skip_demo and success:
            if not run_demo():
                success = False

        # Summary
        print()
        print(f"{Colors.BOLD}Setup Summary{Colors.END}")

        if success:
            print_success("Phase 2 setup completed successfully!")
            print()
            print("Next steps:")
            print("  1. Review the .env file and adjust settings as needed")
            if not services["postgresql"]:
                print("  2. Optional: Install PostgreSQL for persistent storage")
            if not services["redis"]:
                print("  3. Optional: Install Redis for session caching")
            print("  4. Run 'python examples/phase2_demo.py' to test functionality")
            print("  5. Ready for Phase 3 implementation!")
        else:
            print_error("Setup completed with some issues")
            print("Please review the errors above and fix any issues")
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print_warning("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
