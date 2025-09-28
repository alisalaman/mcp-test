#!/usr/bin/env python3
"""
Docker setup script for AI Agent application.

This script manages Docker containers for PostgreSQL and Redis services
required for Phase 2 infrastructure layer.
"""

import subprocess
import sys
import time
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


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        run_command(["docker", "--version"], capture=True)
        print_success("Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker is not available. Please install Docker Desktop")
        return False


def check_docker_compose_available() -> bool:
    """Check if Docker Compose is available."""
    try:
        run_command(["docker", "compose", "version"], capture=True)
        print_success("Docker Compose is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker Compose is not available. Please install Docker Desktop")
        return False


def start_services(environment: str = "development", with_tools: bool = False):
    """Start Docker services."""
    print_step(f"Starting services for {environment} environment")

    compose_files = ["-f", "docker-compose.yml"]

    if environment == "development":
        compose_files.extend(["-f", "docker-compose.dev.yml"])
    elif environment == "production":
        compose_files.extend(["-f", "docker-compose.prod.yml"])

    if with_tools:
        compose_files.extend(["--profile", "tools"])

    cmd = ["docker", "compose"] + compose_files + ["up", "-d"]

    try:
        run_command(cmd)
        print_success("Services started successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to start services")
        return False


def stop_services(environment: str = "development"):
    """Stop Docker services."""
    print_step(f"Stopping services for {environment} environment")

    compose_files = ["-f", "docker-compose.yml"]

    if environment == "development":
        compose_files.extend(["-f", "docker-compose.dev.yml"])
    elif environment == "production":
        compose_files.extend(["-f", "docker-compose.prod.yml"])

    cmd = ["docker", "compose"] + compose_files + ["down"]

    try:
        run_command(cmd)
        print_success("Services stopped successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to stop services")
        return False


def check_service_health(service_name: str, max_attempts: int = 30) -> bool:
    """Check if a service is healthy."""
    print_step(f"Checking health of {service_name}")

    for attempt in range(max_attempts):
        try:
            result = run_command(
                ["docker", "compose", "ps", "--format", "json", service_name],
                capture=True,
                check=False,
            )

            if result and '"Health"' in result and '"healthy"' in result:
                print_success(f"{service_name} is healthy")
                return True

            print(
                f"  Attempt {attempt + 1}/{max_attempts}: Waiting for {service_name}..."
            )
            time.sleep(2)

        except Exception:
            pass

    print_warning(f"{service_name} health check timed out")
    return False


def show_service_status():
    """Show status of all services."""
    print_step("Service Status")

    try:
        run_command(["docker", "compose", "ps"])
    except subprocess.CalledProcessError:
        print_error("Failed to get service status")


def show_service_logs(service_name: str, lines: int = 50):
    """Show logs for a service."""
    print_step(f"Showing logs for {service_name}")

    try:
        run_command(["docker", "compose", "logs", "--tail", str(lines), service_name])
    except subprocess.CalledProcessError:
        print_error(f"Failed to get logs for {service_name}")


def update_env_file(environment: str = "development"):
    """Update .env file with Docker service configuration."""
    print_step("Updating .env file for Docker services")

    env_file = Path(".env")
    if not env_file.exists():
        print_warning(".env file not found, creating from template")
        template_file = Path("env-templates/env.example")
        if template_file.exists():
            env_file.write_text(template_file.read_text())
        else:
            print_error("No .env template found")
            return False

    # Read current .env content
    content = env_file.read_text()

    # Update database settings
    if environment == "development":
        db_host = "localhost"
        db_port = "5433"  # Different port for dev
        redis_host = "localhost"
        redis_port = "6380"  # Different port for dev
    else:
        db_host = "localhost"
        db_port = "5432"
        redis_host = "localhost"
        redis_port = "6379"

    # Update or add database settings
    lines = content.split("\n")
    updated_lines = []
    db_updated = False
    redis_updated = False

    for line in lines:
        if line.startswith("DB_HOST="):
            updated_lines.append(f"DB_HOST={db_host}")
            db_updated = True
        elif line.startswith("DB_PORT="):
            updated_lines.append(f"DB_PORT={db_port}")
        elif line.startswith("REDIS_HOST="):
            updated_lines.append(f"REDIS_HOST={redis_host}")
            redis_updated = True
        elif line.startswith("REDIS_PORT="):
            updated_lines.append(f"REDIS_PORT={redis_port}")
        elif line.startswith("USE_DATABASE="):
            updated_lines.append("USE_DATABASE=true")
        elif line.startswith("USE_REDIS="):
            updated_lines.append("USE_REDIS=true")
        else:
            updated_lines.append(line)

    # Add missing settings if not found
    if not db_updated:
        updated_lines.append(f"DB_HOST={db_host}")
        updated_lines.append(f"DB_PORT={db_port}")
        updated_lines.append("DB_NAME=ai_agent")
        updated_lines.append("DB_USER=postgres")
        updated_lines.append("DB_PASSWORD=password")
        updated_lines.append("USE_DATABASE=true")

    if not redis_updated:
        updated_lines.append(f"REDIS_HOST={redis_host}")
        updated_lines.append(f"REDIS_PORT={redis_port}")
        updated_lines.append("REDIS_DB=0")
        updated_lines.append("USE_REDIS=true")

    # Write updated content
    env_file.write_text("\n".join(updated_lines))
    print_success(".env file updated for Docker services")


def main():
    """Main Docker setup function."""
    import argparse

    parser = argparse.ArgumentParser(description="Docker setup for AI Agent services")
    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "status", "logs", "setup"],
        help="Docker command to run",
    )
    parser.add_argument(
        "--environment",
        "-e",
        default="development",
        choices=["development", "production"],
        help="Environment to use (default: development)",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Include management tools (pgAdmin, Redis Commander)",
    )
    parser.add_argument("--service", "-s", help="Service name for logs command")
    parser.add_argument(
        "--lines", "-n", type=int, default=50, help="Number of log lines to show"
    )

    args = parser.parse_args()

    print(f"{Colors.BOLD}AI Agent Docker Setup{Colors.END}")
    print(f"Environment: {args.environment}")
    print()

    # Check prerequisites
    if not check_docker_available():
        sys.exit(1)

    if not check_docker_compose_available():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("docker-compose.yml").exists():
        print_error("Please run this script from the ai-agent-app directory")
        sys.exit(1)

    success = True

    try:
        if args.command == "start":
            success = start_services(args.environment, args.with_tools)
            if success:
                # Wait for services to be healthy
                check_service_health("postgres")
                check_service_health("redis")

                # Update .env file
                update_env_file(args.environment)

                print()
                print_success("Docker services are ready!")
                print()
                print("Services available:")
                print(
                    f"  PostgreSQL: localhost:{5433 if args.environment == 'development' else 5432}"
                )
                print(
                    f"  Redis: localhost:{6380 if args.environment == 'development' else 6379}"
                )

                if args.with_tools:
                    print("  pgAdmin: http://localhost:8080")
                    print("  Redis Commander: http://localhost:8081")

                print()
                print("Next steps:")
                print(
                    "  1. Run database migrations: python scripts/migrate_database.py migrate"
                )
                print("  2. Test the setup: python examples/phase2_demo.py")

        elif args.command == "stop":
            success = stop_services(args.environment)

        elif args.command == "restart":
            stop_services(args.environment)
            time.sleep(2)
            success = start_services(args.environment, args.with_tools)

        elif args.command == "status":
            show_service_status()

        elif args.command == "logs":
            if not args.service:
                print_error("Please specify a service with --service")
                sys.exit(1)
            show_service_logs(args.service, args.lines)

        elif args.command == "setup":
            # Full setup: start services and update configuration
            success = start_services(args.environment, args.with_tools)
            if success:
                check_service_health("postgres")
                check_service_health("redis")
                update_env_file(args.environment)

                print()
                print_success("Docker setup completed!")
                print("Run 'python examples/phase2_demo.py' to test everything")

        if not success:
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print_warning("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
