#!/usr/bin/env python3
"""Development environment setup script."""

import subprocess
import sys

from .colors import (
    command,
    dim,
    error,
    highlight,
    info,
    print_error,
    print_info,
    print_section,
    print_step,
    print_success,
    print_warning,
)


def check_environment() -> dict[str, str]:
    """Check the current development environment."""
    env_info = {}

    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        env_info["venv"] = "not_active"
    else:
        env_info["venv"] = "active"

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    env_info["python"] = python_version

    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        env_info["git"] = "available"
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info["git"] = "not_available"

    return env_info


def run_command(
    command_list: list[str], description: str, step: int, total: int
) -> bool:
    """Run a command and return True if successful."""
    print_step(step, total, f"Running {description}")
    print(dim(f"Command: {' '.join(command_list)}"))

    try:
        result = subprocess.run(
            command_list, check=True, capture_output=True, text=True
        )
        print_success(f"{description} completed")

        # Show relevant output
        if result.stdout.strip():
            # Show only the most relevant lines
            lines = result.stdout.strip().split("\n")
            relevant_lines = [
                line
                for line in lines
                if any(
                    keyword in line.lower()
                    for keyword in ["installed", "requirement", "successfully"]
                )
            ]
            if relevant_lines:
                print(dim("Output:"))
                for line in relevant_lines[:5]:  # Show first 5 relevant lines
                    print(dim(f"  {line}"))

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed (exit code {e.returncode})")

        if e.stdout.strip():
            print(error("STDOUT:"))
            print(e.stdout.strip())

        if e.stderr.strip():
            print(error("STDERR:"))
            print(e.stderr.strip())

        return False


def main() -> None:
    """Set up development environment."""
    print_section("🚀 DEVELOPMENT ENVIRONMENT SETUP")

    # Check current environment
    env_info = check_environment()

    print_info("Environment check:")
    print(f"  • Python version: {info(env_info['python'])}")
    print(f"  • Virtual environment: {info(env_info['venv'])}")
    print(f"  • Git: {info(env_info['git'])}")

    # Show warnings if needed
    if env_info["venv"] == "not_active":
        print_warning(
            "Not in a virtual environment. Consider using 'uv venv' or 'python -m venv'"
        )

    if env_info["git"] == "not_available":
        print_warning("Git not found. Pre-commit hooks will not be available")

    print()

    # Setup steps
    setup_steps = [
        (["pip", "install", "-e", ".[dev]"], "Installing package in development mode"),
    ]

    # Only add pre-commit if git is available
    if env_info["git"] == "available":
        setup_steps.append((["pre-commit", "install"], "Installing pre-commit hooks"))

    completed_steps = 0
    total_steps = len(setup_steps)

    for i, (command_list, description) in enumerate(setup_steps, 1):
        if run_command(command_list, description, i, total_steps):
            completed_steps += 1
        print()  # Add spacing between steps

    # Final summary
    print_section("📊 SETUP SUMMARY")

    if completed_steps == total_steps:
        print_success(f"All {total_steps} setup steps completed! 🎉")
        print(highlight("Development environment is ready! ✨"))

        print("\n" + command("📋 Available commands:"))
        commands = [
            ("ai-agent", "Start the application"),
            ("ai-agent-dev", "Start in development mode"),
            ("lint", "Run all linting checks"),
            ("format", "Format code with black, isort, and ruff"),
            ("test", "Run tests"),
            ("test --cov", "Run tests with coverage"),
            ("test --unit", "Run only unit tests"),
            ("test --integration", "Run only integration tests"),
            ("test --e2e", "Run only end-to-end tests"),
        ]

        for cmd, desc in commands:
            print(f"  • {highlight(cmd):<20} - {dim(desc)}")

        print("\n" + command("💡 Next steps:"))
        print("  • " + dim("ai-agent-dev") + " - Start the development server")
        print("  • " + dim("test") + " - Run the test suite")
        print("  • " + dim("format && lint") + " - Format and check code quality")

    else:
        failed_steps = total_steps - completed_steps
        print_error(f"{failed_steps}/{total_steps} setup steps failed")

        if completed_steps > 0:
            print_warning(f"However, {completed_steps} steps completed successfully")

        print("\n" + command("💡 Troubleshooting:"))
        print("  • Ensure you have Python 3.12+ installed")
        print("  • Check that pip is up to date: pip install --upgrade pip")
        print("  • Try running setup again after fixing any errors")

        sys.exit(1)


if __name__ == "__main__":
    main()
