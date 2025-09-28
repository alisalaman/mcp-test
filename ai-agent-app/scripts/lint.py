#!/usr/bin/env python3
"""Linting script for the AI Agent application."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import subprocess

from colors import (
    command,
    dim,
    error,
    highlight,
    print_error,
    print_section,
    print_step,
    print_success,
    print_warning,
)


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
        print_success(f"{description} passed")

        # Show output if there are warnings or info messages
        if result.stdout.strip():
            print(dim("Output:"))
            print(dim(result.stdout.strip()))

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed")

        if e.stdout.strip():
            print(error("STDOUT:"))
            print(e.stdout.strip())

        if e.stderr.strip():
            print(error("STDERR:"))
            print(e.stderr.strip())

        return False


def main() -> None:
    """Run all linting checks."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    tests_path = project_root / "tests"
    scripts_path = project_root / "scripts"

    print_section("🚀 LINTING CHECKS")

    checks = [
        (
            ["ruff", "check", str(src_path), str(tests_path), str(scripts_path)],
            "Ruff linting",
        ),
        (
            ["black", "--check", str(src_path), str(tests_path), str(scripts_path)],
            "Black formatting check",
        ),
        (["mypy", str(src_path)], "Type checking"),
    ]

    passed_checks = 0
    total_checks = len(checks)

    for i, (command_list, description) in enumerate(checks, 1):
        if run_command(command_list, description, i, total_checks):
            passed_checks += 1
        print()  # Add spacing between checks

    # Final summary
    print_section("📊 SUMMARY")

    if passed_checks == total_checks:
        print_success(f"All {total_checks} linting checks passed! 🎉")
        print(highlight("Your code is ready for commit! 🚀"))
        sys.exit(0)
    else:
        failed_checks = total_checks - passed_checks
        print_error(f"{failed_checks}/{total_checks} linting checks failed")
        print_warning("Run 'format' to automatically fix formatting issues")

        # Provide helpful suggestions
        print("\n" + command("💡 Suggested fixes:"))
        print("  • " + dim("format") + " - Auto-fix formatting issues")
        print("  • " + dim("ruff check --fix .") + " - Fix auto-fixable linting issues")
        print("  • " + dim("mypy src/") + " - Check specific type errors")

        sys.exit(1)


if __name__ == "__main__":
    main()
