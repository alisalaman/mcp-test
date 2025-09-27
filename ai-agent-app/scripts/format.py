#!/usr/bin/env python3
"""Code formatting script for the AI Agent application."""

import subprocess
import sys
from pathlib import Path

from .colors import (
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
        print_success(f"{description} completed")

        # Show what was changed if there's output
        if result.stdout.strip():
            print(dim("Changes made:"))
            lines = result.stdout.strip().split("\n")
            for line in lines[:10]:  # Show first 10 lines
                print(dim(f"  {line}"))
            if len(lines) > 10:
                print(dim(f"  ... and {len(lines) - 10} more changes"))

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
    """Format code using black, isort, and ruff."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    tests_path = project_root / "tests"
    scripts_path = project_root / "scripts"

    print_section("🎨 CODE FORMATTING")

    formatters = [
        (
            [
                "ruff",
                "check",
                "--fix",
                str(src_path),
                str(tests_path),
                str(scripts_path),
            ],
            "Ruff auto-fixing",
        ),
        (
            ["black", str(src_path), str(tests_path), str(scripts_path)],
            "Black formatting",
        ),
        (
            ["isort", str(src_path), str(tests_path), str(scripts_path)],
            "Import sorting",
        ),
    ]

    completed_steps = 0
    total_steps = len(formatters)

    for i, (command_list, description) in enumerate(formatters, 1):
        if run_command(command_list, description, i, total_steps):
            completed_steps += 1
        print()  # Add spacing between steps

    # Final summary
    print_section("📊 SUMMARY")

    if completed_steps == total_steps:
        print_success(f"All {total_steps} formatting steps completed! 🎉")
        print(highlight("Your code is now beautifully formatted! ✨"))

        # Suggest next steps
        print("\n" + command("💡 Next steps:"))
        print("  • " + dim("lint") + " - Run linting checks")
        print("  • " + dim("test") + " - Run tests")
        print("  • " + dim("git add .") + " - Stage your changes")

        sys.exit(0)
    else:
        failed_steps = total_steps - completed_steps
        print_error(f"{failed_steps}/{total_steps} formatting steps failed")

        if completed_steps > 0:
            print_warning(f"However, {completed_steps} steps completed successfully")

        # Suggest manual intervention
        print("\n" + command("💡 Manual intervention may be required:"))
        print("  • Check the error messages above")
        print("  • Ensure all dependencies are installed")
        print("  • Fix any syntax errors in your code")

        sys.exit(1)


if __name__ == "__main__":
    main()
