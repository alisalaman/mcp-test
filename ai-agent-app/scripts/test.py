#!/usr/bin/env python3
"""Test runner script for the AI Agent application."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import subprocess

from colors import (
    command,
    dim,
    highlight,
    info,
    print_error,
    print_info,
    print_section,
    print_success,
    print_warning,
)


def parse_args() -> tuple[list[str], dict[str, str]]:
    """Parse command line arguments and return test command and options."""
    # Basic test command
    test_command = [
        "pytest",
        "-v",
        "--tb=short",
        "--color=yes",  # Force colored output from pytest
    ]

    options = {}

    # Add coverage if requested
    if "--cov" in sys.argv or "--coverage" in sys.argv:
        test_command.extend(
            [
                "--cov=src/ai_agent",
                "--cov-report=term-missing",
                "--cov-report=html",
            ]
        )
        options["coverage"] = "enabled"

    # Add specific test markers if requested
    if "--unit" in sys.argv:
        test_command.extend(["-m", "unit"])
        options["filter"] = "unit tests only"
    elif "--integration" in sys.argv:
        test_command.extend(["-m", "integration"])
        options["filter"] = "integration tests only"
    elif "--e2e" in sys.argv:
        test_command.extend(["-m", "e2e"])
        options["filter"] = "end-to-end tests only"

    # Add verbose output if requested
    if "--verbose" in sys.argv or "-vv" in sys.argv:
        test_command.append("-vv")
        options["verbosity"] = "high"

    # Add specific test file if provided
    test_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if test_args:
        test_command.extend(test_args)
        options["specific_tests"] = ", ".join(test_args)

    return test_command, options


def main() -> None:
    """Run tests with coverage."""
    project_root = Path(__file__).parent.parent

    print_section("🧪 RUNNING TESTS")

    test_command, options = parse_args()

    # Display test configuration
    if options:
        print_info("Test configuration:")
        for key, value in options.items():
            print(f"  • {key.replace('_', ' ').title()}: {info(value)}")
        print()

    print(dim(f"Command: {' '.join(test_command)}"))
    print()

    try:
        # Run the tests
        result = subprocess.run(test_command, cwd=project_root)

        print()  # Add spacing before summary
        print_section("📊 TEST SUMMARY")

        if result.returncode == 0:
            print_success("All tests passed! 🎉")
            print(highlight("Your code is working correctly! ✨"))

            # Show coverage info if enabled
            if "coverage" in options:
                print_info("Coverage report generated:")
                print(f"  • Terminal: {dim('above output')}")
                print(f"  • HTML: {dim('htmlcov/index.html')}")

            # Suggest next steps
            print("\n" + command("💡 Next steps:"))
            print("  • " + dim("lint") + " - Run linting checks")
            print("  • " + dim("format") + " - Format your code")
            print("  • " + dim("git add .") + " - Stage your changes")

        else:
            print_error(f"Tests failed (exit code: {result.returncode})")

            # Provide helpful suggestions based on common failure scenarios
            print("\n" + command("💡 Debugging suggestions:"))
            print("  • " + dim("test --verbose") + " - Run with more detailed output")
            print("  • " + dim("test --unit") + " - Run only unit tests")
            print(
                "  • "
                + dim("test path/to/specific/test.py")
                + " - Run specific test file"
            )
            print("  • " + dim("pytest --lf") + " - Run only last failed tests")
            print("  • " + dim("pytest --pdb") + " - Drop into debugger on failures")

        sys.exit(result.returncode)

    except KeyboardInterrupt:
        print_warning("\n\nTests interrupted by user")
        sys.exit(130)
    except FileNotFoundError:
        print_error("pytest not found. Install with: pip install pytest")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
