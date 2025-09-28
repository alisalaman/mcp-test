#!/usr/bin/env python3
"""Simple test runner for the AI Agent application."""

import subprocess
import sys
from pathlib import Path


def run_tests(test_path: str = "tests/", verbose: bool = True, coverage: bool = False):
    """Run tests with optional coverage."""
    project_root = Path(__file__).parent

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=src/ai_agent",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ]
        )

    # Add test path
    cmd.append(str(project_root / test_path))

    print(f"Running: {' '.join(cmd)}")
    print(f"Project root: {project_root}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest pytest-cov")
        return 1


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Agent tests")
    parser.add_argument("--path", default="tests/", help="Test path to run")
    parser.add_argument(
        "--no-verbose", action="store_true", help="Disable verbose output"
    )
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--resilience", action="store_true", help="Run only resilience tests"
    )

    args = parser.parse_args()

    # Determine test path
    if args.unit:
        test_path = "tests/unit/"
    elif args.integration:
        test_path = "tests/integration/"
    elif args.resilience:
        test_path = "tests/unit/test_resilience/"
    else:
        test_path = args.path

    # Run tests
    exit_code = run_tests(
        test_path=test_path, verbose=not args.no_verbose, coverage=args.coverage
    )

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
