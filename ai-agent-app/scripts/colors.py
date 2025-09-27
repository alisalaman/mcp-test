"""Terminal color utilities for script output."""

import os
import sys


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check for explicit no-color environment variables
    if os.environ.get("NO_COLOR") or os.environ.get("ANSI_COLORS_DISABLED"):
        return False

    # Check for explicit force color
    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check TERM environment variable
    term = os.environ.get("TERM", "")
    if term in ("dumb", "unknown"):
        return False

    return True


def colorize(text: str, color: str, style: str | None = None) -> str:
    """Apply color and optional style to text if color is supported."""
    if not supports_color():
        return text

    result = color + text + Colors.RESET
    if style:
        result = style + result

    return result


def success(text: str) -> str:
    """Format text as a success message (green)."""
    return colorize(text, Colors.BRIGHT_GREEN, Colors.BOLD)


def error(text: str) -> str:
    """Format text as an error message (red)."""
    return colorize(text, Colors.BRIGHT_RED, Colors.BOLD)


def warning(text: str) -> str:
    """Format text as a warning message (yellow)."""
    return colorize(text, Colors.BRIGHT_YELLOW, Colors.BOLD)


def info(text: str) -> str:
    """Format text as an info message (cyan)."""
    return colorize(text, Colors.BRIGHT_CYAN)


def command(text: str) -> str:
    """Format text as a command (blue)."""
    return colorize(text, Colors.BRIGHT_BLUE, Colors.BOLD)


def highlight(text: str) -> str:
    """Format text as highlighted (magenta)."""
    return colorize(text, Colors.BRIGHT_MAGENTA, Colors.BOLD)


def dim(text: str) -> str:
    """Format text as dimmed."""
    return colorize(text, Colors.BRIGHT_BLACK)


def section_header(text: str) -> str:
    """Format text as a section header."""
    separator = "─" * len(text)
    if supports_color():
        return (
            f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{separator}{Colors.RESET}\n"
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{text}{Colors.RESET}\n"
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{separator}{Colors.RESET}\n"
        )
    else:
        return f"\n{separator}\n{text}\n{separator}\n"


# Convenience functions for common patterns
def print_success(message: str) -> None:
    """Print a success message."""
    print(success(f"✅ {message}"))


def print_error(message: str) -> None:
    """Print an error message."""
    print(error(f"❌ {message}"))


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(warning(f"⚠️  {message}"))


def print_info(message: str) -> None:
    """Print an info message."""
    print(info(f"ℹ️  {message}"))


def print_command(message: str) -> None:
    """Print a command being executed."""
    print(command(f"🔧 {message}"))


def print_step(step: int, total: int, message: str) -> None:
    """Print a step in a multi-step process."""
    step_info = dim(f"[{step}/{total}]")
    print(f"{step_info} {info(message)}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(section_header(title))
