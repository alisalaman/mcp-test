"""Custom log formatters for structured logging."""

import json
from datetime import datetime
from typing import Any

from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform color support
init(autoreset=True)


class JSONFormatter:
    """JSON formatter for structured logs."""

    def __init__(self, ensure_ascii: bool = False, indent: int | None = None):
        self.ensure_ascii = ensure_ascii
        self.indent = indent

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> str:
        """Format log event as JSON."""
        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add log level
        event_dict["level"] = method_name.upper()

        # Add logger name
        if "logger" not in event_dict:
            event_dict["logger"] = logger.name

        # Format as JSON
        return json.dumps(
            event_dict, ensure_ascii=self.ensure_ascii, indent=self.indent, default=str
        )


class ConsoleFormatter:
    """Console formatter with colors and human-readable output."""

    def __init__(self, colors: bool = True, show_timestamp: bool = True):
        self.colors = colors
        self.show_timestamp = show_timestamp

        # Color mapping for log levels
        self.level_colors = {
            "debug": Fore.CYAN,
            "info": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "critical": Fore.RED + Back.WHITE + Style.BRIGHT,
        }

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> str:
        """Format log event for console output."""
        parts = []

        # Add timestamp
        if self.show_timestamp and "timestamp" in event_dict:
            timestamp = event_dict["timestamp"]
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            parts.append(f"[{timestamp}]")

        # Add log level with color
        level = method_name.upper()
        if self.colors and level.lower() in self.level_colors:
            color = self.level_colors[level.lower()]
            parts.append(f"{color}{level}{Style.RESET_ALL}")
        else:
            parts.append(level)

        # Add logger name
        if "logger" in event_dict:
            logger_name = event_dict["logger"]
            if self.colors:
                parts.append(f"{Fore.BLUE}[{logger_name}]{Style.RESET_ALL}")
            else:
                parts.append(f"[{logger_name}]")

        # Add correlation ID if present
        if "correlation_id" in event_dict:
            corr_id = event_dict["correlation_id"]
            if self.colors:
                parts.append(f"{Fore.MAGENTA}[{corr_id}]{Style.RESET_ALL}")
            else:
                parts.append(f"[{corr_id}]")

        # Add main message
        message = event_dict.get("event", "")
        if message:
            parts.append(message)

        # Add additional fields
        additional_fields = []
        for key, value in event_dict.items():
            if key not in ["timestamp", "level", "logger", "correlation_id", "event"]:
                if isinstance(value, dict | list):
                    value = json.dumps(value, default=str)
                additional_fields.append(f"{key}={value}")

        if additional_fields:
            if self.colors:
                parts.append(
                    f"{Fore.WHITE}{', '.join(additional_fields)}{Style.RESET_ALL}"
                )
            else:
                parts.append(", ".join(additional_fields))

        return " ".join(parts)


class StructuredFormatter:
    """Structured formatter with key-value pairs."""

    def __init__(self, separator: str = " | ", key_value_separator: str = "="):
        self.separator = separator
        self.key_value_separator = key_value_separator

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> str:
        """Format log event as structured key-value pairs."""
        parts = []

        # Add timestamp
        if "timestamp" in event_dict:
            timestamp = event_dict["timestamp"]
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            parts.append(f"timestamp{self.key_value_separator}{timestamp}")

        # Add log level
        parts.append(f"level{self.key_value_separator}{method_name.upper()}")

        # Add logger name
        if "logger" in event_dict:
            parts.append(f"logger{self.key_value_separator}{event_dict['logger']}")

        # Add correlation ID
        if "correlation_id" in event_dict:
            parts.append(
                f"correlation_id{self.key_value_separator}{event_dict['correlation_id']}"
            )

        # Add main message
        if "event" in event_dict:
            parts.append(f"message{self.key_value_separator}{event_dict['event']}")

        # Add additional fields
        for key, value in event_dict.items():
            if key not in ["timestamp", "level", "logger", "correlation_id", "event"]:
                if isinstance(value, dict | list):
                    value = json.dumps(value, default=str)
                parts.append(f"{key}{self.key_value_separator}{value}")

        return self.separator.join(parts)


class CustomFormatter:
    """Custom formatter with configurable template."""

    def __init__(self, template: str, **kwargs: Any) -> None:
        self.template = template
        self.kwargs = kwargs

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> str:
        """Format log event using custom template."""
        # Prepare context
        context = {
            "timestamp": event_dict.get("timestamp", datetime.utcnow().isoformat()),
            "level": method_name.upper(),
            "logger": event_dict.get("logger", logger.name),
            "correlation_id": event_dict.get("correlation_id", ""),
            "message": event_dict.get("event", ""),
            **event_dict,
            **self.kwargs,
        }

        # Format template
        try:
            return self.template.format(**context)
        except KeyError as e:
            return f"Template error: missing key {e}"


class LogFilter:
    """Filter for log records based on criteria."""

    def __init__(
        self,
        min_level: str | None = None,
        max_level: str | None = None,
        include_loggers: list[str] | None = None,
        exclude_loggers: list[str] | None = None,
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
    ):
        self.min_level = min_level
        self.max_level = max_level
        self.include_loggers = include_loggers
        self.exclude_loggers = exclude_loggers
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields

        # Level hierarchy
        self.level_hierarchy = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> bool:
        """Filter log event based on criteria."""
        # Check level filters
        if self.min_level:
            current_level = self.level_hierarchy.get(method_name.lower(), 0)
            min_level = self.level_hierarchy.get(self.min_level.lower(), 0)
            if current_level < min_level:
                return False

        if self.max_level:
            current_level = self.level_hierarchy.get(method_name.lower(), 0)
            max_level = self.level_hierarchy.get(self.max_level.lower(), 0)
            if current_level > max_level:
                return False

        # Check logger filters
        logger_name = event_dict.get("logger", logger.name)
        if self.include_loggers and logger_name not in self.include_loggers:
            return False

        if self.exclude_loggers and logger_name in self.exclude_loggers:
            return False

        # Check field filters
        if self.include_fields:
            if not any(field in event_dict for field in self.include_fields):
                return False

        if self.exclude_fields:
            if any(field in event_dict for field in self.exclude_fields):
                return False

        return True


class LogSampler:
    """Sample logs based on criteria."""

    def __init__(self, sample_rate: float = 1.0, sample_key: str | None = None):
        self.sample_rate = sample_rate
        self.sample_key = sample_key

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> bool:
        """Determine if log should be sampled."""
        if self.sample_rate >= 1.0:
            return True

        # Use correlation ID or other key for consistent sampling
        if self.sample_key and self.sample_key in event_dict:
            key_value = str(event_dict[self.sample_key])
            # Simple hash-based sampling
            hash_value = hash(key_value) % 100
            return hash_value < (self.sample_rate * 100)

        # Random sampling
        import random

        return random.random() < self.sample_rate


class LogAggregator:
    """Aggregate similar log events."""

    def __init__(self, window_size: int = 60, max_events: int = 1000):
        self.window_size = window_size
        self.max_events = max_events
        self.events: dict[str, Any] = {}
        self.last_cleanup = datetime.utcnow()

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Aggregate log event or return None if aggregated."""
        current_time = datetime.utcnow()

        # Cleanup old events
        if (current_time - self.last_cleanup).seconds > self.window_size:
            self._cleanup_old_events(current_time)
            self.last_cleanup = current_time

        # Create aggregation key
        key = self._create_aggregation_key(method_name, event_dict)

        if key in self.events:
            # Increment count
            self.events[key]["count"] += 1
            self.events[key]["last_seen"] = current_time
            return None  # Don't log this event
        else:
            # New event
            self.events[key] = {
                "count": 1,
                "first_seen": current_time,
                "last_seen": current_time,
                "event": event_dict,
            }
            return event_dict

    def _create_aggregation_key(
        self, method_name: str, event_dict: dict[str, Any]
    ) -> str:
        """Create aggregation key for similar events."""
        # Use logger, level, and message as key
        logger_name = event_dict.get("logger", "")
        message = event_dict.get("event", "")
        return f"{logger_name}:{method_name}:{message}"

    def _cleanup_old_events(self, current_time: datetime) -> None:
        """Remove old events from aggregation."""
        cutoff_time = current_time.timestamp() - self.window_size
        keys_to_remove = []

        for key, event_data in self.events.items():
            if event_data["last_seen"].timestamp() < cutoff_time:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.events[key]

        # Limit total events
        if len(self.events) > self.max_events:
            # Remove oldest events
            sorted_events = sorted(self.events.items(), key=lambda x: x[1]["last_seen"])
            for key, _ in sorted_events[: len(self.events) - self.max_events]:
                del self.events[key]
