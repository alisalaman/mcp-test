"""Logging configuration and setup."""

import logging
import sys
from enum import Enum
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from .formatters import JSONFormatter, ConsoleFormatter, StructuredFormatter
from .correlation import CorrelationIDProcessor


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Logging formats."""

    JSON = "json"
    CONSOLE = "console"
    STRUCTURED = "structured"


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.JSON,
    log_file: str | None = None,
    enable_correlation: bool = True,
    enable_colors: bool = True,
    include_timestamps: bool = True,
    include_caller_info: bool = False,
) -> None:
    """Setup structured logging configuration."""

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level.value),
        format="%(message)s",
        stream=sys.stdout if not log_file else None,
        handlers=[
            (
                logging.StreamHandler(sys.stdout)
                if not log_file
                else logging.FileHandler(log_file)
            )
        ],
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add correlation ID processor
    if enable_correlation:
        processors.append(CorrelationIDProcessor())

    # Add caller info processor
    if include_caller_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Add timestamp processor
    if include_timestamps:
        processors.append(structlog.processors.TimeStamper(fmt="ISO"))

    # Add formatter based on format type
    if format_type == LogFormat.JSON:
        processors.append(JSONFormatter())
    elif format_type == LogFormat.CONSOLE:
        processors.append(ConsoleFormatter(colors=enable_colors))
    elif format_type == LogFormat.STRUCTURED:
        processors.append(StructuredFormatter())

    # Configure structlog
    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    logger = structlog.get_logger(name)
    return logger  # type: ignore[no-any-return]


class LoggingConfig:
    """Logging configuration class."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.JSON,
        log_file: str | None = None,
        enable_correlation: bool = True,
        enable_colors: bool = True,
        include_timestamps: bool = True,
        include_caller_info: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        rotation_when: str = "midnight",
        rotation_interval: int = 1,
    ):
        self.level = level
        self.format_type = format_type
        self.log_file = log_file
        self.enable_correlation = enable_correlation
        self.enable_colors = enable_colors
        self.include_timestamps = include_timestamps
        self.include_caller_info = include_caller_info
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.rotation_when = rotation_when
        self.rotation_interval = rotation_interval

    def setup(self) -> None:
        """Setup logging with this configuration."""
        setup_logging(
            level=self.level,
            format_type=self.format_type,
            log_file=self.log_file,
            enable_correlation=self.enable_correlation,
            enable_colors=self.enable_colors,
            include_timestamps=self.include_timestamps,
            include_caller_info=self.include_caller_info,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "level": self.level.value,
            "format_type": self.format_type.value,
            "log_file": self.log_file,
            "enable_correlation": self.enable_correlation,
            "enable_colors": self.enable_colors,
            "include_timestamps": self.include_timestamps,
            "include_caller_info": self.include_caller_info,
            "max_file_size": self.max_file_size,
            "backup_count": self.backup_count,
            "rotation_when": self.rotation_when,
            "rotation_interval": self.rotation_interval,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LoggingConfig":
        """Create configuration from dictionary."""
        return cls(
            level=LogLevel(config.get("level", "INFO")),
            format_type=LogFormat(config.get("format_type", "json")),
            log_file=config.get("log_file"),
            enable_correlation=config.get("enable_correlation", True),
            enable_colors=config.get("enable_colors", True),
            include_timestamps=config.get("include_timestamps", True),
            include_caller_info=config.get("include_caller_info", False),
            max_file_size=config.get("max_file_size", 10 * 1024 * 1024),
            backup_count=config.get("backup_count", 5),
            rotation_when=config.get("rotation_when", "midnight"),
            rotation_interval=config.get("rotation_interval", 1),
        )


# Global logging configuration
_logging_config: LoggingConfig | None = None


def configure_logging(config: LoggingConfig) -> None:
    """Configure global logging settings."""
    global _logging_config
    _logging_config = config
    config.setup()


def get_logging_config() -> LoggingConfig | None:
    """Get current logging configuration."""
    return _logging_config


def setup_development_logging() -> None:
    """Setup logging for development environment."""
    config = LoggingConfig(
        level=LogLevel.DEBUG,
        format_type=LogFormat.CONSOLE,
        enable_colors=True,
        include_caller_info=True,
        include_timestamps=True,
    )
    configure_logging(config)


def setup_production_logging(log_file: str) -> None:
    """Setup logging for production environment."""
    config = LoggingConfig(
        level=LogLevel.INFO,
        format_type=LogFormat.JSON,
        log_file=log_file,
        enable_correlation=True,
        include_timestamps=True,
        include_caller_info=False,
    )
    configure_logging(config)


def setup_testing_logging() -> None:
    """Setup logging for testing environment."""
    config = LoggingConfig(
        level=LogLevel.WARNING,
        format_type=LogFormat.STRUCTURED,
        enable_correlation=False,
        include_timestamps=False,
        include_caller_info=False,
    )
    configure_logging(config)
