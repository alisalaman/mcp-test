"""Development environment configuration."""

from ai_agent.config.settings import DevelopmentSettings

# Development-specific configuration overrides
development_config = DevelopmentSettings(
    # Development server settings
    host="127.0.0.1",
    port=8000,
    workers=1,
    # Development storage
    use_memory=True,
    use_database=False,
    use_redis=False,
    # Development features
    debug=True,
)
