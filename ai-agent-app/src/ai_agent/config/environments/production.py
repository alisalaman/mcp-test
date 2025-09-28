"""Production environment configuration."""

from ai_agent.config.settings import ProductionSettings

# Production-specific configuration overrides
production_config = ProductionSettings(
    # Production server settings
    host="0.0.0.0",
    port=8000,
    workers=4,
    # Production storage
    use_memory=False,
    use_database=True,
    use_redis=True,
    # Production features
    debug=False,
)
