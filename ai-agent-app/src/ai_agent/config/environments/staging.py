"""Staging environment configuration."""

from ai_agent.config.settings import StagingSettings

# Staging-specific configuration overrides
staging_config = StagingSettings(
    # Staging server settings
    host="0.0.0.0",
    port=8000,
    workers=2,
    # Staging storage (production-like)
    use_memory=False,
    use_database=True,
    use_redis=True,
    # Staging features
    debug=False,
)
