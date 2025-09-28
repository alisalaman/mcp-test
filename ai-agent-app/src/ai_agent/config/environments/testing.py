"""Testing environment configuration."""

from ai_agent.config.settings import TestingSettings

# Testing-specific configuration overrides
testing_config = TestingSettings(
    # Testing server settings
    host="127.0.0.1",
    port=8001,  # Different port to avoid conflicts
    workers=1,
    # Testing storage (in-memory for speed)
    use_memory=True,
    use_database=False,
    use_redis=False,
    # Testing features
    debug=True,
)
