from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Settings
    LOG_LEVEL: str = "INFO"
    APP_LOGGER_NAME: str = "qwen3_asr_api"

    # Model Defaults (can be overridden by .env)
    DEFAULT_MODEL: str = "Qwen/Qwen3-ASR-0.6B"
    DEFAULT_ALIGNER: str = "Qwen/Qwen3-ForcedAligner-0.6B"

    # Internal Docker Paths
    # We store models in /app/models inside the container
    MODEL_CACHE_PATH: Path = Path("/app/models")

    # Allow reading from .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()