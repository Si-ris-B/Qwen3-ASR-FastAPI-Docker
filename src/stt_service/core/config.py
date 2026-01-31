from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    APP_LOGGER_NAME: str = "qwen3_asr_api"

    # Defaults used only if not specified in the /load_model request
    DEFAULT_MODEL: str = "Qwen/Qwen3-ASR-0.6B"
    DEFAULT_DEVICE: str = "cuda"
    DEFAULT_DTYPE: str = "bf16"

    MODEL_CACHE_PATH: Path = Path("/app/models")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()