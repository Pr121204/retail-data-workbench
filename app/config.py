from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./retail.db"
    GEMINI_API_KEY: str = ""
    USE_MOCK_LLM: bool = True
    MAX_UPLOAD_FILES: int = Field(default=10, ge=1, le=100)
    MAX_UPLOAD_BYTES: int = Field(default=25 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
