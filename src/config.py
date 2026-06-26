from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

    app_name: str = "Hospital AI Command Center"
    environment: str = "development"
    debug: bool = False

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hospital_ai"
    database_pool_size: int = 5

    # ChromaDB (embedded, no separate service)
    chroma_persist_path: str = "./data/chroma"
    chroma_collection_name: str = "clinical_knowledge"

    # ML
    model_artifacts_dir: str = "./data/models"

    # Live feed
    feed_interval_seconds: float = 12.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
