import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class HuggingFaceSettings(LLMSettings):
    """Hugging Face-specific settings extending LLMSettings."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    cache_dir: str = Field(default_factory=lambda: os.getenv("HF_CACHE_DIR", "./hf_cache"))


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "mydatabase"))  # Default to 'mydatabase' or your database name
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "ritika02"))     # Custom user defined in docker-compose.yml
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "ritika")) # Custom password defined in docker-compose.yml
    host: str = Field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))  # Default to localhost if not defined
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", 5432)))  # Default to port 5432
    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))



class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings_1"
    embedding_dimensions: int = 384  # Updated for SentenceTransformer
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
