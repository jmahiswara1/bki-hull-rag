"""Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-check all config values.
Copy .env.example to .env and fill in your credentials before running.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Supabase
    supabase_url: str = "https://your-project.supabase.co"
    supabase_key: str = "your-anon-or-service-key"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Model names
    main_model: str = "qwen2.5:7b"
    light_model: str = "qwen2.5:3b"
    embedding_model: str = "bge-m3"

    # LLM-as-judge model for the eval harness; falls back to main_model if unset.
    judge_model: str = "qwen2.5:7b"

    # Data paths
    pdf_path: str = "data/raw/Rules-for-Hull-2026.pdf"
    processed_dir: str = "data/processed"

    # Retrieval guardrails
    confidence_threshold: float = 0.65

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def pdf_path_resolved(self) -> Path:
        """Return PDF path as a resolved Path object."""
        return Path(self.pdf_path).resolve()

    @property
    def processed_dir_resolved(self) -> Path:
        """Return processed directory as a resolved Path object."""
        return Path(self.processed_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
