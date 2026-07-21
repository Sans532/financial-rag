"""Environment-based configuration. Reads from .env via python-dotenv."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"

    sec_user_agent: str = "financial-rag research-project you@example.com"

    finnhub_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "sec_filings"
    # If set, Qdrant runs embedded (on-disk, no server process) instead of connecting to
    # qdrant_url — the no-Docker-required path from Qdrant's official "local mode". Still a
    # real persistent store (not in-memory-only), just without a separate server to run.
    qdrant_local_path: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"

    data_dir: Path = PROJECT_ROOT / "data"
    raw_filings_dir: Path = PROJECT_ROOT / "data" / "raw"
    eval_set_path: Path = PROJECT_ROOT / "data" / "eval_set.json"
    retrieval_labels_path: Path = PROJECT_ROOT / "data" / "retrieval_labels.json"
    eval_report_path: Path = PROJECT_ROOT / "eval_report.md"


@lru_cache
def get_settings() -> Settings:
    return Settings()
