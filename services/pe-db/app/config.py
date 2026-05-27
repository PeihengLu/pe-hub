"""Application configuration for the PE Database service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Path(
        os.getenv("DATA_ROOT") or os.getenv("PE_DATA_ROOT") or (_REPO_ROOT / "datasets")
    ).expanduser().resolve()
    database_url: str = ""

    @property
    def catalog_db_path(self) -> Path:
        return self.data_root / "catalog" / "pe_database.db"

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.catalog_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.catalog_db_path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
