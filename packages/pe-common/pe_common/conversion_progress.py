"""Cross-process conversion progress lines for PE-DB -> ensemble job logs."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_TOKEN_PATTERN = re.compile(r"^[a-f0-9]{8,64}$")


def progress_root() -> Path:
    configured = os.getenv("PE_CONVERSION_PROGRESS_DIR", "").strip()
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "pe-conversion-progress"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_token(progress_token: str) -> str:
    token = progress_token.strip().lower()
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid progress token")
    return token


def progress_file_path(progress_token: str) -> Path:
    token = _validate_token(progress_token)
    return progress_root() / f"{token}.log"


def clear_progress(progress_token: str) -> None:
    path = progress_file_path(progress_token)
    if path.is_file():
        path.unlink()


def append_progress(progress_token: str, message: str) -> None:
    text = message.strip()
    if not text:
        return
    path = progress_file_path(progress_token)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
