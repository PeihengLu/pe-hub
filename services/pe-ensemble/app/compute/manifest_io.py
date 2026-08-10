"""Atomic JSON manifest helpers for filesystem-backed job registries."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON via a temp file + replace so concurrent readers never see empty files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_json_retry(
    path: Path,
    *,
    missing_message: str,
    retries: int = 20,
    delay: float = 0.01,
) -> Dict[str, Any]:
    """Read JSON, briefly retrying if a concurrent writer truncated the file."""
    if not path.is_file():
        raise FileNotFoundError(missing_message)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if not text.strip():
                raise json.JSONDecodeError("Expecting value", text, 0)
            return json.loads(text)
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(delay)
    assert last_error is not None
    raise last_error
