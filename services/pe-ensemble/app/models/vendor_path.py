import os
from pathlib import Path
from typing import Iterable


def _candidate_roots() -> Iterable[Path]:
    """Yield potential roots that may contain vendor/models.

    Priority order:
    1) MODEL_PATH environment variable (expected to point at vendor/models)
    2) Directories ascending from this file location
    3) /app (Docker workdir)
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        env_root = Path(env_path).expanduser()
        yield env_root
        yield env_root.parent

    here = Path(__file__).resolve()
    yield here.parent
    yield from here.parents

    yield Path("/app")


def resolve_vendor_models_path(*subdirs: str) -> Path:
    """Resolve the vendor/models path (optionally with a subdirectory).

    Returns the first existing path found across the candidate roots.
    Raises RuntimeError with guidance if the path cannot be located.
    """
    suffix = Path(*subdirs) if subdirs else None

    for root in _candidate_roots():
        vendor_candidates = []

        vendor_root = root / "vendor" / "models"
        if vendor_root.is_dir():
            vendor_candidates.append(vendor_root)

        if root.name == "models" and root.parent.name == "vendor" and root.is_dir():
            vendor_candidates.append(root)

        for candidate_root in vendor_candidates:
            target = candidate_root / suffix if suffix else candidate_root
            if target.is_dir():
                return target

    missing = f"/vendor/models/{suffix}" if suffix else "/vendor/models"
    raise RuntimeError(
        f"Could not locate {missing}. Set MODEL_PATH or mount vendor/models in the container."
    )
