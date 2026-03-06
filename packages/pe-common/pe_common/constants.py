"""Constants for PE Database project"""
import os
import pathlib

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None


def _resolve_project_root() -> pathlib.Path:
    env_root = os.getenv("PE_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")
    if env_root:
        return pathlib.Path(env_root).expanduser().resolve()

    def _looks_like_project_root(path: pathlib.Path) -> bool:
        return (
            (path / "packages" / "pe-common" / "pe_common").exists()
            or (
                (path / "datasets").exists()
                and (path / "services").exists()
                and (path / "requirements.txt").exists()
            )
        )
    # Check current working directory and its parents
    cwd = pathlib.Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if _looks_like_project_root(candidate):
            return candidate

    current = pathlib.Path(__file__).resolve()
    for candidate in [current.parent] + list(current.parents):
        if _looks_like_project_root(candidate):
            return candidate

    return cwd


def _normalize_data_root(path: pathlib.Path) -> pathlib.Path:
    if path.name in {"standardized", "raw"}:
        return path.parent
    return path


def _resolve_data_root(project_root: pathlib.Path) -> pathlib.Path:
    for env_var in ("DATA_ROOT", "PE_DATA_ROOT", "DATA_PATH"):
        value = os.getenv(env_var)
        if value:
            candidate = pathlib.Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return _normalize_data_root(candidate)

    return (project_root / "datasets").resolve()


def _resolve_model_root(project_root: pathlib.Path) -> pathlib.Path:
    for env_var in ("MODEL_ROOT", "MODEL_PATH"):
        value = os.getenv(env_var)
        if value:
            candidate = pathlib.Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if candidate.name == "vendor":
                return (candidate / "models").resolve()
            if candidate.name == "models" and candidate.parent.name == "vendor":
                return candidate.resolve()
            if (candidate / "vendor" / "models").exists():
                return (candidate / "vendor" / "models").resolve()

            return candidate.resolve()

    return (project_root / "vendor" / "models").resolve()


def _torch_version_at_least(major: int, minor: int) -> bool:
    if torch is None or not hasattr(torch, "__version__"):
        return False
    version = torch.__version__.split("+")[0]
    try:
        parts = [int(part) for part in version.split(".")[:2]]
    except ValueError:
        return True
    return parts[0] > major or (parts[0] == major and parts[1] >= minor)

# Constants for device configuration
if torch is None:
    DEVICE = "cpu"
elif _torch_version_at_least(2, 0):
    mps_backend = getattr(torch.backends, "mps", None)
    DEVICE = (
        "mps" if mps_backend and torch.backends.mps.is_available()  # Apple Silicon
        else "cuda" if torch.cuda.is_available()  # NVIDIA GPU
        else "cpu"
    )
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants for project paths
PROJECT_ROOT = _resolve_project_root()
DATA_ROOT = _resolve_data_root(PROJECT_ROOT)
MODEL_ROOT = _resolve_model_root(PROJECT_ROOT)

# Commonly used paths
DEEPSPCAS9_MODEL_DIR = MODEL_ROOT.joinpath("DeepSpCas9").resolve()