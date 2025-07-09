import pathlib
import torch

# Constants for project paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT.joinpath("data").resolve()
MODEL_ROOT = PROJECT_ROOT.joinpath("models").resolve()
DATABASE_ROOT = PROJECT_ROOT.joinpath("database").resolve()

# Constants for device configuration
DEVICE = (
    # "mps" if torch.backends.mps.is_available() # Apple Silicon
    # else "xla" if torch.xla.is_available() # TPU
    "cuda" if torch.cuda.is_available() # NVIDIA GPU
    else "cpu") # Fallback to CPU if no other device is available