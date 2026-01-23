"""Constants for PE Database project"""
import pathlib
import torch

# Constants for project paths
# When installed as a package, these paths point to the project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT.joinpath("datasets").resolve()
MODEL_ROOT = PROJECT_ROOT.joinpath("vendor", "models").resolve()

# Constants for device configuration
# if torch version >= 2.0, use the new device selection logic
if torch.__version__ >= "2.0":
    DEVICE = (
        "mps" if torch.backends.mps.is_available()  # Apple Silicon
        else "cuda" if torch.cuda.is_available()  # NVIDIA GPU
        else "cpu")  # Fallback to CPU if no other device is available
else:
    # For older versions of PyTorch, use the traditional device selection logic
    DEVICE = (
        "cuda" if torch.cuda.is_available()  # NVIDIA GPU
        else "cpu")

# Commonly used paths
DEEPSPCAS9_MODEL_DIR = MODEL_ROOT.joinpath("DeepSpCas9").resolve()