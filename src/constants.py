import pathlib

# Constants for project paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT.joinpath("data").resolve()
MODEL_ROOT = PROJECT_ROOT.joinpath("models").resolve()
DATABASE_ROOT = PROJECT_ROOT.joinpath("database").resolve()
