# src/utils.py
# -*- coding: utf-8 -*-
import pathlib


# Path utility functions
def get_project_root() -> str:
    """Returns the root path of the project."""
    return str(pathlib.Path(__file__).parent.parent.resolve())

def get_data_path() -> str:
    """Returns the path to the data directory."""
    return str(pathlib.Path(__file__).parent.parent.joinpath("data").resolve())

def get_model_path() -> str:
    """Returns the path to the model directory."""
    return str(pathlib.Path(__file__).parent.parent.joinpath("models").resolve())

def get_database_path() -> str:
    """Returns the path to the database directory."""
    return str(pathlib.Path(__file__).parent.parent.joinpath("database").resolve())