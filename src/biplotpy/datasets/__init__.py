"""Classic datasets used in the examples."""
from importlib import resources

import pandas as pd


def _read(name):
    with resources.files(__package__).joinpath(name).open("r") as f:
        return pd.read_csv(f)


def load_iris() -> pd.DataFrame:
    """Fisher's iris data (150 x 5)."""
    return _read("iris.csv")


def load_mtcars() -> pd.DataFrame:
    """Motor Trend cars (32 x 12, with a ``name`` column)."""
    return _read("mtcars.csv").set_index("name")
