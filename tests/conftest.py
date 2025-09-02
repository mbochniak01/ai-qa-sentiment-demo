# tests/conftest.py
import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def model() -> Pipeline:
    """Load the trained sentiment model."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Load the test dataset."""
    return pd.read_csv(DATA_DIR / "test.csv")
