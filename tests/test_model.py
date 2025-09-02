"""
Automated AI QA tests for the sentiment analysis model.

Tests cover:
- Data quality
- Model performance
- Prediction class balance
- Robustness to simple input perturbations
"""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

# Constants
MIN_ACCURACY = 0.8  # minimum acceptable test accuracy


def test_data_quality(test_data: pd.DataFrame):
    """Check for missing texts or labels in test data."""
    assert not test_data["text"].isnull().any(), "Test data contains empty texts"
    assert not test_data["label"].isnull().any(), "Test data contains missing labels"
    assert all(test_data["label"].isin(["pos", "neg"])), "Labels contain unexpected values"


def test_model_accuracy(model: Pipeline, test_data: pd.DataFrame):
    """Check that the model achieves minimum accuracy."""
    preds = model.predict(test_data["text"])
    accuracy = (preds == test_data["label"]).mean()
    assert accuracy >= MIN_ACCURACY, f"Test accuracy below minimum: {accuracy:.2f}"


def test_prediction_class_balance(model: Pipeline, test_data: pd.DataFrame):
    """Check that predicted classes are roughly balanced."""
    preds = model.predict(test_data["text"])
    class_counts = pd.Series(preds).value_counts(normalize=True)
    for cls, freq in class_counts.items():
        assert 0.3 <= freq <= 0.7, f"Class {cls} is imbalanced ({freq:.2f})"


def test_model_robustness(model: Pipeline):
    """Check model handles minor input variations without errors."""
    test_samples = [
        "This movie was amazing and thrilling!",
        "Absolutely terrible film, I hated it.",
        "The movie was okay, not great, not bad."
    ]

    # Add minor perturbations
    perturbations = [
        lambda x: x.upper(),
        lambda x: x + " ",
        lambda x: x.replace("movie", "film")
    ]

    for sample in test_samples:
        for perturb in perturbations:
            perturbed_input = [perturb(sample)]
            preds = model.predict(perturbed_input)
            assert preds[0] in ["pos", "neg"], "Prediction produced unexpected label"
