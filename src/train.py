"""
Train a baseline sentiment analysis model.

- Loads processed train/test CSVs
- Trains TF-IDF + Logistic Regression classifier
- Saves trained model to models/model.pkl
"""

import logging
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs into DataFrames."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    logging.info("Loaded %d train samples and %d test samples", len(train_df), len(test_df))
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Train a TF-IDF + Logistic Regression pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train_df["text"], train_df["label"])
    logging.info("Model training complete")
    return pipeline


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> None:
    """Evaluate model on test data and print metrics."""
    preds = model.predict(test_df["text"])
    acc = accuracy_score(test_df["label"], preds)
    logging.info("Test Accuracy: %.4f", acc)
    logging.info("\n%s", classification_report(test_df["label"], preds))


def save_model(model: Pipeline) -> None:
    """Save trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logging.info("Saved trained model to %s", MODEL_PATH.relative_to(PROJECT_ROOT))


def main() -> None:
    train_df, test_df = load_data()
    model = train_model(train_df)
    evaluate_model(model, test_df)
    save_model(model)


if __name__ == "__main__":
    main()
