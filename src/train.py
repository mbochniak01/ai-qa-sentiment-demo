"""
Train a baseline sentiment analysis model.

- Loads processed train/test CSVs
- Trains TF-IDF + Logistic Regression classifier
- Saves trained model to models/model.pkl
- Uses caching to speed up repeated transformations
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from joblib import Memory
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
CACHE_DIR = PROJECT_ROOT / "cache"

CACHE_DIR.mkdir(exist_ok=True)

# Constants
TFIDF_MAX_FEATURES = 5000
LOGREG_MAX_ITER = 1000
RANDOM_STATE = 42


def load_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from CSV files."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logging.info("Loaded %d train samples and %d test samples", len(train_df), len(test_df))
    return train_df, test_df


def build_pipeline(cache_dir: Path = CACHE_DIR) -> Pipeline:
    """
    Build a scikit-learn pipeline: TF-IDF vectorizer + Logistic Regression.

    Args:
        cache_dir (Path): Directory to cache transformer outputs

    Returns:
        Pipeline: scikit-learn pipeline
    """
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)),
            ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_STATE)),
        ],
        memory=Memory(cache_dir, verbose=0)
    )
    return pipeline


def train_model(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    """Fit the pipeline on training data."""
    pipeline.fit(train_df["text"], train_df["label"])
    logging.info("Model training complete")
    return pipeline


def evaluate_model(pipeline: Pipeline, test_df: pd.DataFrame) -> None:
    """Evaluate the model on the test set and log metrics."""
    predictions = pipeline.predict(test_df["text"])
    acc = accuracy_score(test_df["label"], predictions)
    logging.info("Test Accuracy: %.4f", acc)
    logging.info("\n%s", classification_report(test_df["label"], predictions))


def save_model(pipeline: Pipeline, model_path: Path = MODEL_PATH) -> None:
    """Save the trained pipeline to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logging.info("Saved trained model to %s", model_path.relative_to(PROJECT_ROOT))


def main() -> None:
    """Main function to load data, train, evaluate, and save the model."""
    train_df, test_df = load_data()
    pipeline = build_pipeline()
    pipeline = train_model(pipeline, train_df)
    evaluate_model(pipeline, test_df)
    save_model(pipeline)


if __name__ == "__main__":
    main()
