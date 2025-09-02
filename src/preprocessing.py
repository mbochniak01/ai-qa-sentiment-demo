"""
Preprocessing script for sentiment analysis demo.

- Downloads the NLTK movie reviews dataset
- Converts it into a DataFrame
- Splits into train/test sets
- Saves CSVs into data/processed/ under project root
"""

import logging
from pathlib import Path
from typing import Tuple

import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_movie_reviews() -> pd.DataFrame:
    """
    Download and load the NLTK movie reviews dataset.

    Returns:
        pd.DataFrame: DataFrame with columns [text, label]
    """
    nltk.download("movie_reviews", quiet=True)

    data = [
        (" ".join(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    df = pd.DataFrame(data, columns=["text", "label"])
    logging.info("Loaded %d samples from NLTK movie reviews", len(df))
    return df


def split_dataset(
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.

    Args:
        df (pd.DataFrame): Input dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, test_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    logging.info("Split dataset into %d train and %d test samples", len(train_df), len(test_df))
    return train_df, test_df


def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """
    Save train and test datasets to CSV files.

    Args:
        train_df (pd.DataFrame): Training set
        test_df (pd.DataFrame): Test set
        output_dir (Path): Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info("Saved train set to %s", train_path.relative_to(PROJECT_ROOT))
    logging.info("Saved test set to %s", test_path.relative_to(PROJECT_ROOT))


def main() -> None:
    """Main entrypoint for preprocessing pipeline."""
    df = load_movie_reviews()
    train_df, test_df = split_dataset(df)
    save_datasets(train_df, test_df)


if __name__ == "__main__":
    main()
