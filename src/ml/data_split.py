"""
Data Split before the model training.

Splits the data into train and test subsets using the test size configured
in `config/settings.yml` (data_evaluation.test_data_size).
"""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import logging
from utils.exception import MyException
from utils.config_loader import settings


class DataSplit:
    """Splits a DataFrame into stratified train/test sets."""

    def __init__(self, target_column: str = "content", label_column: str = "label"):
        self.target_column = target_column
        self.label_column = label_column
        try:
            self.test_size = float(settings["data_evaluation"]["test_data_size"])
        except (KeyError, TypeError, ValueError) as e:
            logging.warning(
                f"Could not read 'data_evaluation.test_data_size' from settings: {e}. "
                "Falling back to default 0.2."
            )
            self.test_size = 0.2

    def data_split(self, df: pd.DataFrame) -> tuple:
        """
        Split the data into X_train, X_test, y_train, y_test.

        Uses stratified splitting on the label column so class proportions
        are preserved across train/test. Falls back to a plain split when
        any class is too rare to stratify.
        """
        try:
            logging.info("Start Data splitting")
            logging.info(f"Original Data shape: {df.shape}")

            if df is None or df.empty:
                raise ValueError("The provided DataFrame is empty or None.")

            if self.target_column not in df.columns:
                raise KeyError(f"Target column '{self.target_column}' not found in the input DataFrame.")
            if self.label_column not in df.columns:
                raise KeyError(f"Label column '{self.label_column}' not found in the input DataFrame.")

            df = df.dropna(subset=[self.target_column, self.label_column]).reset_index(drop=True)

            X = df[self.target_column]
            y = df[self.label_column]

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=42,
                    stratify=y,
                )
            except ValueError as e:
                logging.warning(
                    f"Stratified split failed ({e}). Falling back to a plain random split."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=42,
                )

            logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
            logging.info("Data splitting completed successfully.")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("An error occurred during data splitting.")
            raise MyException(e, sys)




# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        sample_df = pd.DataFrame(
            {
                "content": [
                    "Great app, works flawlessly!",
                    "Crashes every time I open it.",
                    "Love the new features.",
                    "Battery drains too fast.",
                    "Best productivity tool.",
                    "Keeps losing my data.",
                    "Simple and clean UI.",
                    "Too many ads.",
                    "Very intuitive.",
                    "Poor customer support.",
                ],
                "score": [5, 1, 4, 2, 5, 1, 4, 2, 5, 2],
            }
        )

        splitter = DataSplit()
        X_train, X_test, y_train, y_test = splitter.data_split(sample_df)
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
    except MyException as e:
        print(f"Data split failed: {e}")
