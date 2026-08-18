"""
Sentiment analysis helpers for the 'sentiment' column.

Provides percentage distribution of sentiment labels and the top reviews
per sentiment class, ordered by thumbs-up count.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.exception import MyException
from utils.logger import logging

SENTIMENT_COLUMN = "sentiment"
SENTIMENT_CLASSES = ["positive", "neutral", "negative"]
THUMBS_UP_COLUMN = "thumbsUpCount"
TOP_N = 5

REVIEW_COLUMNS = ["content", THUMBS_UP_COLUMN]


def _validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
    """Validate the input DataFrame and raise a clear error if invalid."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"The following required column(s) are missing from the "
            f"DataFrame: {', '.join(missing)}."
        )


def get_sentiment_percentages(df: pd.DataFrame) -> dict:
    """
    Calculate the percentage distribution of the 'sentiment' column.

    Missing values (NaN) are included in the distribution. When the
    DataFrame is empty, an empty dictionary is returned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'sentiment' column.

    Returns
    -------
    dict
        Mapping of sentiment value -> percentage, rounded to 2 decimals.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    KeyError
        If the 'sentiment' column does not exist.
    """
    try:
        _validate_dataframe(df, [SENTIMENT_COLUMN])

        if df.empty:
            logging.info("Input DataFrame is empty; returning empty result.")
            return {}

        # dropna=False ensures missing values (NaN) are counted and included.
        percentages = (
            df[SENTIMENT_COLUMN]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
            .to_dict()
        )

        logging.info(
            f"Sentiment percentages computed across "
            f"{len(df)} rows ({len(percentages)} distinct values)."
        )
        return percentages

    except Exception as e:
        logging.error(f"Failed to compute sentiment percentages: {e}")
        raise MyException(e, sys)


def get_top_thumbs_up(df: pd.DataFrame) -> dict:
    """
    Return the top reviews per sentiment class by thumbs-up count.

    Each value is a DataFrame with the 'content' and 'thumbsUpCount'
    columns, sorted descending by thumbs-up count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'sentiment', 'content', and
        'thumbsUpCount' columns.

    Returns
    -------
    dict
        Mapping of sentiment class -> top-N DataFrame. Classes that
        have no rows in the input map to an empty DataFrame.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    KeyError
        If any required column is missing.
    """
    try:
        _validate_dataframe(df, [SENTIMENT_COLUMN] + REVIEW_COLUMNS)

        if df.empty:
            logging.warning("Input DataFrame is empty; returning empty result.")
            return {cls: pd.DataFrame(columns=REVIEW_COLUMNS) for cls in SENTIMENT_CLASSES}

        results = {}
        for cls in SENTIMENT_CLASSES:
            subset = df[df[SENTIMENT_COLUMN] == cls]
            top = (
                subset.nlargest(TOP_N, THUMBS_UP_COLUMN)[REVIEW_COLUMNS]
                .reset_index(drop=True)
                if not subset.empty
                else pd.DataFrame(columns=REVIEW_COLUMNS)
            )
            results[cls] = top

        logging.info(
            f"Top {TOP_N} reviews computed for classes: {', '.join(results)}."
        )
        return results

    except Exception as e:
        logging.error(f"Failed to compute top thumbs-up reviews: {e}")
        raise MyException(e, sys)


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    # Example setup including valid sentiments, a rare sentiment,
    # and a missing value (None).
    data = {
        "sentiment": ["positive", "negative", "positive", None, "neutral", "positive"],
        "content": [
            "Great app!",
            "Terrible experience.",
            "Love it!",
            "Missing sentiment row.",
            "It is okay.",
            "Awesome!",
        ],
        "thumbsUpCount": [20, 3, 45, 8, 5, 30],
    }
    df_results = pd.DataFrame(data)

    print("\n=== Sentiment Percentages ===")
    print(get_sentiment_percentages(df_results))

    print("\n=== Top Thumbs-Up Reviews ===")
    top_reviews = get_top_thumbs_up(df_results)
    for cls, reviews in top_reviews.items():
        print(f"\n[{cls}]")
        print(reviews)
