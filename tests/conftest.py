"""
Shared pytest fixtures and environment setup for the RevuAI test suite.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Keep MLflow/DagsHub initialization out of the test process: marking the
# shared setup as already-initialized makes every later import of
# src.ml.train / src.ml.registry / src.inference.prediction skip the
# network round-trip against DagsHub.
import src.ml.mlflow_setup as _mlflow_setup  # noqa: E402

_mlflow_setup._initialized = True


# ==============================================
# Data Fixtures
# ==============================================
@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Raw reviews DataFrame mirroring the source CSV schema."""
    return pd.DataFrame(
        {
            "content": [
                "Beautiful app! 😍 visit https://google.com <b>Love it!</b>",
                "😃🎉🚀",
                "This app keeps CRASHING every single day.",
                "",
                None,
                "It is okay, nothing special.",
            ],
            "score": [5, 5, 1, 3, 2, 3],
            "thumbsUpCount": [10, 0, 25, 1, 2, 7],
            "label": [
                "positive",
                "positive",
                "negative",
                "neutral",
                "negative",
                "neutral",
            ],
        }
    )


@pytest.fixture
def transformed_df(sample_raw_df) -> pd.DataFrame:
    """Sample data already processed through TransformData (small batches)."""
    from src.etl.transform import TransformData

    transformer = TransformData()
    transformer.batch_size = 2
    return transformer.data_transformation(sample_raw_df.copy())


@pytest.fixture
def tiny_model_and_vectorizer():
    """
    A real (tiny) LinearSVC + fitted TF-IDF vectorizer trained in-memory,
    used to exercise PredictionPipeline without touching MLflow.
    """
    texts = [
        "love this app amazing great experience",
        "best app ever wonderful and beautiful",
        "absolutely fantastic works perfectly fine",
        "terrible app crashes constantly every day",
        "worst experience bugs everywhere broken",
        "horrible support totally useless update",
        "it is okay average nothing special really",
        "standard features average results overall",
        "normal app does its job as expected",
        "amazing love it great work team",
        "broken crashes bugs terrible horrible",
        "average okay normal nothing more",
    ]
    labels = [2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 1]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)

    model = LinearSVC(C=1.0, max_iter=10000)
    model.fit(X, labels)
    return model, vectorizer
