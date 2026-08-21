"""
Tests for the inference layer.

Covers the main methods:
    - get_sentiment_percentages / get_top_thumbs_up (analyze_sentiment)
    - PredictionPipeline.predict / predict_text (with an injected
      in-memory model + vectorizer, so no MLflow / network is needed)
"""

import numpy as np
import pandas as pd
import pytest

from utils.exception import MyException
from src.inference.analyze_sentiment import (
    get_sentiment_percentages,
    get_top_thumbs_up,
    SENTIMENT_CLASSES,
)
from src.inference.prediction import PredictionPipeline


@pytest.fixture
def prediction_pipeline(tiny_model_and_vectorizer) -> PredictionPipeline:
    """PredictionPipeline with a pre-loaded model (skips MLflow entirely)."""
    model, vectorizer = tiny_model_and_vectorizer
    return PredictionPipeline(model=model, vectorizer=vectorizer)


# ==============================================
# SENTIMENT PERCENTAGES
# ==============================================
class TestGetSentimentPercentages:
    def test_percentages_sum_to_100(self):
        df = pd.DataFrame(
            {"sentiment": ["positive", "positive", "negative", "neutral"]}
        )
        result = get_sentiment_percentages(df)

        assert abs(sum(result.values()) - 100.0) < 0.01
        assert set(result.keys()) == {"positive", "negative", "neutral"}

    def test_values_are_rounded_to_two_decimals(self):
        df = pd.DataFrame({"sentiment": ["positive"] * 3 + ["negative"]})
        result = get_sentiment_percentages(df)

        assert all(round(v, 2) == v for v in result.values())
        assert result["positive"] == 75.0

    def test_nan_values_are_included(self):
        df = pd.DataFrame({"sentiment": ["positive", None]})
        result = get_sentiment_percentages(df)

        assert len(result) == 2

    def test_empty_dataframe_returns_empty_dict(self):
        assert get_sentiment_percentages(pd.DataFrame({"sentiment": []})) == {}

    def test_missing_column_raises_my_exception(self):
        with pytest.raises(MyException):
            get_sentiment_percentages(pd.DataFrame({"label": ["positive"]}))

    def test_non_dataframe_raises_my_exception(self):
        with pytest.raises(MyException):
            get_sentiment_percentages(["positive", "negative"])


# ==============================================
# TOP THUMBS-UP REVIEWS
# ==============================================
class TestGetTopThumbsUp:
    def test_returns_all_sentiment_classes(self):
        df = pd.DataFrame(
            {
                "sentiment": ["positive", "negative"],
                "content": ["love it", "hate it"],
                "thumbsUpCount": [5, 2],
            }
        )
        result = get_top_thumbs_up(df)

        assert set(result.keys()) == set(SENTIMENT_CLASSES)
        # Classes without rows map to empty DataFrames.
        assert result["neutral"].empty

    def test_sorted_descending_by_thumbs_up(self):
        df = pd.DataFrame(
            {
                "sentiment": ["positive"] * 3,
                "content": ["a", "b", "c"],
                "thumbsUpCount": [1, 50, 10],
            }
        )
        top = get_top_thumbs_up(df)["positive"]

        assert top["thumbsUpCount"].tolist() == [50, 10, 1]
        assert top["content"].tolist() == ["b", "c", "a"]

    def test_top_n_is_capped_at_five(self):
        df = pd.DataFrame(
            {
                "sentiment": ["positive"] * 8,
                "content": [f"review {i}" for i in range(8)],
                "thumbsUpCount": list(range(8)),
            }
        )
        top = get_top_thumbs_up(df)["positive"]

        assert len(top) == 5

    def test_empty_dataframe_returns_empty_frames_per_class(self):
        empty_df = pd.DataFrame(
            columns=["sentiment", "content", "thumbsUpCount"]
        )
        result = get_top_thumbs_up(empty_df)

        for cls in SENTIMENT_CLASSES:
            assert isinstance(result[cls], pd.DataFrame)
            assert result[cls].empty

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"sentiment": ["positive"], "content": ["hi"]})

        with pytest.raises(MyException):
            get_top_thumbs_up(df)


# ==============================================
# PREDICTION PIPELINE
# ==============================================
class TestPredictionPipeline:
    def test_predict_adds_label_column(self, prediction_pipeline):
        df = pd.DataFrame(
            {
                "content": ["love this app amazing great experience", "broken crashes bugs terrible"],
                "score": [5, 1],
            }
        )
        result = prediction_pipeline.predict(df)

        assert "label" in result.columns
        assert len(result) == 2
        # Original columns are preserved.
        assert "score" in result.columns

    def test_predict_decode_labels_adds_sentiment(self, prediction_pipeline):
        df = pd.DataFrame({"content": ["love this app amazing great experience"]})
        result = prediction_pipeline.predict(df, decode_labels=True)

        assert result.iloc[0]["sentiment"] == "positive"
        assert result.iloc[0]["label"] == 2

    def test_predict_negative_review(self, prediction_pipeline):
        result = prediction_pipeline.predict_text(
            "terrible broken crashes constantly every day"
        )
        assert result == "negative"

    def test_predict_text_single_string_returns_scalar(self, prediction_pipeline):
        result = prediction_pipeline.predict_text("amazing love it great work team")
        assert isinstance(result, str)

    def test_predict_text_list_returns_list(self, prediction_pipeline):
        results = prediction_pipeline.predict_text(
            [
                "amazing love it great work team",
                "broken crashes bugs terrible horrible",
                "average okay normal nothing more",
            ]
        )
        assert isinstance(results, list)
        assert len(results) == 3
        assert set(results) == {"positive", "negative", "neutral"}

    def test_predict_accepts_raw_list_input(self, prediction_pipeline):
        result = prediction_pipeline.predict(
            ["amazing love it great work team"], decode_labels=True
        )
        assert result.iloc[0]["sentiment"] == "positive"

    def test_predict_accepts_string_input(self, prediction_pipeline):
        result = prediction_pipeline.predict("amazing love it great work team")
        assert len(result) == 1

    def test_emoji_only_text_gets_na_label(self, prediction_pipeline):
        result = prediction_pipeline.predict(["😃🎉🚀"])

        assert pd.isna(result.iloc[0]["label"])

    def test_empty_dataframe_returns_empty_with_label_column(
        self, prediction_pipeline
    ):
        result = prediction_pipeline.predict(pd.DataFrame({"content": []}))

        assert result.empty
        assert "label" in result.columns

    def test_invalid_input_type_raises_my_exception(self, prediction_pipeline):
        with pytest.raises(MyException):
            prediction_pipeline.predict(12345)

    def test_missing_content_column_raises_my_exception(self, prediction_pipeline):
        with pytest.raises(MyException):
            prediction_pipeline.predict(pd.DataFrame({"text": ["hello"]}))

    def test_batch_processing_keeps_all_valid_rows(self, prediction_pipeline):
        texts = (
            ["amazing love it great work team"] * 7
            + ["broken crashes bugs terrible horrible"] * 7
        )
        result = prediction_pipeline.predict(texts, batch_size=5)

        assert len(result) == 14
        assert result["label"].notna().sum() == 14
