"""
Tests for the ETL layer: extract -> transform -> load.

Covers the main methods:
    - ExtractData.data_extraction
    - clean_text / TransformData.data_transformation / _sentiment_to_number
    - _hash_content / LoadData._prepare_rows / _safe_int / load_data_async
"""

import asyncio
import uuid

import pandas as pd
import pytest

from utils.exception import MyException
from src.etl.transform import TransformData, clean_text
from src.etl.load import LoadData, _hash_content


# ==============================================
# EXTRACT
# ==============================================
class TestExtractData:
    def test_extracts_csv_into_dataframe(self, tmp_path, monkeypatch):
        import src.etl.extract as extract_module

        csv_file = tmp_path / "reviews.csv"
        pd.DataFrame({"content": ["a", "b"], "score": [5, 1]}).to_csv(
            csv_file, index=False
        )
        monkeypatch.setattr(extract_module, "file_path", str(csv_file))

        df = extract_module.ExtractData().data_extraction()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["content", "score"]

    def test_missing_file_raises_my_exception(self, tmp_path, monkeypatch):
        import src.etl.extract as extract_module

        monkeypatch.setattr(
            extract_module, "file_path", str(tmp_path / "missing.csv")
        )

        with pytest.raises(MyException):
            extract_module.ExtractData().data_extraction()


# ==============================================
# TRANSFORM
# ==============================================
class TestCleanText:
    def test_removes_html_tags(self):
        assert clean_text("<b>Hello</b> world") == "Hello world"

    def test_removes_urls(self):
        assert clean_text("visit https://google.com now") == "visit now"
        assert clean_text("see www.example.com ok") == "see ok"

    def test_removes_emojis(self):
        assert clean_text("Great! 😍🎉🚀") == "Great!"

    def test_collapses_whitespace(self):
        assert clean_text("too    many     spaces") == "too many spaces"

    def test_non_string_input_is_coerced(self):
        assert clean_text(12345) == "12345"

    def test_emoji_only_text_becomes_empty(self):
        assert clean_text("😃🎉🚀") == ""


class TestSentimentToNumber:
    def test_known_mappings(self):
        transformer = TransformData()
        assert transformer._sentiment_to_number("negative") == 0
        assert transformer._sentiment_to_number("neutral") == 1
        assert transformer._sentiment_to_number("positive") == 2

    def test_unknown_label_passes_through(self):
        transformer = TransformData()
        assert transformer._sentiment_to_number("unknown") == "unknown"


class TestTransformData:
    def test_drops_nan_empty_and_emoji_only_rows(self, sample_raw_df):
        transformer = TransformData()
        transformer.batch_size = 2

        result = transformer.data_transformation(sample_raw_df.copy())

        # 6 input rows -> only 3 survive (None, "" and emoji-only dropped)
        assert result.shape[0] == 3
        assert (result["content"].str.len() > 0).all()

    def test_lowercases_and_cleans_text(self, transformed_df):
        contents = transformed_df["content"].tolist()

        assert all(c == c.lower() for c in contents)
        assert not any("http" in c for c in contents)
        assert not any("<" in c for c in contents)

    def test_labels_converted_to_numbers(self, transformed_df):
        # positive -> 2, negative -> 0, neutral -> 1
        assert sorted(transformed_df["label"].unique().tolist()) == [0, 1, 2]

    def test_batch_processing_preserves_row_order(self, sample_raw_df):
        transformer = TransformData()
        transformer.batch_size = 1

        result = transformer.data_transformation(sample_raw_df.copy())
        assert "crashing" in result.iloc[1]["content"]

    def test_empty_dataframe_raises(self):
        transformer = TransformData()
        with pytest.raises(MyException):
            transformer.data_transformation(pd.DataFrame())

    def test_none_dataframe_raises(self):
        transformer = TransformData()
        with pytest.raises(MyException):
            transformer.data_transformation(None)

    def test_missing_target_column_raises(self):
        transformer = TransformData()
        with pytest.raises(MyException):
            transformer.data_transformation(pd.DataFrame({"text": ["hi"]}))



# ==============================================
# LOAD
# ==============================================
class TestHashContent:
    def test_hash_is_stable(self):
        assert _hash_content("hello") == _hash_content("hello")

    def test_hash_normalizes_case_and_outer_whitespace(self):
        assert _hash_content("Hello World") == _hash_content("  hello world\t")

    def test_different_content_different_hash(self):
        assert _hash_content("hello") != _hash_content("world")


class TestSafeInt:
    def test_valid_values(self):
        assert LoadData._safe_int(5) == 5
        assert LoadData._safe_int("7") == 7

    def test_none_and_nan_return_none(self):
        assert LoadData._safe_int(None) is None
        assert LoadData._safe_int(float("nan")) is None

    def test_garbage_returns_none(self):
        assert LoadData._safe_int("abc") is None


class TestPrepareRows:
    def test_builds_rows_with_hash_and_lineage(self):
        loader = LoadData(source="unit_test_source")
        df = pd.DataFrame(
            {
                "content": ["great app", "bad app"],
                "score": [5, 1],
                "thumbsUpCount": [10, 3],
                "label": [2, 0],
            }
        )
        batch_id = uuid.uuid4()

        rows = loader._prepare_rows(df, batch_id)

        assert len(rows) == 2
        first = rows[0]
        assert first["content"] == "great app"
        assert first["score"] == 5
        assert first["thumbs_up_count"] == 10
        assert first["label"] == 2
        assert first["content_hash"] == _hash_content("great app")
        assert first["source"] == "unit_test_source"
        assert first["ingestion_batch_id"] == batch_id

    def test_skips_blank_content_rows(self):
        loader = LoadData()
        df = pd.DataFrame(
            {"content": ["good", "   ", None], "score": [5, 1, 3]}
        )

        rows = loader._prepare_rows(df, uuid.uuid4())

        assert len(rows) == 1
        assert rows[0]["content"] == "good"


class TestLoadDataAsync:
    def test_empty_dataframe_short_circuits_without_db(self):
        loader = LoadData()

        summary = asyncio.run(loader.load_data_async(pd.DataFrame()))

        assert summary == {"batch_id": None, "attempted": 0, "inserted": 0}

    def test_sync_wrapper_empty_dataframe(self):
        loader = LoadData()

        summary = loader.load_data(pd.DataFrame())

        assert summary["attempted"] == 0
        assert summary["inserted"] == 0
