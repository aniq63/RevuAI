"""
Tests for the ML layer.

Covers the main methods:
    - DataSplit.data_split
    - FeatureTransformation.apply_tf_idf
    - ModelTraining.model_training (MLflow tracking disabled)
    - ModelRegistry.register (against a fake MlflowClient)
    - DataIngestion._to_dataframe
"""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.exception import MyException
from src.ml.data_split import DataSplit
from src.ml.features import FeatureTransformation
from src.ml.train import ModelTraining
from src.ml.registry import ModelRegistry


# ==============================================
# Shared helpers / fixtures
# ==============================================
def _make_labelled_df(n_per_class: int = 5) -> pd.DataFrame:
    """Balanced content/label DataFrame big enough for a stratified split."""
    rows = []
    for label, text in [(2, "great amazing love this app"), (0, "terrible broken crashes"), (1, "average okay normal app")]:
        for _ in range(n_per_class):
            rows.append({"content": text, "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def vectorized_data():
    """Small train/test matrices ready for ModelTraining."""
    train_texts = [
        "love this app amazing great experience",
        "best app ever wonderful and beautiful",
        "terrible app crashes constantly every day",
        "worst experience bugs everywhere broken",
        "it is okay average nothing special really",
        "normal app does its job as expected",
        "amazing love it great work team",
        "broken crashes bugs terrible horrible",
    ]
    test_texts = [
        "wonderful amazing app works great",
        "crashes and bugs terrible experience",
        "an average okay kind of app",
    ]
    y_train = [2, 2, 0, 0, 1, 1, 2, 0]
    y_test = [2, 0, 1]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, y_train, y_test, vectorizer


# ==============================================
# DATA SPLIT
# ==============================================
class TestDataSplit:
    def test_split_shapes_are_consistent(self):
        df = _make_labelled_df()
        X_train, X_test, y_train, y_test = DataSplit().data_split(df)

        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_test_size_respected(self):
        df = _make_labelled_df()
        splitter = DataSplit()
        _, X_test, _, _ = splitter.data_split(df)

        assert abs(len(X_test) / len(df) - splitter.test_size) < 0.05

    def test_stratification_preserves_class_proportions(self):
        df = _make_labelled_df(n_per_class=10)
        _, _, y_train, y_test = DataSplit().data_split(df)

        train_props = y_train.value_counts(normalize=True).sort_index()
        test_props = y_test.value_counts(normalize=True).sort_index()

        pd.testing.assert_series_equal(train_props, test_props, check_names=False)

    def test_rare_class_falls_back_to_plain_split(self):
        # Only one 'negative' row -> stratify is impossible, must not raise.
        df = pd.DataFrame(
            {
                "content": ["good"] * 9 + ["bad"],
                "label": [2] * 9 + [0],
            }
        )
        X_train, X_test, y_train, y_test = DataSplit().data_split(df)

        assert len(X_train) + len(X_test) == 10

    def test_empty_dataframe_raises(self):
        with pytest.raises(MyException):
            DataSplit().data_split(pd.DataFrame())

    def test_missing_target_column_raises(self):
        with pytest.raises(MyException):
            DataSplit().data_split(pd.DataFrame({"label": [0, 1]}))

    def test_missing_label_column_raises(self):
        with pytest.raises(MyException):
            DataSplit().data_split(pd.DataFrame({"content": ["a", "b"]}))


# ==============================================
# FEATURE TRANSFORMATION (TF-IDF)
# ==============================================
class TestFeatureTransformation:
    def test_returns_matrices_and_fitted_vectorizer(self):
        train = pd.Series(
            [
                "machine learning is great",
                "text data processing helps",
                "vectorizer transforms text data",
                "machine learning models need data",
                "processing text with vectorizers",
                "great apps use machine learning",
            ]
        )
        test = pd.Series(["machine learning text"])

        X_train_vec, X_test_vec, vectorizer = FeatureTransformation().apply_tf_idf(train, test)

        assert X_train_vec.shape[0] == 6
        assert X_test_vec.shape[0] == 1
        assert hasattr(vectorizer, "vocabulary_")
        assert len(vectorizer.vocabulary_) > 0

    def test_train_and_test_share_vocabulary(self):
        train = pd.Series(
            [
                "alpha beta gamma",
                "beta gamma delta",
                "alpha delta epsilon",
                "gamma epsilon zeta",
                "alpha beta zeta",
                "delta epsilon eta",
            ]
        )
        test = pd.Series(["alpha delta epsilon unknownword"])

        _, X_test_vec, vectorizer = FeatureTransformation().apply_tf_idf(train, test)

        # Test matrix columns must match the TRAIN-fitted vocabulary,
        # unseen words are ignored.
        assert X_test_vec.shape[1] == len(vectorizer.vocabulary_)

    def test_invalid_input_raises_my_exception(self):
        with pytest.raises(MyException):
            FeatureTransformation().apply_tf_idf(None, None)


# ==============================================
# MODEL TRAINING (no MLflow)
# ==============================================
class TestModelTraining:
    def test_training_returns_model_metrics_and_no_run_id(
        self, vectorized_data
    ):
        X_train, X_test, y_train, y_test, vectorizer = vectorized_data

        trainer = ModelTraining(
            X_train,
            X_test,
            y_train,
            y_test,
            tfidf_vectorizer=vectorizer,
            track_with_mlflow=False,
        )
        model, test_f1, run_id = trainer.model_training()

        assert run_id is None
        assert 0.0 <= test_f1 <= 1.0
        assert hasattr(model, "predict")

    def test_model_predicts_correct_shape(self, vectorized_data):
        X_train, X_test, y_train, y_test, vectorizer = vectorized_data

        trainer = ModelTraining(
            X_train,
            X_test,
            y_train,
            y_test,
            tfidf_vectorizer=vectorizer,
            track_with_mlflow=False,
        )
        model, _, _ = trainer.model_training()

        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_perfectly_separable_data_scores_high(self, vectorized_data):
        X_train, X_test, y_train, y_test, vectorizer = vectorized_data

        trainer = ModelTraining(
            X_train,
            X_test,
            y_train,
            y_test,
            tfidf_vectorizer=vectorizer,
            track_with_mlflow=False,
        )
        _, test_f1, _ = trainer.model_training()

        assert test_f1 > 0.5


# ==============================================
# MODEL REGISTRY (fake MlflowClient)
# ==============================================
class FakeVersion:
    def __init__(self, version: str, run_id: str):
        self.version = version
        self.run_id = run_id


class FakeRun:
    def __init__(self, run_id: str, test_f1: float):
        self.info = SimpleNamespace(run_id=run_id)
        self.data = SimpleNamespace(metrics={"test_f1": test_f1}, tags={})


class FakeMlflowClient:
    """In-memory stand-in covering the registry API surface."""

    def __init__(self):
        self.models = {}  # name -> {"versions": {v: run_id}, "aliases": {alias: v}}
        self.runs = {}  # run_id -> test_f1
        self._next_version = 1

    def seed_run(self, run_id: str, test_f1: float):
        self.runs[run_id] = test_f1

    # --- API used by ModelRegistry ---
    def get_registered_model(self, name):
        if name not in self.models:
            raise Exception(f"Registered model {name} not found")
        return SimpleNamespace(name=name)

    def create_registered_model(self, name):
        self.models[name] = {"versions": {}, "aliases": {}}

    def create_model_version(self, name, source, run_id):
        version = str(self._next_version)
        self._next_version += 1
        self.models[name]["versions"][version] = run_id
        return FakeVersion(version, run_id)

    def set_registered_model_alias(self, name, alias, version):
        self.models[name]["aliases"][alias] = version

    def get_model_version_by_alias(self, name, alias):
        version = self.models[name]["aliases"][alias]
        return FakeVersion(version, self.models[name]["versions"][version])

    def get_run(self, run_id):
        return FakeRun(run_id, self.runs[run_id])


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeMlflowClient()
    monkeypatch.setattr("src.ml.registry.MlflowClient", lambda: client)
    return client


def _seed_champion(client: FakeMlflowClient, f1: float = 0.80):
    """Pre-register a Champion at version 1."""
    client.seed_run("champion_run", f1)
    registry = ModelRegistry(run_id="champion_run", test_f1=f1)
    registry.register()


class TestModelRegistry:
    def test_first_model_becomes_champion(self, fake_client):
        fake_client.seed_run("run_1", 0.85)
        registry = ModelRegistry(run_id="run_1", test_f1=0.85)

        version, champion_updated = registry.register()

        assert version.version == "1"
        assert champion_updated is True
        champion = registry.client.get_model_version_by_alias(
            registry.model_name, "Champion"
        )
        assert champion.version == "1"
        assert champion.run_id == "run_1"

    def test_better_model_is_promoted(self, fake_client):
        _seed_champion(fake_client, f1=0.80)
        fake_client.seed_run("run_2", 0.90)

        registry = ModelRegistry(run_id="run_2", test_f1=0.90)
        version, champion_updated = registry.register()

        assert version.version == "2"
        assert champion_updated is True
        champion = registry.client.get_model_version_by_alias(
            registry.model_name, "Champion"
        )
        assert champion.version == "2"

    def test_worse_model_keeps_current_champion(self, fake_client):
        _seed_champion(fake_client, f1=0.90)
        fake_client.seed_run("run_2", 0.70)

        registry = ModelRegistry(run_id="run_2", test_f1=0.70)
        version, champion_updated = registry.register()

        assert version.version == "1"  # still the old champion
        assert champion_updated is False

    def test_below_quality_gate_never_promotes(self, fake_client):
        _seed_champion(fake_client, f1=0.80)
        fake_client.seed_run("run_2", 0.95)

        # Beats the champion but sits below the configured floor.
        registry = ModelRegistry(
            run_id="run_2", test_f1=0.95, min_test_f1=0.99
        )
        version, champion_updated = registry.register()

        assert version.version == "1"
        assert champion_updated is False
        assert "2" not in fake_client.models[registry.model_name]["versions"]

    def test_always_promote_replaces_champion(self, fake_client):
        _seed_champion(fake_client, f1=0.90)
        fake_client.seed_run("run_2", 0.50)

        registry = ModelRegistry(
            run_id="run_2", test_f1=0.50, always_promote=True
        )
        version, champion_updated = registry.register()

        assert version.version == "2"
        assert champion_updated is True

    def test_first_model_below_gate_raises(self, fake_client):
        fake_client.seed_run("run_1", 0.10)

        registry = ModelRegistry(run_id="run_1", test_f1=0.10, min_test_f1=0.5)

        with pytest.raises(MyException):
            registry.register()

    # ------------------------------------------------------------------
    # New tests: champion_updated signal for every decision branch
    # ------------------------------------------------------------------

    def test_equal_f1_does_not_update_champion(self, fake_client):
        """new_f1 == champion_f1 → no promotion → champion_updated=False."""
        _seed_champion(fake_client, f1=0.80)
        fake_client.seed_run("run_2", 0.80)

        registry = ModelRegistry(run_id="run_2", test_f1=0.80)
        version, champion_updated = registry.register()

        assert champion_updated is False
        assert version.version == "1"  # old champion unchanged

    def test_no_champion_first_model_champion_updated_true(self, fake_client):
        """No Champion exists → first model registered → champion_updated=True."""
        fake_client.seed_run("first_run", 0.75)

        registry = ModelRegistry(run_id="first_run", test_f1=0.75)
        _, champion_updated = registry.register()

        assert champion_updated is True

    def test_min_test_f1_blocks_promotion_champion_updated_false(self, fake_client):
        """min_test_f1 blocks even a better model → champion_updated=False."""
        _seed_champion(fake_client, f1=0.50)
        fake_client.seed_run("run_2", 0.60)  # better than champion

        registry = ModelRegistry(
            run_id="run_2", test_f1=0.60, min_test_f1=0.70  # floor above both
        )
        _, champion_updated = registry.register()

        assert champion_updated is False

    def test_always_promote_champion_updated_true(self, fake_client):
        """always_promote=True forces promotion → champion_updated=True."""
        _seed_champion(fake_client, f1=0.99)
        fake_client.seed_run("run_2", 0.10)  # much worse, but always_promote

        registry = ModelRegistry(
            run_id="run_2", test_f1=0.10, always_promote=True
        )
        _, champion_updated = registry.register()

        assert champion_updated is True


# ==============================================
# DATA INGESTION
# ==============================================
class TestDataIngestion:
    def _fake_record(self, idx: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=idx,
            content=f"review number {idx}",
            score=5,
            thumbs_up_count=3,
            label=2,
            content_hash=uuid.uuid4().hex,
            source="unit_test",
            ingestion_batch_id=uuid.uuid4(),
            ingested_at=datetime(2026, 1, 1),
        )

    def test_to_dataframe_maps_all_table_columns(self):
        from src.ml.data_ingestion import DataIngestion
        from database.models import ReviewRecord

        records = [self._fake_record(1), self._fake_record(2)]
        df = DataIngestion._to_dataframe(records)

        expected_cols = [col.name for col in ReviewRecord.__table__.columns]
        assert list(df.columns) == expected_cols
        assert df.shape == (2, len(expected_cols))
        assert df.iloc[0]["content"] == "review number 1"

    def test_empty_records_give_empty_dataframe_with_columns(self):
        from src.ml.data_ingestion import DataIngestion
        from database.models import ReviewRecord

        df = DataIngestion._to_dataframe([])

        expected_cols = [col.name for col in ReviewRecord.__table__.columns]
        assert list(df.columns) == expected_cols
        assert df.empty

    def test_default_sample_size(self):
        from src.ml.data_ingestion import DataIngestion

        assert DataIngestion().sample_size == 35000
        assert DataIngestion(sample_size=100).sample_size == 100
