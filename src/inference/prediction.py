"""
Prediction pipeline for RevuAI.

Loads the current "Champion" model and its fitted TF-IDF vectorizer from the
MLflow model registry (hosted on DagsHub), then runs sentiment predictions on
Play Store review text. The output mirrors the numeric labels used during
training (0=negative, 1=neutral, 2=positive).
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.exception import MyException
from utils.logger import logging

from src.etl.transform import TransformData
from src.ml.mlflow_setup import init_mlflow_tracking


@contextmanager
def _neutralize_transformers_lazy_imports():
    """
    MLflow 3.x loads sklearn models saved with the 'skops' serialization
    format. On import, skops' trusted-type discovery walks every module in
    ``sys.modules`` and calls ``getattr()`` on it.

    The ``transformers`` package (a transitive dependency of
    sentence-transformers) implements ``__getattr__`` as a lazy loader, so one
    of those probes triggers an eager import of an optional
    ``torchvision``-dependent module. When torchvision is not installed this
    crashes ``mlflow.sklearn.load_model`` with
    ``ModuleNotFoundError: No module named 'torchvision'``.

    This context manager temporarily replaces the lazy ``__getattr__`` stored
    in the ``transformers`` module (and its loaded submodules) with a strict
    one (raising ``AttributeError``) for the duration of the load, so the probe
    simply skips them. The original loaders are restored afterwards, keeping
    sentence-transformers fully functional.
    """
    touched = []

    def _strict_getattr(attr):
        # PEP 562: a module-level __getattr__ is called with only the
        # attribute name (not bound to the module).
        raise AttributeError(
            f"module has no attribute {attr!r}"
        )

    try:
        for name, module in list(sys.modules.items()):
            if name != "transformers" and not name.startswith("transformers."):
                continue

            original = vars(module).get("__getattr__")
            if original is None:
                continue

            vars(module)["__getattr__"] = _strict_getattr
            touched.append((module, original))

        yield

    finally:
        for module, original in touched:
            vars(module)["__getattr__"] = original


def _load_mlflow_sklearn_model(model_id: str):
    """Load an MLflow-logged sklearn model while shielding the skops
    trusted-type scan from ``transformers``' lazy imports."""
    with _neutralize_transformers_lazy_imports():
        return mlflow.sklearn.load_model(f"models:/{model_id}")


class PredictionPipeline:
    """
    Loads the Champion model + vectorizer once and predicts sentiment labels
    for arbitrary review text.

    Example:
        pipe = PredictionPipeline()
        df = pd.DataFrame({
            "content": ["Great app!", "Crashes all the time"],
            "score": [5, 1],
            "thumbsUpCount": [20, 3],
        })
        df = pipe.predict(df)
        # columns: content, score, thumbsUpCount, label
    """

    MODEL_NAME = "linear_svc_classifier"
    MODEL_ALIAS = "Champion"

    # Numeric labels produced by the model -> human readable sentiment.
    # Mirrors TransformData._sentiment_to_number used during training.
    LABEL_TO_SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model_name: str = None,
        model_alias: str = None,
        model=None,
        vectorizer=None,
    ):
        try:
            logging.info("Initializing Prediction Pipeline")

            self.model_name = model_name or self.MODEL_NAME
            self.model_alias = model_alias or self.MODEL_ALIAS

            # If both are already loaded (e.g. from app startup / lifespan),
            # skip the MLflow round-trip entirely.
            if model is not None and vectorizer is not None:
                self.model = model
                self.vectorizer = vectorizer
                self.transformer = TransformData()
                logging.info(
                    "Using pre-loaded model and vectorizer. "
                    "Skipped MLflow loading."
                )
                return

            # Point the MLflow client at the same DagsHub-hosted
            # tracking + registry server used by training/registration.
            init_mlflow_tracking()

            self.client = MlflowClient()

            # --------------------------------------------------
            # Get Champion model version
            # --------------------------------------------------
            champion = self.client.get_model_version_by_alias(
                self.model_name,
                self.model_alias,
            )

            self.run_id = champion.run_id
            self.model_version = champion.version

            logging.info(f"Champion model version: {self.model_version}")
            logging.info(f"Champion model run_id: {self.run_id}")

            # --------------------------------------------------
            # Retrieve the MLflow 3 Logged Model IDs from tags stored
            # on the Champion run by src/ml/train.py
            # --------------------------------------------------
            champion_run = self.client.get_run(self.run_id)

            model_id = champion_run.data.tags.get("model_id")
            vectorizer_model_id = champion_run.data.tags.get(
                "vectorizer_model_id"
            )

            if not model_id:
                raise ValueError(
                    "model_id tag not found in Champion run"
                )
            if not vectorizer_model_id:
                raise ValueError(
                    "vectorizer_model_id tag not found in Champion run"
                )

            logging.info(f"Champion model_id: {model_id}")
            logging.info(
                f"Champion vectorizer_model_id: {vectorizer_model_id}"
            )

            # --------------------------------------------------
            # Load MLflow 3 Logged Models by their model IDs.
            # This guarantees the vectorizer matches the exact run
            # that produced the Champion model (same fitted vocabulary).
            # --------------------------------------------------
            self.model = _load_mlflow_sklearn_model(model_id)
            self.vectorizer = _load_mlflow_sklearn_model(
                vectorizer_model_id
            )

            # Shared text cleaning pipeline (same as the ETL side).
            self.transformer = TransformData()

            logging.info(
                "Model and TF-IDF vectorizer loaded successfully."
            )

        except Exception as e:
            logging.error(
                f"Error loading model and vectorizer: {e}"
            )
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        data,
        decode_labels: bool = False,
        batch_size: int = 500,
    ) -> pd.DataFrame:
        """
        Predict sentiment labels for review text.

        The data is processed in batches of `batch_size` rows: each batch is
        cleaned (transform), vectorized, and scored end-to-end, which keeps
        memory bounded and is much faster than row-by-row scoring.

        Parameters
        ----------
        data : str | list | tuple | pd.Series | pd.DataFrame
            Text to score. A DataFrame must contain a 'content' column;
            all other columns (e.g. score, thumbsUpCount) are preserved.
        decode_labels : bool, default False
            When True, also add a 'sentiment' column with human readable
            values ('negative' / 'neutral' / 'positive').
        batch_size : int, default 2000
            Number of rows processed per batch.

        Returns
        -------
        pd.DataFrame
            The same input rows/columns with a 'label' column appended.
            Rows whose text became empty after cleaning get a NaN label.
        """
        try:
            logging.info("Starting prediction pipeline")

            df = self._validate_input(data)

            labels = pd.Series(pd.NA, index=df.index, dtype="Int64")

            if df.empty:
                logging.warning("Input DataFrame is empty.")
                return self._build_output(df, labels, decode_labels)

            # --------------------------------------------------
            # Process in batches: transform -> vectorize -> predict
            # --------------------------------------------------
            total_valid = 0
            for start in range(0, len(df), batch_size):
                chunk = df.iloc[start : start + batch_size].copy()
                chunk["__review_index__"] = np.arange(
                    start, start + len(chunk)
                )

                transformed = self.transformer.data_transformation(
                    chunk[["content", "__review_index__"]]
                )

                if transformed.empty:
                    logging.warning(
                        f"All rows in batch {start} to "
                        f"{start + len(chunk)} were dropped during "
                        "text cleaning."
                    )
                    continue

                predictions = self._predict(transformed["content"])
                labels.iloc[
                    transformed["__review_index__"].to_numpy()
                ] = predictions

                total_valid += len(predictions)
                logging.info(
                    f"Batch {start} to {start + len(chunk)} "
                    f"predicted ({len(predictions)} valid)."
                )

            logging.info(
                f"Prediction completed for {len(df)} records "
                f"({total_valid} valid)."
            )

            return self._build_output(df, labels, decode_labels)

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise MyException(e, sys)

    def predict_text(self, text, decode_labels: bool = True):
        """
        Convenience wrapper for scoring a single string or a list of strings.

        Returns a single value for a string input, otherwise a list.
        """
        frame = pd.DataFrame(
            {"content": [text] if isinstance(text, str) else list(text)}
        )
        result = self.predict(frame, decode_labels=decode_labels)

        column = "sentiment" if decode_labels else "label"
        values = result[column].tolist()

        return values[0] if len(values) == 1 else values

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(data) -> pd.DataFrame:
        """Normalize flexible inputs into a non-empty-proof DataFrame."""
        if isinstance(data, str):
            data = pd.DataFrame({"content": [data]})
        elif isinstance(data, (list, tuple)):
            data = pd.DataFrame({"content": list(data)})
        elif isinstance(data, pd.Series):
            data = pd.DataFrame({"content": data.tolist()})
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a string, list, tuple, pandas Series, "
                "or pandas DataFrame."
            )

        if "content" not in data.columns:
            raise ValueError(
                "Input DataFrame must contain a 'content' column."
            )

        df = data.copy().reset_index(drop=True)
        df = df[df["content"].notna()].reset_index(drop=True)

        return df

    @staticmethod
    def _build_output(df, labels, decode_labels: bool) -> pd.DataFrame:
        """Attach the prediction columns to the original DataFrame."""
        result = df.copy()
        result["label"] = labels

        if decode_labels:
            result["sentiment"] = (
                result["label"]
                .map(PredictionPipeline.LABEL_TO_SENTIMENT)
                .fillna("unknown")
            )

        return result

    def _predict(self, content: pd.Series) -> np.ndarray:
        """Vectorize a batch of cleaned text and return model predictions."""
        X = self.vectorizer.transform(content)
        return self.model.predict(X)





# ==============================================
# CLI TESTING
# ==============================================

if __name__ == "__main__":
    import time

    try:
        start = time.perf_counter()

        pipeline = PredictionPipeline()

        sample = pd.DataFrame({
            "content": [
                "Amazing app!",
                "Too many bugs and crashes every day.",
                "It is okay, nothing special.",
                "Beautiful app",
                "Absolutely love this! Best user experience ever.",
                "The recent update completely broke the login screen.",
                "It does what it says, but the UI could be better.",
                "Highly recommended! Saves me so much time daily.",
                "Total waste of time. It freezes constantly on my phone.",
                "Just downloaded it. It works fine for now.",
                "Incredibly fast and very intuitive to navigate.",
                "Extremely disappointed. Terrible customer support.",
                "An average application, standard features like others.",
                "Perfect tool! I cannot imagine my routine without it."
            ],
            "score": [
                5, 1, 3, 3,
                5, 1, 3, 5, 1, 3, 5, 1, 3, 5
            ],
            "thumbsUpCount": [
                20, 3, 8, 10,
                45, 12, 5, 30, 18, 2, 25, 14, 7, 50
            ],
        })

        result = pipeline.predict(
            sample,
            decode_labels=True,
            batch_size=500
        )

        print("\n=== Prediction Output Sample ===")
        print(result)

        print("\nOutput columns:", result.columns.tolist())

        print("\n=== Single Text ===")
        print(pipeline.predict_text("I love this app!"))

        print("\n=== Empty / Emoji-only Text ===")
        print(pipeline.predict_text("😃🎉🚀", decode_labels=False))

        end_time = time.perf_counter()

        execution_time = end_time - start
        print(f"Execution time: {execution_time:.6f} seconds")

    except MyException as e:
        print(f"Prediction failed: {e}")