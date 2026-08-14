"""
Train the model (Linear SVC) and track the run using MLflow.
"""
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

import mlflow
import mlflow.sklearn

from utils.config_loader import settings
from utils.exception import MyException
from utils.logger import logging


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from warnings import filterwarnings

from src.ml.mlflow_setup import init_mlflow_tracking

init_mlflow_tracking()

filterwarnings("ignore")

class ModelTraining:
    """Trains a LinearSVC and tracks parameters, metrics and artifacts with MLflow."""

    DEFAULT_PARAMS = {
        "c": 1.0,
        "tol": 1e-4,
        "loss": "squared_hinge",
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": 1000,
    }

    def __init__(self, X_train, X_test, y_train, y_test, tfidf_vectorizer, track_with_mlflow: bool = True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.tfidf_vectorize = tfidf_vectorizer
        self.track_with_mlflow = track_with_mlflow
        self.run_id = None  # populated after model_training() if tracked

        try:
            params = settings["model_params"]
            self.c = float(params["c"])
            self.tol = float(params["tol"])
            self.loss = str(params["loss"])
            self.fit_intercept = self._to_bool(params["fit_intercept"])
            self.class_weight = params["class_weight"]
            self.max_iter = int(params["max_iter"])
        except (KeyError, TypeError, ValueError) as e:
            logging.warning(
                f"Couldn't fetch the model params: {e}. "
                "Falling back to model default params of sklearn."
            )
            self.c = self.DEFAULT_PARAMS["c"]
            self.tol = self.DEFAULT_PARAMS["tol"]
            self.loss = self.DEFAULT_PARAMS["loss"]
            self.fit_intercept = self.DEFAULT_PARAMS["fit_intercept"]
            self.class_weight = self.DEFAULT_PARAMS["class_weight"]
            self.max_iter = self.DEFAULT_PARAMS["max_iter"]

    @staticmethod
    def _to_bool(value):
        """Coerce a config value into a boolean (handles str/bool/int inputs)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    def _train(self, model: LinearSVC) -> tuple:
        """Fit the model and return (model, metrics, test_pred)."""
        logging.info("Training the model...")
        model.fit(self.X_train, self.y_train)

        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)

        metrics = {
            "train_accuracy": accuracy_score(self.y_train, train_pred),
            "test_accuracy": accuracy_score(self.y_test, test_pred),
            "train_precision": precision_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "test_precision": precision_score(self.y_test, test_pred, average="weighted", zero_division=0),
            "train_recall": recall_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "test_recall": recall_score(self.y_test, test_pred, average="weighted", zero_division=0),
            "train_f1": f1_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "test_f1": f1_score(self.y_test, test_pred, average="weighted", zero_division=0),
        }

        logging.info(f"Train Accuracy : {metrics['train_accuracy']:.4f}")
        logging.info(f"Test Accuracy  : {metrics['test_accuracy']:.4f}")
        logging.info(f"Train Precision: {metrics['train_precision']:.4f}")
        logging.info(f"Test Precision : {metrics['test_precision']:.4f}")
        logging.info(f"Train Recall   : {metrics['train_recall']:.4f}")
        logging.info(f"Test Recall    : {metrics['test_recall']:.4f}")
        logging.info(f"Train F1       : {metrics['train_f1']:.4f}")
        logging.info(f"Test F1        : {metrics['test_f1']:.4f}")

        return model, metrics, test_pred

    def _log_mlflow(self, model: LinearSVC, metrics: dict, test_pred) -> None:
        """Log parameters, metrics, report and the model to MLflow."""
        # =========================
        # Log Parameters
        # =========================
        mlflow.log_param("C", self.c)
        mlflow.log_param("tol", self.tol)
        mlflow.log_param("loss", self.loss)
        mlflow.log_param("max_iter", self.max_iter)
        mlflow.log_param("class_weight", self.class_weight)
        mlflow.log_param("fit_intercept", self.fit_intercept)

        # =========================
        # Log Metrics
        # =========================
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # =========================
        # Classification Report
        # =========================
        report = classification_report(self.y_test, test_pred, zero_division=0)

        # Write the report to a temporary file and log it to MLflow
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
          f.write(report)
          report_path = f.name

        mlflow.log_artifact(report_path)


        #============================
        # Log Fitted TFIDF_Vectorize
        #============================
        mlflow.sklearn.log_model(
           sk_model=self.tfidf_vectorize,
           name="tfidf_vectorizer"
        )

        # =========================
        # Log Model
        # =========================
        mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_svc_model",
        )

        logging.info("Model logged successfully to MLflow.")

    def model_training(self) -> tuple:
        """Train the LinearSVC and return (model, test_f1, run_id)."""
        logging.info("Initialize the Model")

        try:
            model = LinearSVC(
                C=self.c,
                tol=self.tol,
                loss=self.loss,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                fit_intercept=self.fit_intercept,
            )

            model, metrics, test_pred = self._train(model)

            self.run_id = None
            if self.track_with_mlflow:
                logging.info("Starting MLflow tracking")
                with mlflow.start_run() as run:
                    self.run_id = run.info.run_id
                    self._log_mlflow(model, metrics, test_pred)
            else:
                logging.info("Skipping MLflow tracking (track_with_mlflow=False)")

            return model, metrics["test_f1"], self.run_id

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise MyException(e, sys)




# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        logging.info("Initializing CLI Test for ModelTraining")

        sample_df = pd.DataFrame(
            {
                "content": [
                    "machine learning is great and machine learning is fun",
                    "natural language processing and machine learning",
                    "tf idf vectorizer transforms text data into numbers",
                    "text data processing is essential for machine learning",
                    "machine learning and text processing",
                    "data science is fun and great",
                    "natural language and text data",
                    "vectorizer transforms text into numbers",
                    "machine learning models need enough text data",
                    "processing text with vectorizers is common",
                    "natural language data is very useful",
                    "great apps are simple and fun to use",
                    "text data drives modern machine learning",
                    "vectorizers turn text data into numbers",
                    "language processing helps great apps",
                    "data science is a great field",
                ],
                "score": [5, 1, 4, 2, 5, 1, 4, 2, 5, 1, 4, 2, 5, 1, 4, 2],
            }
        )

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            sample_df["content"],
            sample_df["score"],
            test_size=0.25,
            random_state=42,
            stratify=sample_df["score"],
        )

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        trainer = ModelTraining(
            X_train_vec, X_test_vec, y_train, y_test,tfidf_vectorizer=vectorizer,track_with_mlflow=True
        )
        model, test_f1, run_id = trainer.model_training()

        test_pred = model.predict(X_test_vec)
        print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"MLflow run_id: {run_id}")

        logging.info("CLI Test completed without errors.")
    except Exception as e:
        print(f"Test Failed: {e}")