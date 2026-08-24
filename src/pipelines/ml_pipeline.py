"""
ML Pipeline
- Data Ingestion
- Feature Transformation (TF-IDF)
- Model Training & Evaluation
- Model Registry for Production
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ml.data_ingestion import DataIngestion
from src.ml.data_split import DataSplit
from src.ml.features import FeatureTransformation
from src.ml.train import ModelTraining
from src.ml.registry import ModelRegistry

from utils.logger import logging
from utils.exception import MyException
from utils.config_loader import settings


class MLPipeline:
    """
    Data Ingestion -> Feature Transformation -> Model Training & Evaluation -> Model Registry

    Chains the ML steps into a single pipeline and returns the trained model
    along with the test F1 score and the MLflow run id.
    """
    def __call__(self, sample_size : int = 35000):
        self.sample_size = sample_size

    async def run_ml_pipeline(self) -> tuple:
        """Run the ML Pipeline and return (model, test_f1, run_id)."""
        logging.info("Start the ML Pipeline")

        try:
            # ======================
            # Data Ingestion
            # ======================
            logging.info("Start the Data ingestion from the data lake")
            data_ingest = DataIngestion(sample_size=self.sample_size)
            df = await data_ingest.fetch_latest_async()
            logging.info("Data Ingestion Completed")

            # ======================
            # Data Splitting
            # ======================
            logging.info("Splitting the data...")
            splitter = DataSplit(label_column="label")
            X_train, X_test, y_train, y_test = splitter.data_split(df=df)
            logging.info("Data Splitting Completed")

            # ======================
            # Feature Transformation
            # ======================
            logging.info("Start Feature Transformation...")
            transform_obj = FeatureTransformation()
            X_train_vec, X_test_vec, vectorizer = transform_obj.apply_tf_idf(
                X_train, X_test
            )
            logging.info("Feature Transformation Completed")

            # ======================
            # Model Training
            # ======================
            logging.info("Start the Model Training, Evaluation and Tracking Pipeline")
            trainer = ModelTraining(
                X_train_vec,
                X_test_vec,
                y_train,
                y_test,
                tfidf_vectorizer=vectorizer,
                track_with_mlflow=True,
            )
            model, test_f1, run_id = trainer.model_training()

            logging.info("Model Training, Evaluation and Tracking Completed")

            # ======================
            # Model Registry
            # ======================
            logging.info("Model Registry for Production Pipeline start ...")
            try:
                always_promote = bool(
                    settings.get("mlflow", {}).get("always_promote", False)
                )
            except Exception:
                always_promote = False
            try:
                raw_min_f1 = settings.get("mlflow", {}).get("min_test_f1")
                min_test_f1 = float(raw_min_f1) if raw_min_f1 is not None else None
            except Exception:
                min_test_f1 = None
            registry = ModelRegistry(
                run_id=run_id,
                test_f1=test_f1,
                always_promote=always_promote,
                min_test_f1=min_test_f1,
            )
            version = registry.register()
            logging.info(
                f"Model Registry for Production is Completed. "
                f"Registered version: {version.version}"
            )

            logging.info(
                f"ML Pipeline completed successfully. "
                f"Test F1: {test_f1:.4f} | run_id: {run_id} | "
                f"version: {version.version}"
            )

            return model, test_f1, run_id

        except Exception as e:
            logging.error(f"Error during ML pipeline execution: {e}")
            raise MyException(e, sys)


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    import asyncio

    try:
        pipeline = MLPipeline()
        model, test_f1, run_id = asyncio.run(pipeline.run_ml_pipeline())
        print(f"Model trained. Test F1: {test_f1:.4f} | run_id: {run_id}")
    except MyException as e:
        print(f"ML Pipeline failed: {e}")
