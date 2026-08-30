"""
End-to-end ETL pipeline.

Steps:
    1. Scrape   -> pull raw reviews from the Google Play Store
                   (2 random apps, 1 random fetch strategy each).
    2. Label    -> run each review through the sentiment model
                   (Positive / Negative / Neutral).
    3. Transform-> clean/normalize text, encode labels, batch process.
    4. Load     -> append the final DataFrame into the Supabase datalake.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd

from utils.logger import logging
from utils.exception import MyException

from src.etl.scrapping import Reviews_Scrapping_Data
from src.etl.transform import TransformData
from src.etl.load import LoadData

from datasource.data_labeling.data_label import predict_sentiment_batch


class ETLProPipeline:
    """
    Orchestrates the full Scrape -> Label -> Transform -> Load pipeline.
    """

    def __init__(self, source: str = "app_reviews_labeled.csv"):
        self.transformer = TransformData()
        self.loader = LoadData(source=source)

    def run(self) -> dict:
        """
        Executes the full ETL pipeline end-to-end.

        Returns
        -------
        dict
            Summary of the load step: {"batch_id", "attempted", "inserted"}.
        """
        try:
            # ===============================
            # Step 1: Scrape
            # ===============================
            logging.info("ETL PIPELINE | Step 1: Scraping reviews...")

            raw_df = Reviews_Scrapping_Data.main()

            if raw_df is None or raw_df.empty:
                logging.warning("Scraping returned no data. Aborting pipeline.")
                return {"batch_id": None, "attempted": 0, "inserted": 0}

            logging.info(
                f"ETL PIPELINE | Step 1 complete. Scraped {raw_df.shape[0]} reviews."
            )

            # ===============================
            # Step 2: Label
            # ===============================
            logging.info("ETL PIPELINE | Step 2: Labeling reviews with sentiment model...")

            labels = predict_sentiment_batch(raw_df["content"].tolist())
            raw_df["label"] = [label.lower() for label in labels]

            logging.info("ETL PIPELINE | Step 2 complete. Labeling finished.")

            # ===============================
            # Step 3: Transform
            # ===============================
            logging.info("ETL PIPELINE | Step 3: Transforming/cleaning data...")

            transformed_df = self.transformer.data_transformation(raw_df)

            logging.info(
                f"ETL PIPELINE | Step 3 complete. "
                f"Transformed data shape: {transformed_df.shape}"
            )

            # ===============================
            # Step 4: Load
            # ===============================
            logging.info("ETL PIPELINE | Step 4: Loading data into Supabase datalake...")

            summary = self.loader.load_data(transformed_df)

            logging.info(f"ETL PIPELINE | Step 4 complete. Load summary: {summary}")

            logging.info("ETL PIPELINE | Pipeline finished successfully.")

            return summary

        except Exception as e:
            logging.error("ETL PIPELINE | Pipeline failed.")
            raise MyException(e, sys) from e


def main():
    pipeline = ETLProPipeline()
    return pipeline.run()


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        result = main()
        print("\n=== ETL Pipeline Summary ===")
        print(result)
    except MyException as e:
        print(f"Pipeline failed: {e}")