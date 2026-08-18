"""
Data ingestion for model training.

Fetches the latest rows from the append-only `reviews_datalake`
table and returns them as a pandas DataFrame. This is the ML side of the
loop -- ETL keeps growing the datalake, this pulls a training chunk out.
"""

import asyncio
import sys
from pathlib import Path
from IPython.display import display

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from sqlalchemy import select

from utils.logger import logging
from utils.exception import MyException
from database.connection import AsyncSessionLocal
from database.models import ReviewRecord


class DataIngestion:
    """Fetches the latest rows from the datalake into a pandas DataFrame."""

    def __init__(self, sample_size: int = 35000):
        self.sample_size = sample_size

    @staticmethod
    def _to_dataframe(records: list) -> pd.DataFrame:
        """Convert ORM records into a pandas DataFrame."""
        columns = [col.name for col in ReviewRecord.__table__.columns]
        rows = [
            {col: getattr(record, col) for col in columns}
            for record in records
        ]
        return pd.DataFrame(rows, columns=columns)

    async def fetch_latest_async(self) -> pd.DataFrame:
        """
        Fetch the latest `sample_size` rows from the datalake and
        return them as a pandas DataFrame.

        Uses Postgres `ORDER BY id DESC LIMIT n` so every pull returns
        the most recently ingested rows of the growing historical record.
        """
        try:
            stmt = (
                select(ReviewRecord)
                .order_by(ReviewRecord.id.desc())
                .limit(self.sample_size)
            )

            async with AsyncSessionLocal() as session:
                result = await session.execute(stmt)
                records = result.scalars().all()

            df = self._to_dataframe(records)
            logging.info(
                f"Ingested {df.shape[0]} latest rows from the datalake "
                f"(requested {self.sample_size}). Shape: {df.shape}."
            )
            return df[['id', 'content', 'score', 'label', 'thumbs_up_count']]

        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise MyException(e, sys)

    def fetch_latest(self) -> pd.DataFrame:
        """Synchronous convenience wrapper around `fetch_latest_async`."""
        return asyncio.run(self.fetch_latest_async())


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        ingester = DataIngestion()
        df = ingester.fetch_latest()
        display(df.head())
        print("Shape:", df.shape)
        print("Columns",df.columns)
    except MyException as e:
        print(f"Ingestion failed: {e}")
