"""
Load step of the ETL pipeline.

Loads a (transformed) DataFrame of reviews into Supabase Postgres, which we
use as an append-only "datalake" for continuous learning.
"""

import asyncio
import hashlib
import sys
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from utils.logger import logging
from utils.exception import MyException
from database.connection import AsyncSessionLocal, init_db
from database.models import ReviewRecord


def _hash_content(content: str) -> str:
    """Stable hash used purely for de-duplication, not security."""
    return hashlib.sha256(str(content).strip().lower().encode("utf-8")).hexdigest()


class LoadData:
    """
    Handles writing DataFrames into the Supabase append-only datalake.
    """

    def __init__(self, source: str = "app_reviews_labeled.csv", batch_size: int = 500):
        self.source = source
        self.batch_size = batch_size

    def _prepare_rows(self, df: pd.DataFrame, batch_id: uuid.UUID) -> list:
        """Convert a DataFrame into a list of dicts ready for bulk insert."""
        records = []
        for row in df.to_dict(orient="records"):
            content = row.get("content")
            if content is None or str(content).strip() == "":
                continue

            records.append(
                {
                    "content": str(content),
                    "score": self._safe_int(row.get("score")),
                    "thumbs_up_count": self._safe_int(row.get("thumbsUpCount")),
                        "label": self._safe_label(row.get("label")),
                    "content_hash": _hash_content(content),
                    "source": self.source,
                    "ingestion_batch_id": batch_id,
                }
            )
        return records

    @staticmethod
    def _safe_int(value):
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _safe_label(cls, value):
        """Normalize sentiment labels to the integer codes used by the schema."""
        if isinstance(value, str):
            label_codes = {"negative": 0, "neutral": 1, "positive": 2}
            normalized = value.strip().lower()
            if normalized in label_codes:
                return label_codes[normalized]
        return cls._safe_int(value)

    async def load_data_async(self, df: pd.DataFrame, batch_id: Optional[uuid.UUID] = None) -> dict:
        """
        Append `df` into the `reviews_datalake` table in Supabase.

        Returns a small summary dict: {"batch_id", "attempted", "inserted"}.
        Duplicate rows (same content_hash + source) are skipped via
        ON CONFLICT DO NOTHING -- this is still a pure append operation,
        it just avoids inserting the exact same review twice.
        """
        try:
            if df is None or df.empty:
                logging.warning("load_data_async received an empty DataFrame. Nothing to load.")
                return {"batch_id": None, "attempted": 0, "inserted": 0}

            batch_id = batch_id or uuid.uuid4()
            logging.info(f"Starting load into Supabase datalake. batch_id={batch_id}")

            # Make sure the table exists (safe/no-op if it already does).
            await init_db()

            records = self._prepare_rows(df, batch_id)
            total = len(records)
            inserted = 0

            async with AsyncSessionLocal() as session:
                for start in range(0, total, self.batch_size):
                    chunk = records[start : start + self.batch_size]
                    if not chunk:
                        continue

                    stmt = pg_insert(ReviewRecord).values(chunk)
                    # APPEND-ONLY: on duplicate content, do nothing (never UPDATE).
                    stmt = stmt.on_conflict_do_nothing(
                        constraint="uq_reviews_content_source"
                    )
                    result = await session.execute(stmt)
                    inserted += result.rowcount or 0

                    logging.info(
                        f"Inserted batch rows {start} to {start + len(chunk)} "
                        f"(cumulative new rows inserted: {inserted})."
                    )

                await session.commit()

            logging.info(
                f"Load complete. batch_id={batch_id} attempted={total} "
                f"newly_inserted={inserted} skipped_duplicates={total - inserted}."
            )
            return {"batch_id": str(batch_id), "attempted": total, "inserted": inserted}

        except Exception as e:
            logging.error("An error occurred during data loading into Supabase.")
            raise MyException(e, sys)

    def load_data(self, df: pd.DataFrame, batch_id: Optional[uuid.UUID] = None) -> dict:
        """Synchronous convenience wrapper around `load_data_async`."""
        return asyncio.run(self.load_data_async(df, batch_id=batch_id))


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        sample_df = pd.DataFrame(
            {
                "content": [
                    "Great app, works flawlessly!",
                    "Crashes every time I open it.",
                ],
                "score": [5, 1],
                "thumbsUpCount": [10, 3],
                "label": ["positive", "negative"],
            }
        )

        loader = LoadData()
        summary = loader.load_data(sample_df)
        print("Load summary:", summary)
    except MyException as e:
        print(f"Load failed: {e}")
