"""
End-to-end ETL orchestration for RevuAI.

Flow:
    1. EXTRACT -> ExtractData.data_extraction()  reads the full source
                                                   CSV into a DataFrame.
    2. TRANSFORM -> TransformData.data_transformation()
                                                    cleans + stems review
                                                    text.
    3. LOAD    -> LoadData.load_data_async()      appends the batch into
                                                   the Supabase
                                                   `reviews_datalake`
                                                   table. Nothing is ever
                                                   overwritten -- this is
                                                   how we build up a
                                                   growing dataset for
                                                   continuous learning.

Run directly:
    python -m src.pipelines.etl_pipeline
"""

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging
from utils.exception import MyException

from src.etl.extract import ExtractData
from src.etl.transform import TransformData
from src.etl.load import LoadData


class ETLPipeline:
    """Wires extract -> transform -> load into a single runnable pipeline."""

    def __init__(self, load_batch_size: int = 500):
        self.extractor = ExtractData()
        self.transformer = TransformData()
        self.loader = LoadData(batch_size=load_batch_size)

    async def run_async(self) -> dict:
        started_at = time.time()
        logging.info("===== ETL pipeline run started =====")

        try:
            # 1. Extract the full source dataset.
            full_df = self.extractor.data_extraction()

            # 2. Clean / preprocess the text column.
            transformed_df = self.transformer.data_transformation(full_df)

            # 3. Append the transformed batch into Supabase (never overwrite).
            summary = await self.loader.load_data_async(transformed_df)

            elapsed = round(time.time() - started_at, 2)
            logging.info(
                f"===== ETL pipeline run finished in {elapsed}s | "
                f"batch_id={summary.get('batch_id')} "
                f"attempted={summary.get('attempted')} "
                f"inserted={summary.get('inserted')} ====="
            )
            summary["elapsed_seconds"] = elapsed
            return summary

        except MyException:
            # Already logged with full context by the failing step.
            raise
        except Exception as e:
            logging.error("ETL pipeline run failed unexpectedly.")
            raise MyException(e, sys)

    def run(self) -> dict:
        """Synchronous convenience wrapper for `run_async`."""
        return asyncio.run(self.run_async())


# ==============================================
# CLI ENTRYPOINT
# ==============================================
if __name__ == "__main__":
    try:
        pipeline = ETLPipeline()
        result = pipeline.run()
        print("ETL pipeline summary:", result)
    except MyException as e:
        print(f"ETL pipeline failed: {e}")
