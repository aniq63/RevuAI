"""
Database models for RevuAI.

This defines the append-only "datalake" table that stores every review
that has ever passed through the ETL pipeline. We NEVER update or delete
rows here -- every pipeline run only INSERTs new rows. That gives us a
full historical record to (re)train models on for continuous learning,
and lets us tell batches apart later using `ingestion_batch_id`.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from database.connection import Base


class ReviewRecord(Base):
    """
    Append-only datalake table for raw + lightly transformed reviews.

    Notes on the append-only design:
      - `id` is a plain surrogate primary key (auto-incrementing), so every
        insert always succeeds as a brand-new row.
      - `content_hash` + `source` has a UNIQUE constraint so that if the
        exact same review is ingested twice (e.g. the ingestion job
        re-reads overlapping data), we don't create infinite duplicate
        rows -- but we still never UPDATE or DELETE an existing row.
      - `ingestion_batch_id` groups all rows that came in together in one
        ETL run, which is handy for auditing / incremental training.
      - `ingested_at` gives us a natural time axis for "continuous
        learning" (e.g. "train on everything ingested since last week").
    """

    __tablename__ = "reviews_datalake"
    __table_args__ = (
        UniqueConstraint("content_hash", "source", name="uq_reviews_content_source"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Business columns (mirrors the source CSV schema: content, score,
    # thumbsUpCount, label)
    content = Column(Text, nullable=False)
    score = Column(Integer, nullable=True)
    thumbs_up_count = Column(Integer, nullable=True)
    label = Column(String(32), nullable=True)

    # Dedup key so re-running ingestion on overlapping data doesn't
    # duplicate rows, while still keeping the table strictly append-only
    # (no updates -- a duplicate insert is simply skipped).
    content_hash = Column(String(64), nullable=False, index=True)

    # Lineage / provenance columns
    source = Column(String(128), nullable=False, default="app_reviews_labeled.csv")
    ingestion_batch_id = Column(
        UUID(as_uuid=True), nullable=False, default=uuid.uuid4, index=True
    )
    ingested_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ReviewRecord id={self.id} label={self.label} batch={self.ingestion_batch_id}>"
