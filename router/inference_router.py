"""
Router for the /inference/analyze endpoint.

Adds a simple in-memory, TTL-based result cache keyed on
(app_id, scraping_method): if the same app was analyzed recently
(within RESULT_CACHE_TTL_SECONDS, configured in main.py's lifespan), the
cached result is returned immediately instead of re-running the full
scrape -> predict -> cluster -> LLM pipeline. This is usually the single
biggest real-world latency win for a browser-extension use case, since the
same app is very likely to be analyzed more than once in a short window.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal

from utils.exception import MyException
from utils.logger import logging

from src.pipelines.inference_pipeline import InferencePipeline

router = APIRouter(prefix="/inference", tags=["Inference"])


# ==========================================
# Request Schema
# ==========================================
class InferenceRequest(BaseModel):
    app_id: str = Field(
        ...,
        description="Play Store app package id, e.g. 'com.supercell.clashofclans'",
        examples=["com.supercell.clashofclans"],
    )
    scraping_method: Literal[
        "all_reviews",
        "new_reviews",
        "relevant_reviews",
    ] = Field(
        default="new_reviews",
        description="Which reviews to scrape before running inference.",
    )
    force_refresh: bool = Field(
        default=False,
        description=(
            "If true, bypass the result cache and re-run the full "
            "pipeline even if a cached result exists for this app_id."
        ),
    )


def _cache_key(app_id: str, scraping_method: str) -> str:
    return f"{app_id}::{scraping_method}"


# ==========================================
# Endpoint
# ==========================================
@router.post("/analyze")
async def analyze_review(data: InferenceRequest, request: Request):
    """
    Run the full inference pipeline (scrape -> predict -> sentiment
    summary -> topic clustering -> LLM insight) for a given app_id.

    Uses the model, TF-IDF vectorizer, and embedding model that were
    pre-loaded at application startup (see main.py lifespan) instead
    of reloading them on every request.

    Also checks an in-memory TTL cache first: if this app_id +
    scraping_method combination was analyzed within the configured TTL
    window, the cached result is returned immediately (unless
    ``force_refresh`` is set).
    """
    # Fail fast if startup hasn't finished / models aren't loaded yet.
    prediction_pipeline = getattr(request.app.state, "prediction_pipeline", None)
    embedder = getattr(request.app.state, "embedder", None)

    if prediction_pipeline is None or embedder is None:
        logging.error(
            "Inference requested before models finished loading "
            f"(prediction_pipeline={prediction_pipeline is not None}, "
            f"embedder={embedder is not None})."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Please try again shortly.",
        )

    cache: dict = getattr(request.app.state, "result_cache", {})
    cache_ttl: int = getattr(request.app.state, "result_cache_ttl", 0)
    key = _cache_key(data.app_id, data.scraping_method)

    # ----------------------------------------
    # Cache lookup
    # ----------------------------------------
    if not data.force_refresh and key in cache:
        cached_result, expires_at = cache[key]
        if time.time() < expires_at:
            logging.info(
                f"Cache hit for app_id={data.app_id} "
                f"(scraping_method={data.scraping_method}); "
                "returning cached result."
            )
            return {**cached_result, "cache_hit": True}
        else:
            # Expired -- drop it so the dict doesn't grow unbounded with
            # stale entries.
            del cache[key]

    try:
        logging.info(
            f"Received /analyze request for app_id={data.app_id} "
            f"(scraping_method={data.scraping_method}, "
            f"force_refresh={data.force_refresh})"
        )

        pipeline = InferencePipeline(
            app_id=data.app_id,
            scraping_method=data.scraping_method,
            prediction_pipeline=prediction_pipeline,
            embedder=embedder,
        )

        result = await pipeline.run_inference_async()

        # ----------------------------------------
        # Store in cache for subsequent requests
        # ----------------------------------------
        if cache_ttl > 0:
            cache[key] = (result, time.time() + cache_ttl)

        return {**result, "cache_hit": False}

    except MyException as e:
        logging.error(f"Inference pipeline failed for app_id={data.app_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    except Exception as e:
        logging.error(
            f"Unexpected error during inference for app_id={data.app_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while running inference.",
        )


@router.delete("/cache/{app_id}")
async def clear_cached_result(app_id: str, request: Request, scraping_method: str = "new_reviews"):
    """Manually invalidate a cached result for an app_id (e.g. after new reviews come in)."""
    cache: dict = getattr(request.app.state, "result_cache", {})
    key = _cache_key(app_id, scraping_method)
    existed = cache.pop(key, None) is not None
    return {"app_id": app_id, "scraping_method": scraping_method, "cleared": existed}