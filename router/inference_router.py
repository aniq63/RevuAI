import sys
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

    try:
        logging.info(
            f"Received /analyze request for app_id={data.app_id} "
            f"(scraping_method={data.scraping_method})"
        )

        pipeline = InferencePipeline(
            app_id=data.app_id,
            scraping_method=data.scraping_method,
            prediction_pipeline=prediction_pipeline,
            embedder=embedder,
        )

        result = await pipeline.run_inference_async()
        return result

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