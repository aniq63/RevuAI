"""
Main FastAPI application entry point for RevuAI.

Uses a `lifespan` context manager to preload heavy, stateless resources
ONCE at application startup:
    - Champion ML model + fitted TF-IDF vectorizer (via PredictionPipeline)
    - Sentence embedding model (SentenceTransformer, used by topic clustering)

These are stored on `app.state` so every request reuses the same in-memory
objects instead of re-loading them from MLflow / disk on every call, which
would otherwise add several seconds of latency per request.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging
from utils.exception import MyException

from src.inference.prediction import PredictionPipeline
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load the Champion model, TF-IDF vectorizer, and the sentence
    embedding model once, and attach them to app.state.

    Shutdown: release references (garbage collected automatically).
    """
    started_at = time.time()
    logging.info("===== Application startup: loading models =====")

    try:
        # --------------------------------------------------
        # 1. Prediction Pipeline (Champion model + vectorizer)
        # --------------------------------------------------
        logging.info("Loading PredictionPipeline (model + TF-IDF vectorizer)...")
        prediction_pipeline = PredictionPipeline()
        app.state.prediction_pipeline = prediction_pipeline
        logging.info("PredictionPipeline loaded successfully.")

        # --------------------------------------------------
        # 2. Sentence Embedding Model (shared with topic clustering)
        # --------------------------------------------------
        logging.info(f"Loading SentenceTransformer('{EMBEDDING_MODEL_NAME}')...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        app.state.embedder = embedder
        logging.info("Embedding model loaded successfully.")

        elapsed = round(time.time() - started_at, 2)
        logging.info(f"===== Startup complete in {elapsed}s =====")

    except Exception as e:
        logging.error(f"Startup failed while loading models: {e}")
        raise MyException(e, sys) from e

    # Application runs while control is paused here
    yield

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    logging.info("===== Application shutdown: releasing resources =====")
    app.state.prediction_pipeline = None
    app.state.embedder = None


app = FastAPI(
    title="RevuAI Inference API",
    description="Scrapes, predicts sentiment, clusters topics, and generates LLM insights for app reviews.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Simple health check to confirm models are loaded and app is ready."""
    models_ready = (
        getattr(app.state, "prediction_pipeline", None) is not None
        and getattr(app.state, "embedder", None) is not None
    )
    return {
        "status": "ok" if models_ready else "loading",
        "prediction_pipeline_loaded": app.state.prediction_pipeline is not None,
        "embedder_loaded": app.state.embedder is not None,
    }

# -------------------------
# Router
# -------------------------
from router.inference_router import router as inference_router
app.include_router(inference_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)