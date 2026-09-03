"""
Main FastAPI application entry point for RevuAI.

Uses a `lifespan` context manager to preload heavy, stateless resources
ONCE at application startup:
    - Champion ML model + fitted TF-IDF vectorizer (via PredictionPipeline)
    - Sentence embedding model (SentenceTransformer, used by topic clustering)

These are stored on `app.state` so every request reuses the same in-memory
objects instead of re-loading them from MLflow / disk on every call.

CPU-specific notes (this app is deployed on CPU-only machines):
    - torch / OpenMP thread counts are pinned explicitly. By default,
      torch and OpenBLAS/MKL each try to use ALL available cores, and on a
      multi-worker or multi-request server that causes thread contention
      that actually slows things down. Pinning to a sane number keeps
      per-request latency predictable.
    - A simple in-memory, TTL-based result cache is attached to app.state so
      repeated /analyze calls for the same app_id within the TTL window
      return instantly instead of re-running the whole pipeline.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- Added for CORS

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging
from utils.exception import MyException

from src.inference.prediction import PredictionPipeline
from sentence_transformers import SentenceTransformer
import torch


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))

RESULT_CACHE_TTL_SECONDS = int(os.environ.get("RESULT_CACHE_TTL_SECONDS", str(30 * 60)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pin CPU thread counts, load the Champion model, TF-IDF
    vectorizer, and the sentence embedding model once, and attach them
    (plus an empty result cache) to app.state.

    Shutdown: release references (garbage collected automatically).
    """
    started_at = time.time()
    logging.info("===== Application startup: loading models =====")

    try:
        # --------------------------------------------------
        # 0. Pin CPU thread counts (see module docstring)
        # --------------------------------------------------
        torch.set_num_threads(TORCH_NUM_THREADS)
        logging.info(f"torch.set_num_threads({TORCH_NUM_THREADS})")

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
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        app.state.embedder = embedder
        logging.info("Embedding model loaded successfully.")

        # --------------------------------------------------
        # 3. Result cache: {cache_key: (result_dict, expires_at_epoch)}
        # --------------------------------------------------
        app.state.result_cache = {}
        app.state.result_cache_ttl = RESULT_CACHE_TTL_SECONDS
        logging.info(f"Result cache TTL set to {RESULT_CACHE_TTL_SECONDS}s")

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
    app.state.result_cache = {}


app = FastAPI(
    title="RevuAI Inference API",
    description="Scrapes, predicts sentiment, clusters topics, and generates LLM insights for app reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

# -------------------------
# CORS Middleware Configuration
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "RevuAI Inference API that Scrapes, predicts sentiment, clusters topics, and generates LLM insights for app reviews"
    }


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
        "cached_results": len(getattr(app.state, "result_cache", {})),
    }


# -------------------------
# Router
# -------------------------
from router.inference_router import router as inference_router
app.include_router(inference_router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)