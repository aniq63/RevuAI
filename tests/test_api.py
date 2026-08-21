"""
API tests for the FastAPI application using the FastAPI TestClient.

The heavy startup resources (Champion model + TF-IDF vectorizer via
MLflow, and the SentenceTransformer embedder) are replaced with lightweight
fakes BEFORE the TestClient enters its lifespan context, and the
/inference/analyze endpoint's InferencePipeline is faked as well so no
scraping / LLM calls happen.

Covered behaviour:
    - GET  /            -> welcome payload
    - GET  /health      -> reports models ready
    - POST /inference/analyze
        * happy path (cache_hit=False)
        * TTL cache hit on the second identical call
        * force_refresh bypasses the cache
        * 422 on an invalid scraping_method
        * 503 when models are not loaded yet
        * 500 when the inference pipeline fails
    - DELETE /inference/cache/{app_id} -> manual cache invalidation
"""

import sys

import pytest
from fastapi.testclient import TestClient

from utils.exception import MyException


# ==============================================
# Fakes injected into the app lifespan + router
# ==============================================
class FakePredictionPipeline:
    """Stand-in so lifespan skips the MLflow model loading."""

    def __init__(self, *args, **kwargs):
        pass


class FakeEmbedder:
    """Stand-in for SentenceTransformer."""

    def encode(self, texts, *args, **kwargs):
        import numpy as np

        return np.zeros((len(texts), 4))


CANNED_RESULT = {
    "app_id": "com.test.app",
    "scraping_method": "new_reviews",
    "review_count": 3,
    "sentiment_percentages": {"positive": 66.67, "negative": 33.33},
    "top_reviews": {
        "positive": [{"content": "great", "thumbsUpCount": 10}],
        "neutral": [],
        "negative": [{"content": "bad", "thumbsUpCount": 1}],
    },
    "topic_summary": {"positive": ["performance", "ui"]},
    "llm_insight": "Users mostly enjoy the app.",
    "insight_generated_at": "2026-01-01T00:00:00Z",
    "execution_time_seconds": 0.01,
}


class FakeInferencePipeline:
    """Stand-in for the full scrape -> predict -> cluster -> LLM pipeline."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def run_inference_async(self, df=None):
        return dict(CANNED_RESULT)


class FailingInferencePipeline(FakeInferencePipeline):
    async def run_inference_async(self, df=None):
        try:
            raise RuntimeError("pipeline exploded")
        except RuntimeError as e:
            # MyException reads sys.exc_info(), so it must be built
            # while the original error is being handled.
            raise MyException(e, sys)


# ==============================================
# Client fixtures
# ==============================================
def _build_client(monkeypatch, pipeline_cls=FakeInferencePipeline):
    import main
    import router.inference_router as inference_router

    # Patch BEFORE entering the TestClient context: main.lifespan resolves
    # these names from the module namespace at startup time.
    monkeypatch.setattr(main, "PredictionPipeline", FakePredictionPipeline)
    monkeypatch.setattr(main, "SentenceTransformer", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(inference_router, "InferencePipeline", pipeline_cls)

    return TestClient(main.app)


@pytest.fixture
def client(monkeypatch):
    with _build_client(monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def failing_client(monkeypatch):
    with _build_client(monkeypatch, pipeline_cls=FailingInferencePipeline) as test_client:
        yield test_client


ANALYZE_BODY = {"app_id": "com.test.app", "scraping_method": "new_reviews"}


# ==============================================
# BASIC ENDPOINTS
# ==============================================
class TestBasicEndpoints:
    def test_root_returns_welcome_message(self, client):
        response = client.get("/")

        assert response.status_code == 200
        body = response.json()
        assert "message" in body
        assert "RevuAI" in body["message"]

    def test_health_reports_models_ready_after_startup(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["prediction_pipeline_loaded"] is True
        assert body["embedder_loaded"] is True


# ==============================================
# POST /inference/analyze
# ==============================================
class TestAnalyzeEndpoint:
    def test_analyze_success_returns_full_result(self, client):
        response = client.post("/inference/analyze", json=ANALYZE_BODY)

        assert response.status_code == 200
        body = response.json()
        assert body["cache_hit"] is False
        assert body["app_id"] == "com.test.app"
        assert body["review_count"] == 3
        assert "sentiment_percentages" in body
        assert "topic_summary" in body
        assert "llm_insight" in body
        assert "insight_generated_at" in body
        assert "execution_time_seconds" in body

    def test_second_identical_call_hits_cache(self, client):
        first = client.post("/inference/analyze", json=ANALYZE_BODY)
        second = client.post("/inference/analyze", json=ANALYZE_BODY)

        assert first.json()["cache_hit"] is False
        assert second.json()["cache_hit"] is True
        assert second.json()["review_count"] == first.json()["review_count"]

    def test_force_refresh_bypasses_cache(self, client):
        client.post("/inference/analyze", json=ANALYZE_BODY)

        refreshed = client.post(
            "/inference/analyze",
            json={**ANALYZE_BODY, "force_refresh": True},
        )

        assert refreshed.json()["cache_hit"] is False

    def test_different_scraping_method_is_a_separate_cache_entry(self, client):
        client.post("/inference/analyze", json=ANALYZE_BODY)

        other = client.post(
            "/inference/analyze",
            json={**ANALYZE_BODY, "scraping_method": "all_reviews"},
        )

        assert other.json()["cache_hit"] is False

    def test_invalid_scraping_method_returns_422(self, client):
        response = client.post(
            "/inference/analyze",
            json={**ANALYZE_BODY, "scraping_method": "bogus_method"},
        )

        assert response.status_code == 422

    def test_missing_app_id_returns_422(self, client):
        response = client.post("/inference/analyze", json={})

        assert response.status_code == 422

    def test_models_not_loaded_returns_503(self, client):
        # Simulate a request arriving before startup finished.
        client.app.state.prediction_pipeline = None

        response = client.post("/inference/analyze", json=ANALYZE_BODY)

        assert response.status_code == 503
        assert "loading" in response.json()["detail"].lower()

    def test_pipeline_failure_returns_500(self, failing_client):
        response = failing_client.post("/inference/analyze", json=ANALYZE_BODY)

        assert response.status_code == 500
        assert "Inference failed" in response.json()["detail"]


# ==============================================
# DELETE /inference/cache/{app_id}
# ==============================================
class TestCacheInvalidationEndpoint:
    def test_clear_existing_cache_entry(self, client):
        client.post("/inference/analyze", json=ANALYZE_BODY)

        response = client.delete("/inference/cache/com.test.app")

        assert response.status_code == 200
        body = response.json()
        assert body["cleared"] is True
        assert body["app_id"] == "com.test.app"

        # Cache is really gone -> next call recomputes.
        follow_up = client.post("/inference/analyze", json=ANALYZE_BODY)
        assert follow_up.json()["cache_hit"] is False

    def test_clear_missing_entry_reports_cleared_false(self, client):
        response = client.delete("/inference/cache/com.never.analyzed")

        assert response.status_code == 200
        assert response.json()["cleared"] is False
