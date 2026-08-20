"""
Inference pipeline that runs when a user makes a request
from the extension (called via the FastAPI router).

Pipeline
Data Scraping -> Model Prediction -> Sentiment Summary -> Topic Clustering -> LLM Insight

The public entry point is ``start_inference_pipeline`` which returns a fully
JSON-serializable dict, ready to be returned by a FastAPI endpoint. The
``run_inference_async`` wrapper offloads the blocking (CPU/IO heavy) work to a
worker thread so the FastAPI event loop is never blocked.

Pre-loaded objects (model/vectorizer via ``PredictionPipeline`` and the
sentence encoder via ``SentenceTransformer``) can be injected at construction
time so expensive model loading happens once at app startup / lifespan instead
of on every request.
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from utils.exception import MyException
from utils.logger import logging

from src.inference.data_scrapping import Reviews_Scrapping
from src.inference.prediction import PredictionPipeline
from src.inference.analyze_sentiment import get_sentiment_percentages, get_top_thumbs_up
from src.inference.topic_clustering import SentimentTopicClusterer
from src.inference.llm_insight import ReviewInsightGenerator


class InferencePipeline:
    def __init__(
        self,
        app_id: str,
        scraping_method: Literal[
            "all_reviews",
            "new_reviews",
            "relevant_reviews"
        ],
        batch_size: int = 500,
        prediction_pipeline: Optional[PredictionPipeline] = None,
        embedder: Optional["SentenceTransformer"] = None,
    ):
        self.app_id = app_id
        self.scraping_method = scraping_method
        self.batch_size = batch_size
        self.prediction_pipeline = prediction_pipeline
        self.embedder = embedder

    # ====================================
    # Data Scraping
    # =====================================
    def scrape_data(self) -> pd.DataFrame:
        """
        Scrape reviews based on the selected scraping method.

        Returns
        -------
        pd.DataFrame
            Scraped reviews containing:
            content, score, thumbsUpCount
        """

        try:
            logging.info(
                f"Starting inference data scraping for app_id: "
                f"{self.app_id}"
            )

            logging.info(
                f"Selected scraping method: "
                f"{self.scraping_method}"
            )

            # Initialize scraper
            scraper = Reviews_Scrapping(
                app_id=self.app_id
            )

            # ==========================================
            # All Reviews
            # ==========================================
            if self.scraping_method == "all_reviews":

                logging.info(
                    "Fetching all reviews "
                    "(newest + relevant)."
                )

                df = scraper.fetch_all_reviews()

            # ==========================================
            # New Reviews
            # ==========================================
            elif self.scraping_method == "new_reviews":

                logging.info(
                    "Fetching newest reviews."
                )

                df = scraper.fetch_newst_reviews()

            # ==========================================
            # Relevant Reviews
            # ==========================================
            elif self.scraping_method == "relevant_reviews":

                logging.info(
                    "Fetching most relevant reviews."
                )

                df = scraper.fetch_relevant_reviews()

            # ==========================================
            # Invalid Method
            # ==========================================
            else:
                raise ValueError(
                    f"Invalid scraping method: "
                    f"{self.scraping_method}. "
                    "Choose from: all_reviews, "
                    "new_reviews, relevant_reviews."
                )

            logging.info(
                f"Successfully scraped {len(df)} reviews "
                f"using '{self.scraping_method}'."
            )

            return df

        except Exception as e:

            logging.error(
                f"Error during inference data scraping "
                f"for app_id {self.app_id}: {str(e)}"
            )

            raise MyException(e, sys) from e

    # =======================================
    # ML Reviews Prediction
    # =======================================
    def reviews_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the Sentiment on app reviews

        Returns
        --------
        pandas Dataframe
            having column ['content','label','thumbsUpCount','sentiment','score']
        """

        try:
            logging.info(
                f"Running sentiment prediction on "
                f"{len(df)} reviews..."
            )

            pipeline = (
                self.prediction_pipeline
                or PredictionPipeline()
            )
            result = pipeline.predict(
                df,
                decode_labels=True,
                batch_size=self.batch_size,
            )

            return result

        except Exception as e:
            logging.error(f"Error during the prediction pipeline: {e}")
            raise MyException(e, sys) from e

    # ========================================
    # Analyze Sentiment Prediction
    # ========================================
    def get_sentiment_summary(
        self,
        df: pd.DataFrame,
    ) -> tuple[dict, dict]:
        """
        Compute the sentiment distribution and the top
        thumbs-up reviews per sentiment class.

        Returns
        -------
        tuple[dict, dict]
            top_reviews : dict[str, pd.DataFrame]
                Top reviews per sentiment class.
            sentiment_percentages : dict
                {sentiment: percentage} rounded to 2 decimals.
        """

        try:

            # ==========================================
            # Calculate Sentiment Percentages
            # ==========================================
            sentiment_percentages = get_sentiment_percentages(df)

            # ==========================================
            # Get Top Reviews Per Sentiment
            # ==========================================
            top_reviews = get_top_thumbs_up(df)

            logging.info(
                "Successfully generated complete sentiment summary."
            )

            return top_reviews, sentiment_percentages

        except Exception as e:

            logging.error(
                f"Failed to generate sentiment summary: {e}"
            )
            raise MyException(e, sys) from e

    # ========================================
    # Topic Clustering from Predicted Reviews
    # ========================================
    def clustering(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Cluster predicted reviews by sentiment and extract
        the main topics for each sentiment class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least:
            - content
            - sentiment

        Returns
        -------
        tuple[pd.DataFrame, dict]
            clustered_df : pd.DataFrame
                Original review data with cluster and topic columns.

            topic_summary : dict
                User-facing topic summary for positive, neutral,
                and negative sentiments.
        """

        try:
            logging.info(
                "Starting sentiment-based topic clustering."
            )

            # ========================================
            # Initialize Topic Clusterer
            # ========================================
            clusterer = SentimentTopicClusterer(
                embedding_model_name="all-MiniLM-L6-v2",
                top_n_keywords=3,
                min_reviews_to_cluster=30,
                top_topics_to_show=3,
                enable_embedding_cache=True,
                embedder=self.embedder,
            )

            # ========================================
            # Run Clustering Pipeline
            # ========================================
            clusterer.fit(df)

            # ========================================
            # Get Results
            # ========================================
            clustered_df = clusterer.get_final_dataframe()

            topic_summary = (
                clusterer.get_user_facing_summary()
            )

            logging.info(
                "Topic clustering completed successfully."
            )

            return clustered_df, topic_summary

        except Exception as e:
            logging.error(
                f"Error during topic clustering: {str(e)}"
            )

            raise MyException(e, sys) from e

    # ========================================
    # LLM Insights
    # ========================================
    def llm_insights(self, top_reviews: dict, topic_summary: dict) -> str:
        try:
            generator = ReviewInsightGenerator()
            analysis = generator.generate(top_reviews, topic_summary)

            return analysis

        except Exception as e:
            logging.error(f"LLM insights pipeline failed: {e}")
            raise MyException(e, sys) from e

    # ========================================
    # Response Helpers
    # ========================================
    @staticmethod
    def _serialize_top_reviews(top_reviews: dict) -> dict:
        """
        Convert the {sentiment: DataFrame} top-reviews structure
        into JSON-serializable records for the FastAPI response.
        """
        return {
            sentiment: frame.to_dict(orient="records")
            for sentiment, frame in top_reviews.items()
        }

    # ==================================
    # Inference Main Pipeline
    # ===================================
    def start_inference_pipeline(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Run the full inference pipeline end-to-end.

        Parameters
        ----------
        df : pd.DataFrame, optional
            If provided (e.g. the router received review data in the
            request body), scraping is skipped and this DataFrame is
            used as the input. Otherwise reviews are scraped from the
            Play Store using ``self.scraping_method``.

        Returns
        -------
        dict
            JSON-serializable result ready for a FastAPI response:
            app_id, scraping_method, review_count, sentiment_percentages,
            top_reviews, topic_summary, llm_insight, timestamps and
            execution_time_seconds.
        """

        started_at = time.time()
        logging.info("===== Inference pipeline started =====")

        # ==========================================
        # 1. Data Scraping (or injected data)
        # ==========================================
        scrape_df = (
            df
            if (df is not None and not df.empty)
            else self.scrape_data()
        )

        review_count = int(len(scrape_df))
        if review_count == 0:
            logging.warning(
                "No reviews available to analyze for "
                f"app_id: {self.app_id}"
            )
            return self._build_result(
                started_at,
                review_count=0,
                sentiment_percentages={},
                top_reviews={},
                topic_summary={},
                llm_insight="",
            )

        # ==========================================
        # 2. Model Prediction
        # ==========================================
        pred_df = self.reviews_prediction(df=scrape_df)

        # ==========================================
        # 3. Sentiment Summary
        # ==========================================
        top_reviews, sentiment_percentages = (
            self.get_sentiment_summary(df=pred_df)
        )

        # ==========================================
        # 4. Topic Clustering
        # ==========================================
        topic_summary = {}
        if not pred_df.empty:
            _, topic_summary = self.clustering(df=pred_df)

        # ==========================================
        # 5. LLM Insight
        # ==========================================
        llm_insight = self.llm_insights(
            top_reviews=top_reviews,
            topic_summary=topic_summary,
        )

        elapsed = round(time.time() - started_at, 2)
        logging.info(
            f"===== Inference pipeline finished in {elapsed}s "
            f"for app_id: {self.app_id} ====="
        )

        return self._build_result(
            started_at,
            review_count=review_count,
            sentiment_percentages=sentiment_percentages,
            top_reviews=top_reviews,
            topic_summary=topic_summary,
            llm_insight=llm_insight,
        )

    # ========================================
    # Async Facade for FastAPI
    # ========================================
    async def run_inference_async(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Async entry point for FastAPI endpoints.

        The inference work (scraping, model scoring, clustering, LLM) is
        blocking, so it is pushed into a worker thread to keep the event
        loop responsive.
        """
        return await asyncio.to_thread(self.start_inference_pipeline, df)

    # ========================================
    # Result Assembly
    # ========================================
    def _build_result(
        self,
        started_at: float,
        review_count: int,
        sentiment_percentages: dict,
        top_reviews: dict,
        topic_summary: dict,
        llm_insight: str,
    ) -> dict:
        """Assemble a JSON-serializable result dict for the router."""
        return {
            "app_id": self.app_id,
            "scraping_method": self.scraping_method,
            "review_count": review_count,
            "sentiment_percentages": sentiment_percentages,
            "top_reviews": self._serialize_top_reviews(top_reviews),
            "topic_summary": topic_summary,
            "llm_insight": llm_insight,
            "insight_generated_at": datetime.utcnow().isoformat() + "Z",
            "execution_time_seconds": round(time.time() - started_at, 2),
        }


# =======================================
# CLI Testing
# =======================================
if __name__ == "__main__":
    try:
        import asyncio

        result = asyncio.run(
            InferencePipeline(
                app_id="com.supercell.clashofclans",
                scraping_method="new_reviews",
            ).run_inference_async()
        )

        print("\n=== Inference Pipeline Result ===")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except MyException as e:
        print(f"Inference pipeline failed: {e}")