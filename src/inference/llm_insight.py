"""
LLM-based review insight generator.

Takes the outputs of the sentiment/topic pipeline (top thumbs-up reviews
per sentiment + clustered topic summary) and asks an LLM to produce a
plain, human-readable analysis: overall summary, what users like, what
users complain about, patterns in the reviews, and recommendations.

Expects the following inputs (already computed elsewhere in the pipeline):
    - top_reviews   : dict[str, pd.DataFrame]  -> output of get_top_thumbs_up()
    - topic_summary : dict[str, list[dict]]     -> output of
                       SentimentTopicClusterer.get_user_facing_summary()
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.exception import MyException
from utils.logger import logging

from config.llm_constants import SENTIMENT_LABELS , SYSTEM_PROMPT , USER_PROMPT

load_dotenv()

class ReviewInsightGenerator:
    """
    Generates a plain-text, human-readable analysis of app reviews using
    an LLM, based on top thumbs-up reviews and clustered topic summaries.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        request_timeout: float = 60.0,
        max_retries: int = 3,
        top_n_reviews: int = 5,
    ):
        try:
            self.model_name = model_name
            self.top_n_reviews = top_n_reviews

            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. Make sure it is set in your .env file."
                )

            self.llm = ChatGroq(
                model_name=self.model_name,
                groq_api_key=self.api_key,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )

            self.prompt = PromptTemplate(
                template=SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
                input_variables=[
                    "topic_clusters",
                    "top_positive",
                    "top_neutral",
                    "top_negative",
                ],
            )

            self.chain = self.prompt | self.llm | StrOutputParser()

            self.result_text: Optional[str] = None

            logging.info(f"ReviewInsightGenerator initialized with model={model_name}")

        except Exception as e:
            logging.error(f"Failed to initialize ReviewInsightGenerator: {e}")
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Helper: convert a reviews DataFrame into compact text for the prompt
    # ------------------------------------------------------------------
    def _df_to_review_text(self, df: pd.DataFrame) -> str:
        try:
            if df is None or df.empty:
                return "No reviews available."

            lines = []
            for _, row in df.head(self.top_n_reviews).iterrows():
                content = str(row.get("content", "")).strip().replace("\n", " ")
                thumbs = row.get("thumbsUpCount", 0)
                if content:
                    lines.append(f"- ({thumbs} upvotes) {content}")

            return "\n".join(lines) if lines else "No reviews available."

        except Exception as e:
            logging.error(f"Failed to format review DataFrame: {e}")
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Helper: validate inputs coming from the pipeline
    # ------------------------------------------------------------------
    def _validate_inputs(
        self, top_reviews: Dict[str, pd.DataFrame], topic_summary: dict
    ) -> None:
        if not isinstance(top_reviews, dict):
            raise TypeError("top_reviews must be a dict of DataFrames.")
        if not isinstance(topic_summary, dict):
            raise TypeError("topic_summary must be a dict.")

        missing = [cls for cls in SENTIMENT_LABELS if cls not in top_reviews]
        if missing:
            raise KeyError(
                f"top_reviews is missing sentiment class(es): {', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate(
        self,
        top_reviews: Dict[str, pd.DataFrame],
        topic_summary: dict,
    ) -> str:
        """
        Generate the plain-text review analysis.

        Parameters
        ----------
        top_reviews : dict[str, pd.DataFrame]
            Output of analyze_sentiment.get_top_thumbs_up(df).
        topic_summary : dict
            Output of SentimentTopicClusterer.get_user_facing_summary().

        Returns
        -------
        str
            Plain-text, markdown-formatted analysis.
        """
        try:
            self._validate_inputs(top_reviews, topic_summary)

            logging.info("Formatting inputs for LLM prompt...")
            top_positive_text = self._df_to_review_text(top_reviews.get("positive"))
            top_neutral_text = self._df_to_review_text(top_reviews.get("neutral"))
            top_negative_text = self._df_to_review_text(top_reviews.get("negative"))
            topic_clusters_text = json.dumps(
                topic_summary, indent=2, ensure_ascii=False
            )

            logging.info(f"Calling LLM ({self.model_name}) for review insights...")
            self.result_text = self.chain.invoke(
                {
                    "topic_clusters": topic_clusters_text,
                    "top_positive": top_positive_text,
                    "top_neutral": top_neutral_text,
                    "top_negative": top_negative_text,
                }
            )

            logging.info("LLM insight generation completed successfully.")
            return self.result_text

        except Exception as e:
            logging.error(f"Failed to generate review insights: {e}")
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------
    def get_result(self) -> Optional[str]:
        """Return the last generated analysis text (or None if not generated yet)."""
        return self.result_text


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    from analyze_sentiment import get_top_thumbs_up

    # ---- Dummy data for a quick end-to-end test ----
    data = {
        "sentiment": [
            "positive", "negative", "positive", "neutral",
            "negative", "positive", "neutral", "negative",
        ],
        "content": [
            "Love this app, works great!",
            "Contactless payment keeps failing, very frustrating.",
            "Best app I've used, super smooth.",
            "It's okay, could use more customization for the widget.",
            "Chrome integration is unusable, tabs keep crashing.",
            "Battery friendly and fast, great job devs.",
            "Calendar sync is fine but nothing special.",
            "Alarms don't go off half the time, unreliable.",
        ],
        "thumbsUpCount": [50, 40, 35, 20, 45, 30, 15, 38],
    }
    df_test = pd.DataFrame(data)

    top_reviews_test = get_top_thumbs_up(df_test)

    topic_summary_test = {
        "negative": [
            {"topic": "contactless payment | frustrating", "count": 143, "percentage": 12.0},
            {"topic": "chrome unusable | tabs crashing", "count": 121, "percentage": 10.2},
            {"topic": "alarms unreliable | settings", "count": 119, "percentage": 10.0},
        ],
        "neutral": [
            {"topic": "calendar widget | customization", "count": 65, "percentage": 12.7},
        ],
        "positive": [
            {"topic": "smooth performance | battery friendly", "count": 107, "percentage": 15.1},
        ],
    }

    try:
        generator = ReviewInsightGenerator()
        analysis = generator.generate(top_reviews_test, topic_summary_test)

        print("\n=== LLM Review Analysis ===\n")
        print(analysis)

    except MyException as e:
        print(f"Pipeline failed: {e}")