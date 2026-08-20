"""
Topic clustering for reviews.

Clusters reviews per sentiment label and extracts representative keywords
(topics) for each cluster using sentence embeddings, UMAP, HDBSCAN, and
KeyBERT. Also builds a clean, user-facing summary for the browser extension.

Expected DataFrame columns:
    - content  : review text
    - sentiment: predicted sentiment label (e.g. "negative", "neutral", "positive")
"""

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import hdbscan
import numpy as np
import pandas as pd
import umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.exception import MyException
from utils.logger import logging

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
REQUIRED_COLUMNS = {"content", "sentiment"}


class SentimentTopicClusterer:
    """
    Cluster reviews by sentiment and extract keywords/topics per cluster.

    Expected DataFrame columns:
        - content  : review text
        - sentiment: predicted sentiment label (e.g. "negative", "neutral", "positive")
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_n_keywords: int = 3,
        min_reviews_to_cluster: int = 30,
        top_topics_to_show: int = 3,      # for user-facing summary
        max_clusters: int = 3,            # cap the number of clusters kept per sentiment
        enable_embedding_cache: bool = True,  # in-memory cache
        embedder: Optional[SentenceTransformer] = None,
    ):
        self.top_n_keywords = top_n_keywords
        self.min_reviews_to_cluster = min_reviews_to_cluster
        self.top_topics_to_show = top_topics_to_show
        self.max_clusters = max_clusters
        self.enable_embedding_cache = enable_embedding_cache

        logging.info("Loading embedding + keyword models...")
        with warnings.catch_warnings():
            # SentenceTransformer/UMAP emit deprecation warnings at load time.
            warnings.simplefilter("ignore")
            self.embedder = (
                embedder if embedder is not None
                else SentenceTransformer(embedding_model_name)
            )
        self.kw_model = KeyBERT(self.embedder)

        # In-memory embedding cache: key = hash of texts -> embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

        self.df = None
        self.embeddings = None
        self.results_by_sentiment = {}
        self.topics_by_sentiment = {}
        self.df_final = None
        self.app_summary = {}
        self.user_facing_summary = {}   # clean summary for the extension

    # ------------------------------------------------------------------
    # Helper: create a stable cache key from the list of texts
    # ------------------------------------------------------------------
    @staticmethod
    def _make_cache_key(texts: List[str]) -> str:
        joined = "||".join(texts)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """
        Run the full pipeline on the given DataFrame.

        Returns
        -------
        SentimentTopicClusterer
            Self, populated with clustering results.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            if not REQUIRED_COLUMNS.issubset(df.columns):
                raise ValueError(
                    "DataFrame must contain 'content' and 'sentiment' columns."
                )

            self.df = df.copy()
            self.df["content"] = self.df["content"].astype(str)
            texts = self.df["content"].tolist()

            # ---- Embeddings with optional caching ----
            cache_key = (
                self._make_cache_key(texts) if self.enable_embedding_cache else None
            )

            if self.enable_embedding_cache and cache_key in self._embedding_cache:
                logging.info("Using cached embeddings...")
                self.embeddings = self._embedding_cache[cache_key]
            else:
                logging.info("Embedding all reviews...")
                self.embeddings = self.embedder.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=True,
                )
                if self.enable_embedding_cache and cache_key is not None:
                    self._embedding_cache[cache_key] = self.embeddings

            self.df["_embedding_idx"] = range(len(self.df))

            # ---- Cluster per sentiment ----
            self.results_by_sentiment = {}
            self.topics_by_sentiment = {}

            for sentiment in SENTIMENT_LABELS:
                clustered_df, topics = self._cluster_sentiment_group(sentiment)
                self.results_by_sentiment[sentiment] = clustered_df
                self.topics_by_sentiment[sentiment] = topics

            # ---- Combine results ----
            valid = [v for v in self.results_by_sentiment.values() if v is not None]
            self.df_final = (
                pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
            )

            # ---- Build keyword summary (raw) ----
            self.app_summary = {}
            for sentiment, topics in self.topics_by_sentiment.items():
                keyword_list = []
                for cluster_id, name in topics.items():
                    if cluster_id == -1:
                        continue
                    keyword_list.extend(name.split(" | "))
                self.app_summary[sentiment] = keyword_list

            # ---- Build clean user-facing summary (top topics + counts) ----
            self._build_user_facing_summary()

            return self

        except Exception as e:
            logging.error(f"Topic clustering failed: {e}")
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Clustering for one sentiment
    # ------------------------------------------------------------------
    def _cluster_sentiment_group(self, sentiment_label: str):
        subset = self.df[self.df["sentiment"] == sentiment_label]
        n = len(subset)

        if n < self.min_reviews_to_cluster:
            logging.info(
                f"[{sentiment_label}] Too few reviews ({n}) to cluster "
                "meaningfully. Skipping."
            )
            return None, {}

        sub_embeddings = self.embeddings[subset["_embedding_idx"].values]

        # ---- UMAP ----
        reducer = umap.UMAP(
            n_components=min(10, n - 2),
            n_neighbors=min(15, n - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(sub_embeddings)

        # ---- HDBSCAN ----
        min_cluster_size = int(np.clip(n * 0.02, 15, 80))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(5, min_cluster_size // 5),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        cluster_labels = clusterer.fit_predict(reduced)

        # Keep only the largest clusters (by review count) to bound the
        # number of KeyBERT extractions and overall runtime. The removed
        # clusters are treated as noise (-1).
        if self.max_clusters is not None and self.max_clusters > 0:
            cluster_series = pd.Series(cluster_labels)
            cluster_counts = cluster_series.value_counts()
            cluster_counts = cluster_counts[cluster_counts.index != -1]
            biggest_clusters = set(
                cluster_counts.index[: self.max_clusters]
            )
            cluster_labels = np.array(
                [
                    lab if lab in biggest_clusters else -1
                    for lab in cluster_labels
                ]
            )

        subset = subset.copy()
        subset["cluster"] = cluster_labels

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()
        logging.info(
            f"[{sentiment_label.upper()}] {n} reviews -> {n_clusters} clusters, "
            f"{n_noise} noise ({n_noise / n * 100:.1f}%)"
        )

        # ---- KeyBERT keyword extraction ----
        cluster_names = {}
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                cluster_names[-1] = "Uncategorized"
                continue

            cluster_reviews = (
                subset[subset["cluster"] == cluster_id]["content"].tolist()
            )
            combined_text = " ".join(cluster_reviews[:200])[:20000]

            try:
                keywords = self.kw_model.extract_keywords(
                    combined_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    top_n=self.top_n_keywords,
                    use_mmr=True,
                    diversity=0.5,
                )
                topic_name = (
                    " | ".join([kw[0] for kw in keywords]) if keywords else "N/A"
                )
            except Exception:
                logging.warning(
                    f"Keyword extraction failed for cluster {cluster_id}; "
                    "falling back to 'N/A'.",
                    exc_info=True,
                )
                topic_name = "N/A"

            cluster_names[cluster_id] = topic_name
            logging.info(
                f"  Cluster {cluster_id:2d} ({len(cluster_reviews):4d} reviews): "
                f"{topic_name}"
            )

        subset["topic"] = subset["cluster"].map(cluster_names)
        return subset, cluster_names

    # ------------------------------------------------------------------
    # Build clean summary for the extension UI
    # ------------------------------------------------------------------
    def _build_user_facing_summary(self):
        """
        Creates a clean structure ready for the extension:

        {
          "negative": [
            {"topic": "battery life | charging", "count": 42, "percentage": 18.5},
            ...
          ],
          "neutral": [...],
          "positive": [...]
        }
        """
        self.user_facing_summary = {}

        for sentiment in SENTIMENT_LABELS:
            clustered_df = self.results_by_sentiment.get(sentiment)

            if clustered_df is None or clustered_df.empty:
                self.user_facing_summary[sentiment] = []
                continue

            total = len(clustered_df)

            # Count reviews per topic (ignore Uncategorized / -1)
            topic_counts = (
                clustered_df[clustered_df["cluster"] != -1]
                .groupby("topic")
                .size()
                .reset_index(name="count")
            )

            # Sort by count descending and keep top N
            topic_counts = topic_counts.sort_values("count", ascending=False)
            topic_counts = topic_counts.head(self.top_topics_to_show)

            topics_list = []
            for _, row in topic_counts.iterrows():
                topics_list.append({
                    "topic": row["topic"],
                    "count": int(row["count"]),
                    "percentage": round(row["count"] / total * 100, 1)
                })

            self.user_facing_summary[sentiment] = topics_list

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Raw keyword list per sentiment (legacy)."""
        return self.app_summary

    def get_user_facing_summary(self) -> dict:
        """Clean summary for the extension UI (recommended)."""
        return self.user_facing_summary

    def print_summary(self):
        """Pretty-print the user-facing summary."""
        print(json.dumps(self.user_facing_summary, indent=2))

    def get_final_dataframe(self) -> pd.DataFrame:
        """Return the combined clustered DataFrame."""
        return self.df_final

    def clear_embedding_cache(self):
        """Clear the in-memory embedding cache if needed."""
        self._embedding_cache.clear()
        logging.info("Embedding cache cleared.")


# ------------------------
# How to use the class
# ------------------------

# clusterer = SentimentTopicClusterer(
#     top_n_keywords=3,
#     top_topics_to_show=3,          # show top 3 topics per sentiment
#     max_clusters=3,                # keep only the 3 largest clusters
#     enable_embedding_cache=True    # cache embeddings
# )

# clusterer.fit(df)   # df must have columns: content, sentiment

# # What the extension should use:
# summary = clusterer.get_user_facing_summary()
# print(json.dumps(summary, indent=2))