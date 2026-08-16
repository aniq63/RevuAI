"""
Scrape app reviews from the Google Play Store.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import pandas as pd
from google_play_scraper import Sort, reviews

from utils.logger import logging
from utils.exception import MyException


class Reviews_Scrapping:
    """
    Scrape Google Play Store reviews using different
    review-sorting strategies.
    """

    def __init__(self, app_id: str):
        self.app_id = app_id
        self.final_sample_size = 4000
        self.lan = "en"
        self.country = "us"
        self.random_state = 42

    def fetch_all_reviews(
        self,
        newest_count: int = 3000,
        relevant_count: int = 3000,
    ) -> pd.DataFrame:
        """
        Collect reviews using two complementary strategies:

        1. NEWEST -> captures current/recent sentiment.
        2. MOST_RELEVANT -> captures important/community-relevant reviews.

        The two datasets are combined, duplicates are removed,
        randomly shuffled, and the final sample is selected.

        Parameters
        ----------
        newest_count : int
            Number of newest reviews to fetch.

        relevant_count : int
            Number of most relevant reviews to fetch.

        Returns
        -------
        pd.DataFrame
            Final review dataset containing:
            content, score, and thumbsUpCount.
        """

        try:
            logging.info(
                f"Starting review scraping for app: {self.app_id}"
            )

            # ===============================
            # Fetch Newest Reviews
            # ===============================

            logging.info(
                f"Fetching {newest_count} newest reviews."
            )

            newest_reviews, _ = reviews(
                self.app_id,
                lang=self.lan,
                country=self.country,
                sort=Sort.NEWEST,
                count=newest_count,
            )

            logging.info(
                f"Successfully fetched {len(newest_reviews)} "
                "newest reviews."
            )

            # ===============================
            # Fetch Most Relevant Reviews
            # ===============================

            logging.info(
                f"Fetching {relevant_count} most relevant reviews."
            )

            relevant_reviews, _ = reviews(
                self.app_id,
                lang=self.lan,
                country=self.country,
                sort=Sort.MOST_RELEVANT,
                count=relevant_count,
            )

            logging.info(
                f"Successfully fetched {len(relevant_reviews)} "
                "most relevant reviews."
            )

            # ===============================
            # Combine Reviews
            # ===============================

            combined_reviews = newest_reviews + relevant_reviews

            logging.info(
                f"Total reviews before deduplication: "
                f"{len(combined_reviews)}"
            )

            # ===============================
            # Convert to DataFrame
            # ===============================

            df = pd.DataFrame(combined_reviews)

            # ===============================
            # Remove Duplicate Reviews
            # ===============================

            df.drop_duplicates(
                subset=["reviewId"],
                inplace=True,
            )

            logging.info(
                f"Total reviews after deduplication: {len(df)}"
            )

            # ===============================
            # Shuffle Reviews
            # ===============================

            df = (
                df.sample(
                    frac=1,
                    random_state=self.random_state,
                )
                .reset_index(drop=True)
            )

            # ===============================
            # Select Final Sample
            # ===============================

            final_sample_size = min(
                self.final_sample_size,
                len(df),
            )

            df_final = (
                df.head(final_sample_size)
                .reset_index(drop=True)
            )

            logging.info(
                f"Final review sample size: {len(df_final)}"
            )

            # ===============================
            # Select Required Columns
            # ===============================

            return df_final[
                ["content", "score", "thumbsUpCount"]
            ]

        except Exception as e:
            logging.error(
                f"Error while scraping reviews for "
                f"{self.app_id}: {str(e)}"
            )

            raise MyException(e, sys) from e

    def fetch_newst_reviews(
        self,
        newst_count: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch the newest reviews from the Google Play Store.

        Reviews are deduplicated, shuffled, and limited
        to the final sample size.

        Parameters
        ----------
        newst_count : int
            Number of newest reviews to fetch.

        Returns
        -------
        pd.DataFrame
            Final shuffled review sample containing:
            content, score, and thumbsUpCount.
        """

        try:
            logging.info(
                f"Fetching {newst_count} newest reviews "
                f"for app: {self.app_id}"
            )

            # ===============================
            # Fetch Newest Reviews
            # ===============================

            newest_reviews, _ = reviews(
                self.app_id,
                lang=self.lan,
                country=self.country,
                sort=Sort.NEWEST,
                count=newst_count,
            )

            logging.info(
                f"Successfully fetched {len(newest_reviews)} "
                "newest reviews."
            )

            # ===============================
            # Convert to DataFrame
            # ===============================

            df = pd.DataFrame(newest_reviews)

            # ===============================
            # Remove Duplicate Reviews
            # ===============================

            df.drop_duplicates(
                subset=["reviewId"],
                inplace=True,
            )

            logging.info(
                f"Reviews after deduplication: {len(df)}"
            )

            # ===============================
            # Shuffle Reviews
            # ===============================

            df = (
                df.sample(
                    frac=1,
                    random_state=self.random_state,
                )
                .reset_index(drop=True)
            )

            # ===============================
            # Select Final Sample
            # ===============================

            final_sample_size = min(
                self.final_sample_size,
                len(df),
            )

            df_final = (
                df.head(final_sample_size)
                .reset_index(drop=True)
            )

            logging.info(
                f"Final newest review sample size: "
                f"{len(df_final)}"
            )

            # ===============================
            # Select Required Columns
            # ===============================

            return df_final[
                ["content", "score", "thumbsUpCount"]
            ]

        except Exception as e:
            logging.error(
                f"Error while fetching newest reviews "
                f"for {self.app_id}: {str(e)}"
            )

            raise MyException(e, sys) from e

    def fetch_relevant_reviews(
        self,
        relevent_count: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch the most relevant reviews from the Google Play Store.

        Reviews are deduplicated, shuffled, and limited
        to the final sample size.

        Parameters
        ----------
        relevent_count : int
            Number of most relevant reviews to fetch.

        Returns
        -------
        pd.DataFrame
            Final shuffled review sample containing:
            content, score, and thumbsUpCount.
        """

        try:
            logging.info(
                f"Fetching {relevent_count} most relevant reviews "
                f"for app: {self.app_id}"
            )

            # ===============================
            # Fetch Most Relevant Reviews
            # ===============================

            relevant_reviews, _ = reviews(
                self.app_id,
                lang=self.lan,
                country=self.country,
                sort=Sort.MOST_RELEVANT,
                count=relevent_count,
            )

            logging.info(
                f"Successfully fetched {len(relevant_reviews)} "
                "most relevant reviews."
            )

            # ===============================
            # Convert to DataFrame
            # ===============================

            df = pd.DataFrame(relevant_reviews)

            # ===============================
            # Remove Duplicate Reviews
            # ===============================

            df.drop_duplicates(
                subset=["reviewId"],
                inplace=True,
            )

            logging.info(
                f"Reviews after deduplication: {len(df)}"
            )

            # ===============================
            # Shuffle Reviews
            # ===============================

            df = (
                df.sample(
                    frac=1,
                    random_state=self.random_state,
                )
                .reset_index(drop=True)
            )

            # ===============================
            # Select Final Sample
            # ===============================

            final_sample_size = min(
                self.final_sample_size,
                len(df),
            )

            df_final = (
                df.head(final_sample_size)
                .reset_index(drop=True)
            )

            logging.info(
                f"Final relevant review sample size: "
                f"{len(df_final)}"
            )

            # ===============================
            # Select Required Columns
            # ===============================

            return df_final[
                ["content", "score", "thumbsUpCount"]
            ]

        except Exception as e:
            logging.error(
                f"Error while fetching relevant reviews "
                f"for {self.app_id}: {str(e)}"
            )
            raise MyException(e, sys) from e



# =======================================
# CLI Testing
# =======================================
if __name__ == "__main__":
    try:
        app_id = "com.supercell.clashofclans"

        scraper = Reviews_Scrapping(app_id)

        # ==========================================
        # Test: Fetch All Reviews
        # ==========================================

        df_all_reviews = scraper.fetch_all_reviews()

        print("\n" + "=" * 60)
        print("ALL REVIEWS")
        print("=" * 60)
        print(f"Total reviews: {len(df_all_reviews)}")
        print(df_all_reviews.head())

        # ==========================================
        # Test: Fetch Newest Reviews
        # ==========================================

        df_newest_reviews = scraper.fetch_newst_reviews()

        print("\n" + "=" * 60)
        print("NEWEST REVIEWS")
        print("=" * 60)
        print(f"Total reviews: {len(df_newest_reviews)}")
        print(df_newest_reviews.head())

        # ==========================================
        # Test: Fetch Most Relevant Reviews
        # ==========================================

        df_relevant_reviews = scraper.fetch_relevant_reviews()

        print("\n" + "=" * 60)
        print("MOST RELEVANT REVIEWS")
        print("=" * 60)
        print(f"Total reviews: {len(df_relevant_reviews)}")
        print(df_relevant_reviews.head())

        print("\nReview scraping completed successfully.")

    except MyException as e:
        print(f"Scrapping failed: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")