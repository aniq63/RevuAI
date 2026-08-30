"""
Scrape app reviews from the Google Play Store.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import pandas as pd
import random

from google_play_scraper import Sort, reviews

from utils.logger import logging
from utils.exception import MyException


class Reviews_Scrapping_Data:
    """
    Scrape Google Play Store reviews using different
    review-sorting strategies.
    """

    # List of popular app IDs to scrape reviews from
    app_ids = [
        "com.instagram.android",
        "com.facebook.katana",
        "com.whatsapp",
        "com.twitter.android",
        "com.snapchat.android",
    ]

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

    @staticmethod
    def main():
        """
        Pick 2 random app_ids and, for each, randomly choose one of the
        three fetching strategies (all / newest / relevant). Combine both
        resulting DataFrames into a single DataFrame and return it.
        """
        try:
            # Validate that we have at least 2 app_ids
            if not Reviews_Scrapping_Data.app_ids or len(Reviews_Scrapping_Data.app_ids) < 2:
                logging.error("app_ids list is empty or has fewer than 2 items")
                raise ValueError("At least 2 app_ids are required to run main()")

            # Pick 2 distinct app ids at random
            selected_app_ids = random.sample(
                Reviews_Scrapping_Data.app_ids, 2
            )

            logging.info(f"Selected app ids: {selected_app_ids}")

            combined_df = pd.DataFrame()

            for app_id in selected_app_ids:
                scraper = Reviews_Scrapping_Data(app_id)

                # Randomly pick one of the 3 fetching strategies
                method_name = random.choice(
                    ["fetch_all_reviews", "fetch_newst_reviews", "fetch_relevant_reviews"]
                )

                logging.info(
                    f"Using method '{method_name}' for app_id: {app_id}"
                )

                fetch_method = getattr(scraper, method_name)
                df = fetch_method()

                # tag rows with source app_id (optional but useful)
                df["app_id"] = app_id

                combined_df = pd.concat([combined_df, df], ignore_index=True)

            logging.info(
                f"Combined dataframe shape: {combined_df.shape}"
            )

            # Return only the required columns
            return combined_df[
                ["content", "score", "thumbsUpCount"]
            ]

        except Exception as e:
            logging.error(f"Error in main(): {str(e)}")
            raise MyException(e, sys) from e


if __name__ == "__main__":
    try:
        df_result = Reviews_Scrapping_Data.main()
        print(df_result.head())
        print(df_result.shape)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")



