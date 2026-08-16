"""
Transform the Extracted Data using Text Preprocessing Steps with Batch Processing
"""

import sys
from pathlib import Path
import re
import pandas as pd

# Ensure system path mapping works if executed directly
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging
from utils.exception import MyException
from utils.config_loader import settings


def clean_text(text):
    """
    Clean Play Store review text.

    Removes:
    - HTML tags
    - URLs
    - Emojis
    - Extra whitespace
    """

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emojis and other Unicode symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub(" ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


class TransformData:
    def __init__(self):
        
        # 1. Fetch configurable values from configurations safely
        try:
            self.target_column = settings.get('data_transformatiom', {}).get('text_column', 'content')
            self.batch_size = int(settings.get('data_transformation', {}).get('batch_size', 500))
        except Exception:
            self.target_column = 'content'
            self.batch_size = 1000  # Default fallback size if key is missing

    def _lower_text(self, text: str) -> str:
        """Converts text strings to standard lowercase form."""
        return text.lower()

    def _apply_stemming(self, text: str) -> str:
        """Splits sentences using NLTK and normalizes words via Porter Stemming."""
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    def _text_preprocessing(self, text: str) -> str:
        """Single pipeline transaction containing sequential string processing logic."""
        text_lower = self._lower_text(text)
        cleaned_content = clean_text(text_lower)
        return cleaned_content

    @staticmethod
    def _sentiment_to_number(sentiment: str) -> int:
        """Convert the sentiment label into a numeric value."""
        if sentiment == "negative":
            return 0
        elif sentiment == "neutral":
            return 1
        elif sentiment == "positive":
            return 2
        return sentiment

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a pandas DataFrame, drops missing target texts, executes 
        text cleansing transformations in customizable batches, and returns the fully processed DataFrame.
        """
        try:
            logging.info("Starting batch data transformation pipeline...")
            
            if df is None or df.empty:
                raise ValueError("The provided DataFrame is empty or None.")

            if self.target_column not in df.columns:
                raise KeyError(f"Target column '{self.target_column}' not found in the input DataFrame.")

            # 2. Record historical row matrix count for metrics tracking
            initial_rows = df.shape[0]

            # 3. Handle data filtering steps cleanly by dropping bad rows up front
            df = df.dropna(subset=[self.target_column])
            df[self.target_column] = df[self.target_column].astype(str).str.strip()
            df = df[df[self.target_column] != ""]
            df = df.reset_index(drop=True)
            
            dropped_rows = initial_rows - df.shape[0]
            if dropped_rows > 0:
                logging.info(f"Dropped {dropped_rows} empty/NaN rows from processing tracks.")

            total_rows = df.shape[0]
            logging.info(f"Processing transformation on remaining {total_rows} rows using a batch size of {self.batch_size}.")

            # 4. Implement Sequential Data Batch Loop Transformations
            transformed_series_list = []
            
            for start_idx in range(0, total_rows, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_rows)
                
                # Slicing the data array matrix per batch cycle safely
                batch_chunk = df[self.target_column].iloc[start_idx:end_idx]
                
                # Perform mapping operations strictly on the extracted batch slice
                processed_batch = batch_chunk.apply(self._text_preprocessing)
                transformed_series_list.append(processed_batch)
                
                logging.info(f"Batch transformation progress milestone: Finished index {start_idx} to {end_idx}.")

            # 5. Concatenate all transformed chunks back into the target column
            if transformed_series_list:
                df[self.target_column] = pd.concat(transformed_series_list, ignore_index=True)

            # 5.5 Replace rows whose text became empty after cleaning (e.g. emoji-only) with NaN, then drop them
            df[self.target_column] = df[self.target_column].replace(r"^\s*$", pd.NA, regex=True)
            dropped_cleaned = df[self.target_column].isna().sum()
            if dropped_cleaned > 0:
                logging.info(f"Dropped {dropped_cleaned} rows that became empty after text cleaning.")
            df = df.dropna(subset=[self.target_column])
            df = df.reset_index(drop=True)

            # 6. Convert the sentiment label into numeric values
            if "label" in df.columns:
                df["label"] = df["label"].apply(self._sentiment_to_number)

            logging.info("Batch data transformation completed successfully.")
            logging.info(f"Final shape configuration of data matrix output: {df.shape}")
            
            return df

        except Exception as e:
            logging.error("An explicit failure occurred inside data transformation procedures.")
            raise MyException(e, sys)




# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        # Create a mock DataFrame for localized testing (Simulate multiple records)
        test_data = pd.DataFrame({
            "content": [
                "Beautiful app! 😍 visit https://google.com <b>Love it!</b>", 
                "😃🎉🚀", 
                "Running and walked swiftly.",
                "",
                "Another evaluation statement text record.",
                "Testing final edge configurations here!"
            ]
        })
        
        # Instantiate object with customized miniature testing attributes directly
        transformer = TransformData()
        transformer.batch_size = 2  # Hardcoded batch sizing threshold override for functional testing
        
        transformed_df = transformer.data_transformation(test_data)
        print("\n=== Transformed Target Output Sample ===")
        print(transformed_df["content"])
        print(f"\nFinal row count (emoji-only and empty rows should be dropped): {transformed_df.shape[0]}")
        
    except MyException as e:
        print(f"Transformation transaction layer failed: {e}")
