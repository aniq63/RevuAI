import sys
from pathlib import Path
import pickle

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from utils.logger import logging
from utils.exception import MyException
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureTransformation:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

    def apply_tf_idf(self, X_train: pd.Series, X_test: pd.Series):
        """
        Fits TF-IDF on the training text and transforms both train and test sets.
        """
        try:
            logging.info("Start the Feature Transformation")
            logging.info("Applying the TF-IDF vectorization...")
            
            # Fit and transform training data, transform test data
            X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
            
            logging.info("TF-IDF Transformation completed successfully")
            return (
                X_train_tfidf,
                X_test_tfidf,
                self.tfidf_vectorizer
            )
            
        except Exception as e:
            logging.error("Exception occurred during TF-IDF transformation")
            raise MyException(e, sys)


# ==============================================
# CLI TESTING
# ==============================================
if __name__ == "__main__":
    try:
        logging.info("Initializing CLI Test for FeatureTransformation")
        
        # 1. Create dummy text data (ensuring words repeat to satisfy min_df=2)
        train_data = pd.Series([
            "machine learning is great and machine learning is fun",
            "natural language processing and machine learning",
            "tf idf vectorizer transforms text data into numbers",
            "text data processing is essential for machine learning"
        ])
        
        test_data = pd.Series([
            "machine learning and text processing",
            "data science is fun"
        ])
        
        # 2. Initialize and run transformation
        transform_obj = FeatureTransformation()
        X_train_vec, X_test_vec, vectorizer = transform_obj.apply_tf_idf(train_data, test_data)
        
        print(f"Train matrix shape: {X_train_vec.shape}")
        print(f"Test matrix shape: {X_test_vec.shape}")

        logging.info("CLI Test completed without errors.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
