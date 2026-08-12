"""
Label the data using the Open Source llm from the Hugging face library

Label the App reviews into three sentiments Positive , Neagitive and Neutral
"""

import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

file_path = ""

# Load dataset
df = pd.read_csv(
    file_path,
    engine='python'
)

# Batch prediction function
def predict_sentiment_batch(texts, batch_size=32):
    results = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Classifying reviews"
    ):
        batch = texts[i:i + batch_size]

        # Handle missing values
        batch = [
            str(text) if pd.notna(text) else ""
            for text in batch
        ]

        # Load the opensource text classification model
        pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

        predictions = pipe(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )

        results.extend([pred["label"] for pred in predictions])

    return results


# Get sentiment labels
df["label"] = predict_sentiment_batch(
    df["content"].tolist(),
    batch_size=32
)

# Save labeled dataset
output_path = ""

df.to_csv(
    output_path,
    index=False
)

print(f"Saved labeled dataset to: {output_path}")