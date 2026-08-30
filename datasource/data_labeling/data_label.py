"""
Label App reviews using an open-source Hugging Face LLM/model.

Labels:
    Positive
    Negative
    Neutral

CLI testing:
    Enter a review in the terminal and get its sentiment.
"""

import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# Load the open-source sentiment classification model once
sentiment_pipe = pipeline(
    "text-classification",
    model=MODEL_NAME
)


def normalize_label(label):
    """Convert model labels to Positive, Negative, Neutral."""

    label = label.lower()

    if "positive" in label:
        return "Positive"

    if "negative" in label:
        return "Negative"

    if "neutral" in label:
        return "Neutral"

    return label


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

        predictions = sentiment_pipe(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )

        results.extend(
            [normalize_label(pred["label"]) for pred in predictions]
        )

    return results


# ---------------------------------------------------------
# CLI Testing
# ---------------------------------------------------------

def cli_testing():
    print("\n" + "=" * 60)
    print("       App Review Sentiment Classifier")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        review = input("Enter app review: ").strip()

        if review.lower() in ["exit", "quit"]:
            print("\nExiting...")
            break

        if not review:
            print("Please enter a review.\n")
            continue

        prediction = sentiment_pipe(
            review,
            truncation=True,
            max_length=512
        )[0]

        label = normalize_label(prediction["label"])
        score = prediction["score"]

        print(f"\nSentiment : {label}")
        print(f"Confidence: {score:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    cli_testing()