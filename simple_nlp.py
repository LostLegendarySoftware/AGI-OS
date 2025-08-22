"""Simple sentiment analysis using NLTK tokenization.

This script tokenizes input text and performs a very basic rule-based
sentiment analysis using small positive and negative word lists.
It demonstrates a minimal natural language processing pipeline
as the repository lacks a realistic NLP implementation.
"""

from __future__ import annotations

import sys
from typing import Dict

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "love",
    "wonderful",
    "positive",
    "fortunate",
    "correct",
    "superior",
}
NEGATIVE_WORDS = {
    "bad",
    "awful",
    "poor",
    "sad",
    "hate",
    "terrible",
    "negative",
    "unfortunate",
    "wrong",
    "inferior",
}

def analyze_sentiment(text: str) -> Dict[str, int | str]:
    """Tokenize text and compute a simple sentiment score."""
    tokens = word_tokenize(text.lower())
    pos_count = sum(token in POSITIVE_WORDS for token in tokens)
    neg_count = sum(token in NEGATIVE_WORDS for token in tokens)

    sentiment: str
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"positive": pos_count, "negative": neg_count, "sentiment": sentiment}


def main() -> None:
    """Run a simple analysis using text from arguments or stdin."""
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    if not text.strip():
        print("No input provided.")
        return
    result = analyze_sentiment(text)
    print(result)


if __name__ == "__main__":
    main()
