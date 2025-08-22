"""Train a basic sentiment classifier using the NLTK movie reviews dataset."""

from __future__ import annotations

from typing import List, Tuple

import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data() -> Tuple[list[str], list[str]]:
    """Return documents and labels from the movie reviews corpus."""
    nltk.download("movie_reviews", quiet=True)
    docs: list[str] = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    labels: list[str] = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]
    return docs, labels


def train_and_evaluate() -> float:
    """Train a logistic regression classifier and return accuracy."""
    docs, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        docs, labels, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc


if __name__ == "__main__":
    train_and_evaluate()
