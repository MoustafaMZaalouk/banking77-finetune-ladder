"""Rung 1: TF-IDF features plus logistic regression.

The classic sparse baseline, with the same hyperparameters as the
banking-intent-router batch-1 repo (max_features=20000, word unigrams and
bigrams, sublinear TF), so the numbers are directly comparable. The
pipeline is deterministic: one run, no seed variance to report.

The vectorizer vocabulary is fit on the training split only; the test
split is only ever transformed. ``tests/test_leakage.py`` pins that.
"""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.harness import compute_metrics

RUNG = "tfidf_lr"
TFIDF_MAX_FEATURES = 20000
LR_MAX_ITER = 1000
LR_SEED = 42  # lbfgs is deterministic given fixed data; recorded anyway


def train_tfidf_lr(train_texts, train_labels,
                   max_features=TFIDF_MAX_FEATURES):
    """Fit TF-IDF (train-split vocabulary only) plus logistic regression."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(
        max_iter=LR_MAX_ITER, random_state=LR_SEED)
    classifier.fit(features, train_labels)
    return vectorizer, classifier


def run(train_texts, train_labels, test_texts, test_labels, **_):
    """Train and evaluate rung 1. Returns one harness row."""
    start = time.perf_counter()
    vectorizer, classifier = train_tfidf_lr(train_texts, train_labels)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = classifier.predict(vectorizer.transform(test_texts))
    eval_seconds = time.perf_counter() - start

    metrics = compute_metrics(test_labels, predictions)
    return {
        "rung": RUNG,
        "seed": None,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "n_train": len(train_texts),
    }
