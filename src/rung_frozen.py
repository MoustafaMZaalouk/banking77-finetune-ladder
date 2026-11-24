"""Rung 2: frozen all-MiniLM-L6-v2 embeddings plus logistic regression.

The sentence encoder is used exactly as released: weights are frozen, we
only call ``encode``. The only trained component is the scikit-learn
logistic regression on top, same as in the banking-intent-router batch-1
repo. Encoding with fixed weights is deterministic, so this rung runs
once and reports no seed variance.
"""

import time

from sklearn.linear_model import LogisticRegression

from src.data import MODEL_NAME, MODEL_REVISION
from src.harness import compute_metrics

RUNG = "frozen_minilm_lr"
ENCODE_BATCH_SIZE = 64
LR_MAX_ITER = 1000
LR_SEED = 42


def frozen_encoder(model_name=MODEL_NAME, revision=MODEL_REVISION):
    """Load the pinned pretrained encoder. Weights are never trained here."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision, device="cpu")


def encode(encoder, texts):
    """L2-normalized 384-d embeddings, matching the batch-1 baseline."""
    return encoder.encode(
        list(texts),
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def run(train_texts, train_labels, test_texts, test_labels, encoder=None,
        **_):
    """Train and evaluate rung 2. Returns one harness row."""
    if encoder is None:
        encoder = frozen_encoder()

    start = time.perf_counter()
    train_features = encode(encoder, train_texts)
    classifier = LogisticRegression(
        max_iter=LR_MAX_ITER, random_state=LR_SEED)
    classifier.fit(train_features, train_labels)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = classifier.predict(encode(encoder, test_texts))
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
