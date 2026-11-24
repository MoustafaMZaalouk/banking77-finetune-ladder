"""Tiny end-to-end smoke run across all five rungs (used by CI).

Trains every rung on a 5-intent, 12-examples-per-intent subsample with a
drastically reduced budget, evaluates on 100 held-out queries, and asserts
each rung learns above the 1/5 chance level. Proves the wiring works in
about a minute; the real numbers come from scripts/run_ladder.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import rung_frozen, rung_fullft, rung_setfit, rung_tfidf
from src.data import load_banking77

CLASSES = 5
TRAIN_PER_CLASS = 12
TEST_SIZE = 100
CHANCE = 1.0 / CLASSES


def main():
    dataset = load_banking77()
    train_texts = np.array(dataset["train"]["text"])
    train_labels = np.array(dataset["train"]["label"])
    keep_classes = list(range(CLASSES))
    mask = np.isin(train_labels, keep_classes)
    train_texts, train_labels = train_texts[mask], train_labels[mask]
    test_mask = np.isin(dataset["test"]["label"], keep_classes)
    test_texts = np.array(dataset["test"]["text"])[test_mask][:TEST_SIZE].tolist()
    test_labels = [int(v) for v in np.array(dataset["test"]["label"])[test_mask][:TEST_SIZE]]

    common = {
        "train_texts": train_texts.tolist(), "train_labels": train_labels,
        "test_texts": test_texts, "test_labels": test_labels,
        "num_labels": CLASSES,
    }
    encoder = rung_frozen.frozen_encoder()
    runs = {
        "tfidf_lr": rung_tfidf.run(**common),
        "frozen_minilm_lr": rung_frozen.run(encoder=encoder, **common),
        "setfit_10": rung_setfit.run(
            per_class=10, seed=0, num_iterations=1, **common),
        "setfit_100": rung_setfit.run(
            per_class=10, seed=0, num_iterations=1, **common),
        "full_finetune": rung_fullft.run(
            seed=0, epochs=1, max_steps=10, **common),
    }
    for name, row in runs.items():
        print(f"{name}: smoke accuracy {row['accuracy']:.3f} "
              f"(chance {CHANCE:.2f})")
        assert row["accuracy"] > CHANCE, f"{name} did not learn above chance"
    print("smoke run OK: all five rungs beat chance on a tiny subsample")


if __name__ == "__main__":
    main()
