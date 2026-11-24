"""Harness math: metrics, aggregation, and the few-shot sampler."""

import numpy as np
import pytest

from src.data import stratified_few_shot_indices
from src.harness import (
    compute_metrics,
    format_mean_std,
    mean_std,
    summarize,
)


def test_compute_metrics_against_hand_computed():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # per-class F1: class0: P=1/2,R=1/2 -> 0.5; class1: P=2/3,R=1 -> 0.8;
    # class2: P=1,R=1/2 -> 2/3
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_mean_std_uses_sample_std():
    mean, std = mean_std([0.90, 0.92, 0.95])
    assert mean == pytest.approx(0.923333, abs=1e-5)
    assert std == pytest.approx(np.std([0.90, 0.92, 0.95], ddof=1))


def test_mean_std_single_run_has_no_std():
    mean, std = mean_std([0.857])
    assert mean == 0.857
    assert std is None
    assert format_mean_std(mean, std) == "0.8570"


def test_summarize_groups_by_rung():
    rows = [
        {"rung": "full_finetune", "seed": 42, "accuracy": 0.90,
         "macro_f1": 0.90, "train_seconds": 10.0, "eval_seconds": 2.0,
         "n_train": 10003},
        {"rung": "full_finetune", "seed": 1337, "accuracy": 0.92,
         "macro_f1": 0.92, "train_seconds": 12.0, "eval_seconds": 2.0,
         "n_train": 10003},
        {"rung": "tfidf_lr", "seed": None, "accuracy": 0.857,
         "macro_f1": 0.856, "train_seconds": 3.0, "eval_seconds": 0.1,
         "n_train": 10003},
    ]
    summary = summarize(rows)
    assert summary["full_finetune"]["runs"] == 2
    assert summary["full_finetune"]["accuracy_mean"] == pytest.approx(0.91)
    assert summary["tfidf_lr"]["accuracy_std"] is None
    assert summary["tfidf_lr"]["total_seconds"] == pytest.approx(3.1)


def test_few_shot_sampler_is_exact_and_deterministic():
    labels = [0] * 50 + [1] * 50 + [2] * 50
    first = stratified_few_shot_indices(labels, per_class=10, seed=42)
    again = stratified_few_shot_indices(labels, per_class=10, seed=42)
    other = stratified_few_shot_indices(labels, per_class=10, seed=1337)
    assert first == again
    assert first != other
    assert len(first) == 30
    for label in (0, 1, 2):
        assert sum(labels[i] == label for i in first) == 10


def test_few_shot_sampler_caps_at_class_size():
    # BANKING77 train has classes with <100 rows (smallest: 35); the
    # 100/class rung must take all of them rather than crash or duplicate.
    labels = [0] * 35 + [1] * 150
    indices = stratified_few_shot_indices(labels, per_class=100, seed=42)
    assert len(indices) == 135
    assert sum(labels[i] == 0 for i in indices) == 35
    assert sum(labels[i] == 1 for i in indices) == 100
    assert len(set(indices)) == len(indices)


def test_few_shot_sampler_stays_inside_the_given_split():
    # Leakage guard: sampling train labels can only return train indices.
    labels = list(range(77)) * 130  # 10,010 train-like rows
    indices = stratified_few_shot_indices(labels, per_class=100, seed=2024)
    assert len(indices) == 7700
    assert all(0 <= i < len(labels) for i in indices)
    assert len(set(indices)) == len(indices)
