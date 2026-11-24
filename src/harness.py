"""Shared evaluation harness: every rung reports through the same code path.

All five rungs write one row per (rung, seed) to ``results/runs.csv`` with
accuracy and macro F1 on the official BANKING77 test split, plus honest
wall-clock timings. ``summarize`` turns those rows into the mean/std table
that ``scripts/make_report.py`` regenerates into the README, so no reported
number is ever hand-typed.
"""

import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"
ENVIRONMENT_JSON = RESULTS_DIR / "environment.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

# The neural rungs (SetFit x2, full fine-tune) run once per seed. The two
# scikit-learn rungs are deterministic, so they run once and their seed
# cell stays empty in runs.csv.
SEEDS = (42, 1337, 2024)

RUNG_ORDER = [
    "tfidf_lr",
    "frozen_minilm_lr",
    "setfit_10",
    "setfit_100",
    "full_finetune",
]

RUNG_LABELS = {
    "tfidf_lr": "TF-IDF + logistic regression",
    "frozen_minilm_lr": "Frozen MiniLM + logistic regression",
    "setfit_10": "SetFit, 10 examples/class",
    "setfit_100": "SetFit, 100 examples/class",
    "full_finetune": "Full fine-tune (all data)",
    # Not a ladder rung: the first full fine-tune attempt at lr 2e-5, kept
    # in the artifacts and README as a labeled undertrained configuration.
    "full_finetune_undertrained":
        "Full fine-tune, undertrained config (lr 2e-5)",
}

RUN_FIELDS = [
    "rung", "seed", "accuracy", "macro_f1", "train_seconds",
    "eval_seconds", "n_train", "cpu",
]


def set_global_seed(seed):
    """Seed every RNG the neural rungs use. CPU torch with one thread is
    deterministic, so identical seeds reproduce identical runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def compute_metrics(y_true, y_pred):
    """Accuracy and F1 scores for one rung's predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def append_run(row):
    """Append one run row to runs.csv, creating the file with a header."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RUN_FIELDS})


def load_runs():
    """All recorded runs as a list of dicts with typed values."""
    if not RUNS_CSV.exists():
        return []
    rows = []
    with open(RUNS_CSV, newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "rung": raw["rung"],
                "seed": int(raw["seed"]) if raw["seed"] else None,
                "accuracy": float(raw["accuracy"]),
                "macro_f1": float(raw["macro_f1"]),
                "train_seconds": float(raw["train_seconds"]),
                "eval_seconds": float(raw["eval_seconds"]),
                "n_train": int(raw["n_train"]),
                "cpu": raw.get("cpu") or "",
            })
    return rows


def completed_runs():
    """The set of (rung, seed) pairs already present in runs.csv."""
    return {(row["rung"], row["seed"]) for row in load_runs()}


def mean_std(values):
    """(mean, sample std) of a sequence; std is None for a single value."""
    values = list(values)
    if len(values) == 1:
        return values[0], None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize(rows=None):
    """Aggregate runs.csv rows into per-rung mean/std statistics."""
    rows = load_runs() if rows is None else rows
    summary = {}
    for rung in RUNG_ORDER:
        rung_rows = [row for row in rows if row["rung"] == rung]
        if not rung_rows:
            continue
        acc_mean, acc_std = mean_std([row["accuracy"] for row in rung_rows])
        f1_mean, f1_std = mean_std([row["macro_f1"] for row in rung_rows])
        summary[rung] = {
            "label": RUNG_LABELS[rung],
            "runs": len(rung_rows),
            "seeds": [row["seed"] for row in rung_rows],
            "accuracies": [row["accuracy"] for row in rung_rows],
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "n_train": rung_rows[0]["n_train"],
            "mean_train_seconds": float(np.mean(
                [row["train_seconds"] for row in rung_rows])),
            "mean_eval_seconds": float(np.mean(
                [row["eval_seconds"] for row in rung_rows])),
            "total_seconds": float(sum(
                row["train_seconds"] + row["eval_seconds"]
                for row in rung_rows)),
        }
    return summary


def format_mean_std(mean, std, digits=4):
    """Render '0.9123 +/- 0.0041', or plain mean for deterministic rungs."""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def environment_info():
    """Hardware and library versions, recorded next to the results."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal"):
                info["mem_total_kb"] = int(line.split()[1])
                break
    for package, module in [
        ("torch", "torch"), ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("setfit", "setfit"), ("scikit-learn", "sklearn"),
        ("datasets", "datasets"), ("numpy", "numpy"),
    ]:
        try:
            info[package] = __import__(module).__version__
        except ImportError:
            info[package] = None
    return info


def record_environment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_JSON.write_text(json.dumps(environment_info(), indent=2))
