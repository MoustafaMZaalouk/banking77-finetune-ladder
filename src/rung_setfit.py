"""Rungs 3 and 4: SetFit few-shot contrastive fine-tuning.

SetFit (Tunstall et al., 2022) fine-tunes the sentence-encoder body on
seeded positive/negative sentence pairs, then fits a logistic-regression
head on the tuned embeddings. Here the body is the pinned
all-MiniLM-L6-v2 checkpoint and the head is scikit-learn, the SetFit
library default.

Compute budget, stated plainly: both few-shot rungs get the same
contrastive training budget of about 30,800 sentence pairs. That is the
SetFit default (num_iterations=20) at 10 examples per class, and
num_iterations=2 at 100 examples per class. Matching the budget keeps the
comparison about sample size, not about one rung getting more compute.
The 100/class sample is capped at class size (BANKING77 train has 11
classes with <100 examples, smallest 35), giving 7,438 queries and about
29,800 pairs; see data.stratified_few_shot_indices.

Every run uses one of three seeds (harness.SEEDS); the seed controls both
the few-shot sample and the trainer RNG.
"""

import tempfile
import time

from src.data import (
    MODEL_NAME,
    MODEL_REVISION,
    stratified_few_shot_indices,
)
from src.harness import compute_metrics, set_global_seed

RUNG_BY_PER_CLASS = {10: "setfit_10", 100: "setfit_100"}

# Contrastive iterations per rung; see the module docstring.
NUM_ITERATIONS = {10: 20, 100: 2}

BATCH_SIZE = 16
NUM_EPOCHS = 1
MAX_LENGTH = 64


def run(train_texts, train_labels, test_texts, test_labels, per_class,
        seed, num_iterations=None, **_):
    """Train and evaluate one SetFit rung for one seed. Returns a row."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    rung = RUNG_BY_PER_CLASS[per_class]
    if num_iterations is None:
        num_iterations = NUM_ITERATIONS[per_class]

    indices = stratified_few_shot_indices(train_labels, per_class, seed)
    few_shot = Dataset.from_dict({
        "text": [train_texts[i] for i in indices],
        "label": [int(train_labels[i]) for i in indices],
    })

    # Bound the contrastive pair sampler's memory. setfit's
    # ContrastiveDataset enumerates all n(n+1)/2 index combinations and,
    # with max_pairs unset, keeps EVERY negative pair in a Python list -
    # at 7,438 queries (100/class rung) that is ~27M dicts and the
    # process is OOM-killed. Setting max_steps makes the library stop
    # collecting at max_steps*batch_size pairs (trainer.py passes
    # train_max_pairs = max_steps * embedding_batch_size). The trained
    # pairs are unchanged: pair order is the fixed RandomState(42)
    # permutation of the same combinations and iteration always starts
    # at index 0, so stopping collection early only drops pairs that
    # were never reached anyway. max_steps equals the default one-epoch
    # step count, so the schedule and number of updates are identical.
    n_pairs = 2 * num_iterations * len(indices)  # positive + negative
    max_steps = -(-n_pairs // BATCH_SIZE)  # ceil

    set_global_seed(seed)
    model = SetFitModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    with tempfile.TemporaryDirectory() as output_dir:
        arguments = TrainingArguments(
            output_dir=output_dir,
            num_iterations=num_iterations,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            max_steps=max_steps,
            seed=seed,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            show_progress_bar=False,
        )
        trainer = Trainer(
            model=model, args=arguments, train_dataset=few_shot)

        start = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(list(test_texts))
        eval_seconds = time.perf_counter() - start

    metrics = compute_metrics(test_labels, predictions)
    return {
        "rung": rung,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "n_train": len(indices),
    }
