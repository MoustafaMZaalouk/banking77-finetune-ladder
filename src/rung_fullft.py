"""Rung 5: full fine-tune of all-MiniLM-L6-v2 on the full train split.

Unlike the frozen rung, every weight is trained here: the transformer
body starts from the pinned pretrained checkpoint, a fresh linear
classification head on the model's sentence embedding is initialized from
the run's seed, and both are updated with AdamW on all 10,003 training
queries.

Two choices were made on train-held-out evidence, not the test split:

- READOUT. The classifier reads the mean-pooled, L2-normalized sentence
  embedding (the readout this checkpoint is trained for), not the BERT
  [CLS] pooler. In a 200-step probe on train-held-out queries the
  mean-pooled readout reached 0.669 accuracy against 0.497 for the
  [CLS] readout.
- LEARNING_RATE. The common 2e-5 fine-tuning rate badly undertrained
  within this CPU budget: per-step Adam updates of ~2e-5 over 1,250
  steps move each weight about 0.025, which cannot separate 77 classes
  (test accuracy 0.4302, kept in results as `full_finetune_undertrained`).
  1e-4 learns properly in the same budget.

Everything runs on CPU with a deliberately small, stated budget
(EPOCHS passes at batch size BATCH_SIZE, sequence length capped at
MAX_LENGTH). The loop is plain PyTorch rather than a trainer framework so
the training behavior is readable end to end.
"""

import time

import numpy as np

from src.data import MODEL_NAME, MODEL_REVISION
from src.harness import compute_metrics, set_global_seed

RUNG = "full_finetune"

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
EPOCHS = 2
MAX_LENGTH = 64
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
EVAL_BATCH_SIZE = 64
EMBEDDING_DIM = 384


def run(train_texts, train_labels, test_texts, test_labels, num_labels,
        seed, epochs=EPOCHS, max_steps=None, **_):
    """Fine-tune and evaluate for one seed. Returns one harness row."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    set_global_seed(seed)
    body = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION,
                               device="cpu")
    body.max_seq_length = MAX_LENGTH
    head = torch.nn.Linear(EMBEDDING_DIM, num_labels)
    parameters = list(body.parameters()) + list(head.parameters())
    body.train()
    head.train()

    def forward(texts):
        return head(body(body.tokenize(list(texts)))
                    ["sentence_embedding"])

    rng = np.random.default_rng(seed)
    steps_per_epoch = -(-len(train_texts) // BATCH_SIZE)
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    start = time.perf_counter()
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(train_texts))
        for begin in range(0, len(train_texts), BATCH_SIZE):
            picked = order[begin:begin + BATCH_SIZE]
            logits = forward([train_texts[i] for i in picked])
            labels = torch.tensor([int(train_labels[i]) for i in picked])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    body.eval()
    head.eval()
    predictions = []
    with torch.no_grad():
        for begin in range(0, len(test_texts), EVAL_BATCH_SIZE):
            logits = forward(test_texts[begin:begin + EVAL_BATCH_SIZE])
            predictions.extend(logits.argmax(dim=-1).tolist())
    eval_seconds = time.perf_counter() - start

    metrics = compute_metrics(test_labels, predictions)
    return {
        "rung": RUNG,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "n_train": len(train_texts),
    }
