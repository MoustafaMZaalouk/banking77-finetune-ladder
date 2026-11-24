# BANKING77 Fine-Tune Ladder

[![CI](https://github.com/MoustafaMZaalouk/banking77-finetune-ladder/workflows/CI/badge.svg)](https://github.com/MoustafaMZaalouk/banking77-finetune-ladder/blob/main/.github/workflows/ci.yml)

*Live workflow status, not a production-readiness or model-quality rating.*

One evaluation harness, five training budgets, on the real
[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) intent
dataset (77 banking intents, 10,003 train / 3,080 test queries):

1. TF-IDF + logistic regression
2. Frozen `all-MiniLM-L6-v2` embeddings + logistic regression
3. SetFit, 10 examples per intent
4. SetFit, 100 examples per intent
5. Full fine-tune of `all-MiniLM-L6-v2` on the full train split

Rungs 1-2 are the baselines measured in the companion
[banking-intent-router](https://github.com/MoustafaMZaalouk/banking-intent-router)
repo, rerun here under this harness. This repo answers the question that
repo left open: what does each extra unit of training actually buy on a
student CPU budget?

## Honesty notes (read first)

- This project was built with AI assistance (Instinct agent), and this
  README says so because pretending otherwise would be dishonest.
- Exactly which weights are trained depends on the rung, and the code
  makes it structural: rungs 1-2 train only scikit-learn logistic
  regression (the MiniLM encoder is frozen, `tests/test_leakage.py`
  pins that the frozen rung has no fit path), rungs 3-4 fine-tune the
  encoder on seeded contrastive pairs, rung 5 trains every weight
  including a freshly initialized classification head.
- Published fine-tuned transformer results on BANKING77 are around
  92-93% accuracy. That range appears below only as context; everything
  in the tables is what this repo measured, on CPU, with the stated
  budget. No SOTA claim is made.
- Every reported number is regenerated from `results/runs.csv` by
  `scripts/make_report.py` (it rewrites the results section of this
  README). Nothing is hand-typed.
- Dataset hygiene: BANKING77 contains 6 identical texts in both the
  train and test splits. This is pinned by
  `tests/test_data.py::test_train_test_text_overlap_is_the_documented_six`
  so any silent data change fails loudly. The few-shot rungs sample from
  the official train split, so those documented duplicates can appear in
  a sample; that is a property of the dataset, not added leakage.
- The dataset is not committed. `make data` downloads the pinned Hugging
  Face revision and verifies SHA-256 checksums before reading anything.
- Rung-5's readout and learning rate are the only choices in this repo
  that were tuned at all, and they were tuned on train-held-out queries
  only. Everything else uses documented defaults.
- One library-internals workaround exists for memory, not accuracy:
  setfit's pair sampler enumerates all n(n+1)/2 index combinations and
  keeps every pair it collects, which OOM-kills a 2 GB box at 7,438
  queries. The rungs therefore pass `max_steps` (exactly the default
  one-epoch step count) so the sampler stops collecting pairs it would
  never reach; the trained pairs, their order, the schedule and the
  step count are identical to the default (see the comment in
  `src/rung_setfit.py`).

## Results (BANKING77 test split, 3,080 queries)

<!-- RESULTS:BEGIN -->

From `results/runs.csv` (one row per rung and seed), aggregated by
`scripts/make_report.py` into `results/summary.json`:

| Rung | Train examples | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 10003 | 0.8571 | 0.8558 |
| Frozen MiniLM + logistic regression | 10003 | 0.9081 | 0.9081 |
| SetFit, 10 examples/class | 770 | 0.7871 +/- 0.0064 | 0.7837 +/- 0.0062 |
| SetFit, 100 examples/class | 7438 | 0.8759 +/- 0.0033 | 0.8757 +/- 0.0032 |
| Full fine-tune (all data) | 10003 | 0.8828 +/- 0.0031 | 0.8764 +/- 0.0038 |

The TF-IDF and frozen-embedding rungs are deterministic, so they
have one run and no seed variance. The three neural rungs ran once
per seed (42, 1337, 2024) and report mean +/- sample std.

### Reading the table

On this CPU budget, frozen MiniLM embeddings + logistic regression (0.9081) beat the full fine-tune (0.8828 +/- 0.0031) by 0.0253 accuracy - about 8x the fine-tune's seed std, so not a variance artifact. Fine-tuning every weight did not pay off at 2 epochs on 1 vCPU; the frozen features were already stronger.
Few-shot scaling is real: SetFit at 100 examples/class (0.8759) beats 10/class (0.7871) by 0.0887 at a matched contrastive budget. The full fine-tune adds a further 0.0069 over 100/class, but with a different objective and the full 10,003-query train split.

### The undertrained first attempt (kept for honesty)

Rung 5 was first trained with the standard fine-tuning learning
rate 2e-5 (2 epochs, 1,250 steps): accuracy 0.4302,
macro F1 0.3628, row `full_finetune_undertrained`
in `results/runs.csv`. Diagnosis: gradients flowed and parameters
updated, but ~2e-5 per-step Adam updates over 1,250 CPU-budget
steps move each weight about 0.025, not enough to separate 77
classes (loss stayed near the 4.34 random level for both a BERT
CLS readout and a mean-pooled readout). The learning rate was
raised and the readout re-chosen on train-held-out data only;
the final rung-5 numbers above use that corrected configuration.

### Compute cost (honest wall-clock)

| Rung | Runs | Mean train wall-clock | Mean eval wall-clock | Total wall-clock |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 1 | 0.1 min | 0.1 s | 0.1 min |
| Frozen MiniLM + logistic regression | 1 | 0.6 min | 9.9 s | 0.7 min |
| SetFit, 10 examples/class | 3 | 31.6 min | 13.1 s | 95.6 min |
| SetFit, 100 examples/class | 3 | 30.9 min | 13.1 s | 93.3 min |
| Full fine-tune (all data) | 3 | 11.9 min | 29.8 s | 37.0 min |

Measured on 1 vCPU, 2 GB RAM, torch 2.14.1+cpu, single thread. Train wall-clock for the neural rungs includes encoding; eval wall-clock is one pass over the 3,080-query test split. Totals are summed over all seeds.
CPU model per run is recorded in `results/runs.csv`. The build sandbox migrated between hosts during the ladder (AMD EPYC 9655 96-Core Processor; Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz), so per-rung wall-clocks are honest measurements but cross-rung timing comparisons carry that caveat.

<!-- RESULTS:END -->

## How the ladder works

All rungs share one harness (`src/harness.py`): same train/test splits,
same metrics (accuracy and macro F1 on the official test split), same
results format. The neural rungs run once per seed (42, 1337, 2024) and
report mean +/- sample std.

| Rung | What is trained | Training data |
| --- | --- | --- |
| 1. TF-IDF + LR | logistic regression only | full train split |
| 2. Frozen MiniLM + LR | logistic regression only | full train split |
| 3. SetFit 10/class | encoder + LR head | 770 queries |
| 4. SetFit 100/class | encoder + LR head | 7,438 queries (capped, see below) |
| 5. Full fine-tune | all weights + new head on the mean-pooled embedding | full train split |

Rung 5 hyperparameters: AdamW, learning rate 1e-4, weight decay 0.01,
batch size 16, 2 epochs, linear schedule with 10% warmup, gradient
clipping at 1.0, sequences capped at 64 tokens. The classifier reads the
mean-pooled, L2-normalized sentence embedding. Both the readout and the
learning rate were chosen on train-held-out queries, never on the test
split: the standard 2e-5 rate undertrained within this CPU budget (kept
in the results as a labeled undertrained configuration), and the
mean-pooled readout learned faster than the BERT [CLS] pooler in a
matched probe. The SetFit rungs use the
library defaults with one deliberate change: both few-shot rungs get the
same contrastive budget of about 30,800 sentence pairs (the SetFit
default of 20 iterations at 10 examples/class, and 2 iterations at 100
examples/class), so the comparison is about sample size, not compute.
BANKING77's train split has 11 classes with fewer than 100 examples
(smallest: 35), so the 100/class rung is capped at class size for those
classes, like setfit.sample_dataset: 7,438 queries and about 29,800
pairs instead of 7,700 and 30,800.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # pinned deps; pulls the CPU torch wheel via
                    # --extra-index-url https://download.pytorch.org/whl/cpu
make data           # download + checksum-verify BANKING77 into data/
make ladder         # all five rungs -> results/runs.csv (see cost below)
make report         # summary.json, plots/ladder.png, README tables
make test           # dataset, harness, and leakage tests
make lint           # ruff
```

`make ladder` is resumable: rung/seed combinations already in
`results/runs.csv` are skipped, so an interrupted run continues where it
stopped. Use `python scripts/run_ladder.py --rungs tfidf_lr` to run a
single rung. `make smoke` exercises all five rungs on a tiny subsample in
about a minute and is what CI runs.

## Limitations

- Three seeds is enough to see the ordering and rough variance, not to
  establish tight confidence intervals.
- No validation split: rung 5 trains for a fixed 2 epochs and is
  evaluated once per seed on the test split. There is no early stopping
  or test-set model selection, but also no dev-set tuning, so the
  hyperparameters above are untuned defaults.
- The compute budget is a student CPU budget. More epochs, longer
  sequences, or larger encoders were not tried.
- BANKING77 is one English banking-domain dataset; none of these numbers
  transfer to other domains or languages as-is.
- The 6 documented train/test duplicate texts give every rung the same
  tiny, known optimism; they are pinned, not silently removed.

## Project layout

```
src/            data loading (pinned + checksummed), harness, one module per rung
scripts/        download, ladder runner, report generator, smoke run
tests/          dataset pins, harness math, leakage tests
results/        runs.csv (raw), summary.json, environment.json
plots/          ladder.png (regenerated by make report)
```

## Credits and licenses

- **Code in this repo:** MIT (see LICENSE). Copyright Moustafa Zaalouk.
- **Dataset:** [BANKING77](https://huggingface.co/datasets/PolyAI/banking77),
  CC BY 4.0. Citation: Casanueva et al., "Efficient Intent Detection with
  Dual Sentence Encoders", Proceedings of the 2nd Workshop on NLP for
  ConvAI - ACL 2020, https://arxiv.org/abs/2003.04807
- **Encoder:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  Apache 2.0. Frozen in rung 2, fine-tuned in rungs 3-5.
- **Few-shot method:** SetFit, Tunstall et al., 2022,
  https://arxiv.org/abs/2209.11055
