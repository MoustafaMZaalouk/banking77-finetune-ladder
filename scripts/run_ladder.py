"""Run the fine-tune ladder, one rung at a time, into results/runs.csv.

Each rung/seed combination is recorded exactly once; re-running the
script skips combinations already in runs.csv unless --force is given, so
a multi-hour ladder can be resumed after an interruption.

    python scripts/run_ladder.py                 # everything not yet run
    python scripts/run_ladder.py --rungs tfidf_lr,frozen_minilm_lr
    python scripts/run_ladder.py --rungs full_finetune --seeds 42
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rung_frozen, rung_fullft, rung_setfit, rung_tfidf
from src.data import load_banking77, split_texts_labels
from src.harness import (
    RUNG_ORDER,
    SEEDS,
    append_run,
    completed_runs,
    record_environment,
)

NEURAL_RUNGS = ("setfit_10", "setfit_100", "full_finetune")


def run_one(rung, seed, train, test, encoder):
    train_texts, train_labels, names = train
    test_texts, test_labels, _ = test
    common = {
        "train_texts": train_texts, "train_labels": train_labels,
        "test_texts": test_texts, "test_labels": test_labels,
        "num_labels": len(names),
    }
    if rung == "tfidf_lr":
        return rung_tfidf.run(**common)
    if rung == "frozen_minilm_lr":
        return rung_frozen.run(encoder=encoder, **common)
    if rung in ("setfit_10", "setfit_100"):
        per_class = 10 if rung == "setfit_10" else 100
        return rung_setfit.run(per_class=per_class, seed=seed, **common)
    if rung == "full_finetune":
        return rung_fullft.run(seed=seed, **common)
    raise ValueError(f"unknown rung {rung}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", default=",".join(RUNG_ORDER),
                        help="comma-separated subset of " + ",".join(RUNG_ORDER))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="comma-separated seeds for the neural rungs")
    parser.add_argument("--force", action="store_true",
                        help="re-run combinations already in runs.csv")
    args = parser.parse_args()

    rungs = args.rungs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    unknown = set(rungs) - set(RUNG_ORDER)
    if unknown:
        parser.error(f"unknown rungs: {sorted(unknown)}")

    dataset = load_banking77()
    train = split_texts_labels(dataset, "train")
    test = split_texts_labels(dataset, "test")
    record_environment()

    done = set() if args.force else completed_runs()
    cpu = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    encoder = None  # the frozen encoder is loaded only if a rung needs it
    for rung in rungs:
        for seed in (seeds if rung in NEURAL_RUNGS else [None]):
            if (rung, seed) in done:
                print(f"skip {rung} seed={seed}: already in runs.csv")
                continue
            if rung == "frozen_minilm_lr" and encoder is None:
                encoder = rung_frozen.frozen_encoder()
            label = f"{rung}" + (f" seed={seed}" if seed is not None else "")
            print(f"[{time.strftime('%H:%M:%S')}] start {label}", flush=True)
            row = run_one(rung, seed, train, test, encoder)
            row["cpu"] = cpu
            append_run(row)
            print(f"[{time.strftime('%H:%M:%S')}] done  {label}: "
                  f"accuracy={row['accuracy']:.4f} "
                  f"macro_f1={row['macro_f1']:.4f} "
                  f"train={row['train_seconds']:.0f}s "
                  f"eval={row['eval_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
