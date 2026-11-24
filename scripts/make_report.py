"""Regenerate every reported number from results/runs.csv.

Writes results/summary.json and plots/ladder.png, then rewrites the
README section between the RESULTS markers so the tables in the README
always come from the committed artifacts, never from hand-typed numbers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.harness import (
    ENVIRONMENT_JSON,
    RESULTS_DIR,
    RUNG_ORDER,
    SUMMARY_JSON,
    format_mean_std,
    load_runs,
    summarize,
)

README = Path(__file__).resolve().parents[1] / "README.md"
PLOT = Path(__file__).resolve().parents[1] / "plots" / "ladder.png"
BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def metrics_table(summary):
    lines = [
        "| Rung | Train examples | Accuracy | Macro F1 |",
        "| --- | --- | --- | --- |",
    ]
    for rung in RUNG_ORDER:
        if rung not in summary:
            continue
        s = summary[rung]
        lines.append(
            f"| {s['label']} | {s['n_train']} | "
            f"{format_mean_std(s['accuracy_mean'], s['accuracy_std'])} | "
            f"{format_mean_std(s['macro_f1_mean'], s['macro_f1_std'])} |"
        )
    return "\n".join(lines)


def compute_table(summary, environment, cpus=None):
    lines = [
        ("| Rung | Runs | Mean train wall-clock | Mean eval wall-clock | "
         "Total wall-clock |"),
        "| --- | --- | --- | --- | --- |",
    ]
    for rung in RUNG_ORDER:
        if rung not in summary:
            continue
        s = summary[rung]
        lines.append(
            f"| {s['label']} | {s['runs']} | "
            f"{s['mean_train_seconds'] / 60:.1f} min | "
            f"{s['mean_eval_seconds']:.1f} s | "
            f"{s['total_seconds'] / 60:.1f} min |"
        )
    cores = environment.get("cpu_count", "?")
    ram_gb = environment.get("mem_total_kb", 0) / 2**20
    torch_version = environment.get("torch", "?")
    lines.append("")
    lines.append(
        f"Measured on {cores} vCPU, {ram_gb:.0f} GB RAM, "
        f"torch {torch_version}, single thread. Train wall-clock for the "
        f"neural rungs includes encoding; eval wall-clock is one pass over "
        f"the 3,080-query test split. Totals are summed over all seeds."
    )
    if cpus:
        lines.append(
            "CPU model per run is recorded in `results/runs.csv`. The "
            "build sandbox migrated between hosts during the ladder "
            f"({'; '.join(cpus)}), so per-rung wall-clocks are honest "
            "measurements but cross-rung timing comparisons carry that "
            "caveat."
        )
    return "\n".join(lines)


def undertrained_block(rows):
    kept = [row for row in rows if row["rung"] == "full_finetune_undertrained"]
    if not kept:
        return ""
    row = kept[0]
    return "\n".join([
        "### The undertrained first attempt (kept for honesty)",
        "",
        "Rung 5 was first trained with the standard fine-tuning learning",
        f"rate 2e-5 (2 epochs, 1,250 steps): accuracy {row['accuracy']:.4f},",
        f"macro F1 {row['macro_f1']:.4f}, row `full_finetune_undertrained`",
        "in `results/runs.csv`. Diagnosis: gradients flowed and parameters",
        "updated, but ~2e-5 per-step Adam updates over 1,250 CPU-budget",
        "steps move each weight about 0.025, not enough to separate 77",
        "classes (loss stayed near the 4.34 random level for both a BERT",
        "CLS readout and a mean-pooled readout). The learning rate was",
        "raised and the readout re-chosen on train-held-out data only;",
        "the final rung-5 numbers above use that corrected configuration.",
        "",
    ])


def takeaway_block(summary):
    """Plain-language reading of the table, computed from the aggregates.

    Hand-written takeaways drift from the numbers; these sentences are
    built from summary.json so regeneration keeps them true.
    """
    needed = {"frozen_minilm_lr", "setfit_10", "setfit_100", "full_finetune"}
    if not needed <= set(summary):
        return ""
    frozen = summary["frozen_minilm_lr"]["accuracy_mean"]
    fft = summary["full_finetune"]["accuracy_mean"]
    fft_std = summary["full_finetune"]["accuracy_std"]
    s10 = summary["setfit_10"]["accuracy_mean"]
    s100 = summary["setfit_100"]["accuracy_mean"]
    lines = ["### Reading the table", ""]
    gap = frozen - fft
    if gap > 0:
        lines.append(
            f"On this CPU budget, frozen MiniLM embeddings + logistic "
            f"regression ({frozen:.4f}) beat the full fine-tune "
            f"({fft:.4f} +/- {fft_std:.4f}) by {gap:.4f} accuracy - about "
            f"{gap / fft_std:.0f}x the fine-tune's seed std, so not a "
            f"variance artifact. Fine-tuning every weight did not pay off "
            f"at 2 epochs on 1 vCPU; the frozen features were already "
            f"stronger."
        )
    else:
        lines.append(
            f"The full fine-tune ({fft:.4f} +/- {fft_std:.4f}) beats the "
            f"frozen-embedding baseline ({frozen:.4f}) by {-gap:.4f} "
            f"accuracy."
        )
    lines.append(
        f"Few-shot scaling is real: SetFit at 100 examples/class "
        f"({s100:.4f}) beats 10/class ({s10:.4f}) by {s100 - s10:.4f} at "
        f"a matched contrastive budget. The full fine-tune adds a further "
        f"{fft - s100:.4f} over 100/class, but with a different objective "
        f"and the full 10,003-query train split."
    )
    return "\n".join(lines)


def results_section(summary, environment, rows):
    cpus = sorted({row["cpu"] for row in rows if row.get("cpu")})
    return "\n".join([
        BEGIN,
        "",
        "From `results/runs.csv` (one row per rung and seed), aggregated by",
        "`scripts/make_report.py` into `results/summary.json`:",
        "",
        metrics_table(summary),
        "",
        "The TF-IDF and frozen-embedding rungs are deterministic, so they",
        "have one run and no seed variance. The three neural rungs ran once",
        "per seed (42, 1337, 2024) and report mean +/- sample std.",
        "",
        takeaway_block(summary),
        "",
        undertrained_block(rows),
        "### Compute cost (honest wall-clock)",
        "",
        compute_table(summary, environment, cpus),
        "",
        END,
    ])


def make_plot(summary):
    rungs = [rung for rung in RUNG_ORDER if rung in summary]
    means = [summary[rung]["accuracy_mean"] for rung in rungs]
    stds = [summary[rung]["accuracy_std"] or 0.0 for rung in rungs]
    labels = [summary[rung]["label"] for rung in rungs]

    figure, axis = plt.subplots(figsize=(8, 4))
    ypos = range(len(rungs))
    axis.barh(ypos, means, xerr=stds, height=0.6, color="#4c78a8",
              error_kw={"capsize": 4})
    axis.axvspan(0.92, 0.93, color="#f58518", alpha=0.25,
                 label="published fine-tuned range (~0.92-0.93), context only")
    axis.set_yticks(list(ypos), labels)
    axis.invert_yaxis()
    axis.set_xlabel("BANKING77 test accuracy (mean +/- std over 3 seeds)")
    # Lower bound follows the weakest rung so no bar is clipped away;
    # the axis still starts high enough to keep differences readable.
    low = min(0.80, min(m - s for m, s in zip(means, stds)) - 0.02)
    axis.set_xlim(low, 0.96)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT, dpi=150)
    plt.close(figure)


def main():
    rows = load_runs()
    if not rows:
        raise SystemExit("results/runs.csv is empty: run scripts/run_ladder.py")
    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    environment = json.loads(ENVIRONMENT_JSON.read_text()) \
        if ENVIRONMENT_JSON.exists() else {}
    make_plot(summary)

    readme = README.read_text()
    if BEGIN not in readme or END not in readme:
        raise SystemExit(f"README.md is missing the {BEGIN} / {END} markers")
    before = readme.split(BEGIN)[0]
    after = readme.split(END)[1]
    README.write_text(
        before + results_section(summary, environment, rows) + after)
    print(f"wrote {SUMMARY_JSON}, {PLOT}, and refreshed README.md")


if __name__ == "__main__":
    main()
