"""Loading helpers for the BANKING77 dataset, pinned by revision and checksum.

BANKING77 (Casanueva et al., 2020) contains 13,083 customer-service
queries labelled with one of 77 banking intents. It is distributed under
CC BY 4.0 at https://huggingface.co/datasets/PolyAI/banking77.

Nothing from the dataset is committed to this repository. ``make data``
downloads the two parquet shards and the dataset card from the pinned
Hugging Face revision below and verifies their SHA-256 checksums before
anything is read, so a silent upstream change fails loudly instead of
silently changing the numbers.
"""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import yaml
from datasets import ClassLabel, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_NAME = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"

# SHA-256 of each file at DATASET_REVISION, recorded when the pin was set.
# Update only after inspecting the upstream change.
EXPECTED_SHA256 = {
    "data/train-00000-of-00001.parquet":
        "8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a",
    "data/test-00000-of-00001.parquet":
        "1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341",
    "README.md":
        "f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1",
}

# The sentence encoder used by every neural rung, pinned to one revision.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_banking77(data_dir=DATA_DIR):
    """Download the pinned dataset files into ``data/`` and verify checksums.

    Safe to re-run: files already present with the right checksum are kept.
    Raises on any checksum mismatch rather than reading altered data.
    """
    data_dir = Path(data_dir)
    for relative, expected in EXPECTED_SHA256.items():
        target = data_dir / Path(relative).name if relative.startswith("data/") \
            else data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = (f"https://huggingface.co/datasets/{DATASET_NAME}"
                   f"/resolve/{DATASET_REVISION}/{relative}")
            with urlopen(url, timeout=120) as response:
                blob = response.read()
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(blob)
            temporary.replace(target)
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected}, got "
                f"{actual}. The pinned dataset changed upstream; inspect the "
                f"change before updating EXPECTED_SHA256."
            )
    return data_dir


def label_names(data_dir=DATA_DIR):
    """The 77 intent names, parsed from the pinned dataset card."""
    readme = Path(data_dir) / "README.md"
    front_matter = readme.read_text().split("---")[1]
    info = yaml.safe_load(front_matter)["dataset_info"]
    label_feature = next(f for f in info["features"] if f["name"] == "label")
    names = label_feature["dtype"]["class_label"]["names"]
    return [names[str(index)] for index in range(len(names))]


def load_banking77(data_dir=DATA_DIR):
    """Load BANKING77 from the pinned local files, downloading them first."""
    data_dir = download_banking77(data_dir)
    dataset = load_dataset("parquet", data_files={
        "train": str(Path(data_dir) / "train-00000-of-00001.parquet"),
        "test": str(Path(data_dir) / "test-00000-of-00001.parquet"),
    })
    names = label_names(data_dir)
    for split in dataset:
        dataset[split] = dataset[split].cast_column(
            "label", ClassLabel(names=names))
    return dataset


def split_texts_labels(dataset, split):
    """Return (texts, numeric_labels, label_names) for one split."""
    texts = dataset[split]["text"]
    labels = dataset[split]["label"]
    names = dataset[split].features["label"].names
    return texts, labels, names


def stratified_few_shot_indices(labels, per_class, seed):
    """Seeded indices with up to ``per_class`` examples of each label.

    Used by the few-shot rungs. Deterministic for a given seed, and the
    returned indices only ever point into the array they were given, so
    passing the train labels can never yield test examples.

    Sampling is capped at the class size: BANKING77 train has 11 classes
    with fewer than 100 examples (smallest has 35), so the 100-per-class
    rung contributes every example of those classes rather than sampling
    with replacement (duplicates would distort contrastive pair mining).
    This matches setfit.sample_dataset behaviour.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(labels):
        where = np.where(labels == label)[0]
        size = min(per_class, len(where))
        keep.extend(rng.choice(where, size=size, replace=False).tolist())
    return sorted(keep)
