"""Dataset validation: pin, schema, label set, and train/test hygiene."""

import pytest

from src.data import (
    DATA_DIR,
    EXPECTED_SHA256,
    _sha256,
    label_names,
    load_banking77,
)

EXPECTED_N_LABELS = 77
EXPECTED_TRAIN_SIZE = 10003
EXPECTED_TEST_SIZE = 3080

# BANKING77 has no ID column, so we check exact normalized text instead.
# The released dataset contains 6 identical texts in both splits; this test
# pins that documented number so any silent data change fails loudly.
KNOWN_TRAIN_TEST_OVERLAP = 6


@pytest.fixture(scope="module")
def dataset():
    return load_banking77()


def test_downloaded_files_match_pinned_checksums():
    # The checksum pin is the whole point of the download step, so test it
    # directly rather than trusting that download_banking77 ran.
    for relative, expected in EXPECTED_SHA256.items():
        name = relative.split("/")[-1]
        assert _sha256(DATA_DIR / name) == expected, relative


def test_schema(dataset):
    for split in ("train", "test"):
        assert dataset[split].column_names == ["text", "label"]
        assert dataset[split].features["text"].dtype == "string"


def test_sizes_and_label_set(dataset):
    assert len(dataset["train"]) == EXPECTED_TRAIN_SIZE
    assert len(dataset["test"]) == EXPECTED_TEST_SIZE
    names = label_names()
    assert len(names) == EXPECTED_N_LABELS
    assert len(set(names)) == EXPECTED_N_LABELS
    assert names[0] == "activate_my_card"
    for split in ("train", "test"):
        assert set(dataset[split]["label"]) == set(range(EXPECTED_N_LABELS))


def test_no_empty_texts(dataset):
    for split in ("train", "test"):
        assert all(text.strip() for text in dataset[split]["text"])


def test_train_test_text_overlap_is_the_documented_six(dataset):
    normalize = lambda texts: {t.strip().lower() for t in texts}
    overlap = normalize(dataset["train"]["text"]) & \
        normalize(dataset["test"]["text"])
    assert len(overlap) == KNOWN_TRAIN_TEST_OVERLAP
