"""Download the pinned BANKING77 files into data/ (safe to re-run)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_banking77

if __name__ == "__main__":
    dataset = load_banking77()
    print({split: len(dataset[split]) for split in dataset})
