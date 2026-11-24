PY ?= python3

.PHONY: install data ladder report test lint smoke all

# --extra-index-url serves the smaller CPU-only torch wheel (2.14.1+cpu);
# without it pip installs the larger PyPI CUDA build of the same version.
install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu

data:
	$(PY) scripts/download_data.py

# Full ladder: about 2.5 hours on 1 vCPU. Safe to re-run; completed
# rung/seed combinations in results/runs.csv are skipped.
ladder:
	$(PY) scripts/run_ladder.py

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check src/ scripts/ tests/

smoke:
	$(PY) scripts/smoke_train.py

all: data ladder report
