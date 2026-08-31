.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install install-all data validate ingest run check eval eval-agent test lint clean docker-up docker-down

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	python3 -m venv .venv

install: venv  ## Install runtime + dev dependencies (no model, no DB needed)
	$(PIP) install -q -r requirements-dev.txt

install-all: install  ## Additionally install the LLM, ML and PDF extras
	$(PIP) install -q -r requirements-llm.txt -r requirements-ml.txt -r requirements-pdf.txt

data:  ## Regenerate the synthetic invoices, labels and access map
	$(PY) scripts/make_dataset.py

validate:  ## Check the golden set is internally consistent
	$(PY) scripts/validate_dataset.py

ingest:  ## Parse, chunk, embed and index the contracts
	$(PY) -c "from app.ingest.pipeline import ingest; r = ingest(); print(r)"

run: ingest  ## Start the API on :8000
	.venv/bin/uvicorn app.main:app --reload --port 8000

check:  ## Check one invoice from the CLI: make check INVOICE=INV-008
	$(PY) scripts/check_invoice.py $(or $(INVOICE),INV-008)

eval:  ## Score the fixed pipeline against the labelled set
	$(PY) scripts/run_eval.py

eval-agent:  ## Score both retrieval strategies side by side
	$(PY) scripts/run_eval.py --compare

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## Lint
	.venv/bin/ruff check app eval scripts tests

clean:  ## Remove the index, traces and caches
	rm -rf var/ .pytest_cache __pycache__
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

docker-up:  ## Start Postgres+pgvector and the API
	docker compose up --build

docker-down:  ## Stop and remove containers
	docker compose down -v
