.PHONY: help install install-dev install-ml lint fmt type test test-edge test-int check serve fleet plan autoscale load bench clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install core runtime deps + package
	pip install -r requirements.txt && pip install -e .

install-dev:  ## Install dev/test deps + package
	pip install -r requirements-dev.txt && pip install -e .

install-ml:  ## Install real GPU model backends (torch/transformers)
	pip install -r requirements-ml.txt

lint:  ## Lint with ruff
	ruff check src tests

fmt:  ## Auto-fix lint issues + format
	ruff check --fix src tests && ruff format src tests

type:  ## Type-check with mypy
	mypy src

test:  ## Run unit tests (no cluster needed)
	pytest tests -q --ignore=tests/test_integration_serve.py

test-edge:  ## Run only the industrial edge-case scenarios
	pytest tests/test_industrial_edge_cases.py -v

test-int:  ## Run the end-to-end Ray Serve integration test
	pytest tests/test_integration_serve.py -q

check: lint type test  ## Lint + type-check + unit tests

serve:  ## Deploy the app + fleet autoscaler locally
	rsa-serve --config config/models.yaml serve --blocking

fleet:  ## Run only the fleet controller (dry-run)
	rsa-serve --config config/models.yaml fleet --dry-run

plan:  ## Explain one control decision for every model and exit
	rsa-serve --config config/models.yaml plan

autoscale:  ## Run only the per-model latency autoscaler (dry-run)
	rsa-serve --config config/models.yaml autoscale --dry-run

load:  ## Ramp load against the sentiment model
	python scripts/load_test.py --model sentiment --ramp 5,20,60,20,5 --stage-seconds 30

bench:  ## Find each model's saturation knee and recommend autoscaling config
	python scripts/gpu_benchmark.py --all --output results/benchmark.json

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
