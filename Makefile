.PHONY: install test lint format clean

## Install project in editable mode with dev dependencies
install:
	pip install -e ".[dev]"

## Run all tests with coverage report
test:
	pytest -v --cov=src/energy_forecasting --cov-report=term-missing

## Check code style
lint:
	ruff check src/ tests/

## Auto-format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage