.PHONY: help install install-dev test lint format clean run-api run-demo etl-pipeline ml-pipeline inference docker-build docker-run

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
PYLINT := pylint
VENV := .venv

help:
	@echo "RevuAI Project Makefile"
	@echo "Available commands:"
	@echo "  make install          - Install dependencies from requirements.txt"
	@echo "  make install-dev      - Install dependencies + dev tools (pytest, black, isort)"
	@echo "  make test             - Run pytest test suite"
	@echo "  make lint             - Run code linting with pylint"
	@echo "  make format           - Format code with black and sort imports with isort"
	@echo "  make clean            - Remove cache files, __pycache__, .pytest_cache"
	@echo "  make clean-all        - Remove everything including venv"
	@echo "  make run-api          - Start FastAPI server"
	@echo "  make run-demo         - Run demo.py"
	@echo "  make etl-pipeline     - Run ETL pipeline"
	@echo "  make ml-pipeline      - Run ML training pipeline"
	@echo "  make inference        - Run inference pipeline"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"

# Setup targets
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest black isort pylint

# Testing targets
test:
	$(PYTEST) tests/ -v --tb=short

test-coverage:
	$(PYTEST) tests/ -v --cov=src --cov-report=html

# Code quality targets
lint:
	$(PYLINT) src/ tests/ --disable=C0111,C0103 || true

format:
	$(BLACK) src/ tests/ config/ utils/ router/ datasource/ --line-length=100
	$(ISORT) src/ tests/ config/ utils/ router/ datasource/ --profile=black

# Cleanup targets
clean:
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Directory -Name __pycache__ | ForEach-Object { Remove-Item -Path $$_ -Recurse -Force -ErrorAction SilentlyContinue }"
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Directory -Name .pytest_cache | ForEach-Object { Remove-Item -Path $$_ -Recurse -Force -ErrorAction SilentlyContinue }"
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Directory -Name .ipynb_checkpoints | ForEach-Object { Remove-Item -Path $$_ -Recurse -Force -ErrorAction SilentlyContinue }"
	powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Filter *.pyc | ForEach-Object { Remove-Item -Path $$_.FullName -Force -ErrorAction SilentlyContinue }"
	powershell -NoProfile -Command "Remove-Item -Path .coverage, htmlcov, .mypy_cache -Recurse -Force -ErrorAction SilentlyContinue"

clean-all: clean
	powershell -NoProfile -Command "Remove-Item -Path $(VENV) -Recurse -Force -ErrorAction SilentlyContinue"

# Application targets
run-api:
	uvicorn router.inference_router:app --reload --host 0.0.0.0 --port 8000

run-demo:
	$(PYTHON) demo.py

# Pipeline targets
etl-pipeline:
	$(PYTHON) -m src.pipelines.etl_pipeline

ml-pipeline:
	$(PYTHON) -m src.pipelines.ml_pipeline

inference-pipeline:
	$(PYTHON) -m src.pipelines.inference_pipeline

# Docker targets
docker-build:
	docker build -t revuai:latest .

docker-run:
	docker run --rm -p 8000:8000 revuai:latest

# Development targets
notebook:
	jupyter notebook notebooks/

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# CI/CD targets
ci: format lint test
	@echo "CI checks passed!"

# Quick start
setup: install-dev
	@echo "Setup complete! Run 'make help' for available commands."
