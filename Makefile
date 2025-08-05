#!/bin/bash
# Development helper scripts

# Format code
format:
	black src/ tests/ *.py
	isort src/ tests/ *.py

# Lint code
lint:
	flake8 src/ tests/ *.py
	mypy src/

# Test
test:
	pytest tests/ -v --cov=src

# Test with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Install in development mode
install-dev:
	pip install -e .
	pip install -r requirements.txt

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Setup pre-commit hooks
setup-hooks:
	pre-commit install

# Run example evaluation
example:
	python main.py --config config.yaml --test-suite sample_test_suite.json --experiment-name example_run

# Validate config
validate-config:
	python -c "from src.config import ConfigManager; ConfigManager.load_config('config.yaml'); print('✅ Config is valid')"

.PHONY: format lint test test-cov install-dev clean setup-hooks example validate-config
