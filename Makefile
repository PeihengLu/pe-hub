.PHONY: help setup install clean test format lint jupyter data-prep

help:
	@echo "PE-DB Project Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Set up the development environment"
	@echo "  make install        - Install all dependencies"
	@echo "  make clean          - Clean up generated files"
	@echo "  make test           - Run tests"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with flake8"
	@echo "  make jupyter        - Start Jupyter Lab locally"
	@echo "  make data-prep      - Prepare datasets"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  (Linux/macOS)"
	@echo "  venv\\Scripts\\activate     (Windows)"

install:
	pip install --upgrade pip
	pip install -e .
	pip install -e .[dev]
	pip install -e packages/pe-common
	python -m ipykernel install --user --name=pe-db --display-name="Python (PE-DB)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

test:
	pytest tests/ -v

format:
	black src/ services/ packages/

lint:
	flake8 src/ services/ packages/

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888

data-prep:
	bash setup.sh
