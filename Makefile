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
	@if command -v python3.11 >/dev/null 2>&1; then \
		python3.11 -m venv venv; \
	elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)'; then \
		python3 -m venv venv; \
	else \
		echo "Error: Python 3.11 required. Use: ./scripts/setup-python-env.sh" >&2; \
		exit 1; \
	fi
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  (Linux/macOS)"
	@echo "  venv\\Scripts\\activate     (Windows)"
	@echo "Then run: ./scripts/install-clis.sh"

install:
	./scripts/install-clis.sh

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
	bash scripts/setup.sh
