.PHONY: install install-dev format lint test test-coverage test-coverage-report eval eval-verbose clean run

install:
	uv sync

install-dev:
	uv sync --all-extras

format:
	uv run black src/
	uv run ruff check --fix src/

lint:
	uv run ruff check src/
	uv run black --check src/

test:
	uv run pytest

test-coverage:
	uv run pytest --cov=oh_no_mcp_server --cov-report=term-missing --cov-report=html

test-coverage-report:
	@echo "Opening coverage report in browser..."
	open htmlcov/index.html || xdg-open htmlcov/index.html

eval:
	uv run pytest evals/ -v --no-cov

eval-verbose:
	uv run pytest evals/ -v -s --no-cov

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist build

run:
	uv run oh-no-mcp
