.PHONY: install dev test lint format run clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies with uv"
	@echo "  dev          - Install with dev dependencies"
	@echo "  test         - Run pytest"
	@echo "  lint         - Run ruff check"
	@echo "  format       - Run ruff format"
	@echo "  run          - Execute main pipeline"
	@echo "  clean        - Remove artifacts"

install:
	uv sync

dev:
	uv sync --all-extras

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

run:
	uv run python -m src.main

run-video:
	@if [ -z "$(VIDEO)" ]; then \
		echo "Usage: make run-video VIDEO=path/to/video.mp4"; \
		exit 1; \
	fi
	uv run python -m src.main --source-video $(VIDEO)

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf *.egg-info dist build
	rm -rf .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
