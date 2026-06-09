# Makefile
.PHONY: help install test lint format format-check type-check build clean all

help:
	@echo "ilma — development targets"
	@echo "  make install     Install dev dependencies via uv"
	@echo "  make test        Run pytest"
	@echo "  make lint        Run ruff check"
	@echo "  make format      Run ruff format (writes)"
	@echo "  make type-check  Run mypy"
	@echo "  make build       Build docker image"
	@echo "  make all         lint + format-check + test"
	@echo "  make clean       Remove build artifacts"

install:
	uv sync --group dev

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

type-check:
	uv run mypy src/ilma --ignore-missing-imports

build:
	docker build -t ilma:test .

all: lint format-check test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
