# =============================================================================
# Makefile for ML Classification Project
# =============================================================================
#
# STUDY NOTE: Makefiles
# ---------------------
# Makefiles provide a standard interface for common project tasks.
# Instead of remembering complex commands, just run "make <target>".
#
# Benefits:
# - Consistent commands across team
# - Self-documenting
# - Works in CI/CD pipelines
# - Cross-platform (with caveats)
#
# Usage:
#   make help      # Show available commands
#   make install   # Install dependencies
#   make test      # Run tests
#   make lint      # Check code quality
# =============================================================================

.PHONY: help install install-dev train api frontend test lint format clean docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo ""
	@echo "Iris Classification Project - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make setup        Full development setup"
	@echo ""
	@echo "Development:"
	@echo "  make train        Train the ML model"
	@echo "  make api          Start FastAPI server"
	@echo "  make frontend     Start Streamlit frontend"
	@echo ""
	@echo "Quality:"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make lint         Run all linters"
	@echo "  make format       Format code with black & isort"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make security     Run security checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker images"
	@echo "  make docker-run   Run with docker-compose"
	@echo "  make docker-stop  Stop docker-compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove generated files"
	@echo ""

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

setup: install-dev
	@echo "✓ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Run 'make train' to train the model"
	@echo "  3. Run 'make api' to start the server"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------
train:
	@echo "Training ML model..."
	python src/ml/train.py

eda:
	@echo "Running EDA..."
	python src/ml/eda.py

api:
	@echo "Starting FastAPI server..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting Streamlit frontend..."
	streamlit run src/frontend/app.py --server.port 8501

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -v -x --tb=short

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------
lint: format-check typecheck flake8
	@echo "✓ All checks passed!"

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✓ Code formatted!"

format-check:
	@echo "Checking code format..."
	black --check src/ tests/
	isort --check-only src/ tests/

flake8:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,E501

typecheck:
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

security:
	@echo "Running security checks..."
	bandit -r src/ -ll
	@echo "Checking dependencies..."
	safety check -r requirements.txt || true

# -----------------------------------------------------------------------------
# Pre-commit
# -----------------------------------------------------------------------------
pre-commit:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build:
	@echo "Building Docker images..."
	docker-compose -f docker/docker-compose.yml build

docker-run:
	@echo "Starting containers..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo ""
	@echo "Services:"
	@echo "  API: http://localhost:8000"
	@echo "  Frontend: http://localhost:8501"

docker-stop:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete!"

clean-models:
	rm -rf models/*.joblib
	@echo "✓ Models removed!"

clean-all: clean clean-models
	rm -rf notebooks/figures/*
	rm -rf data/raw/*
	@echo "✓ Full cleanup complete!"
