#!/bin/bash
# Run linting and type checking

echo "🔍 Running flake8 linting..."
uv run flake8 backend/ main.py

echo "🔬 Running mypy type checking..."
uv run mypy backend/ main.py

echo "✅ Linting and type checking complete!"