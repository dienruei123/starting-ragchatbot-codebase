#!/bin/bash
# Run all quality checks: formatting, linting, and type checking

echo "🚀 Running all quality checks..."

echo "1️⃣ Formatting code..."
./scripts/format.sh

echo ""
echo "2️⃣ Running linting and type checking..."
./scripts/lint.sh

echo ""
echo "3️⃣ Running tests..."
cd backend && uv run python -m pytest tests/ -v

echo ""
echo "🎉 All quality checks complete!"