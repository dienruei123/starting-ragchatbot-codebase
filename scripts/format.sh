#!/bin/bash
# Format code with black and sort imports with isort

echo "🎨 Formatting code with black..."
uv run black backend/ main.py

echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"