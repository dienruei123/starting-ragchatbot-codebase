# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system built with Python FastAPI backend and vanilla HTML/CSS/JS frontend. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides intelligent answers about course materials through semantic search.

## Development Commands

### Quick Start

```bash
./run.sh
```

### Manual Development

```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev

# Start development server
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality

```bash
# Format code (black + isort)
./scripts/format.sh

# Run linting and type checking
./scripts/lint.sh

# Run all quality checks (format + lint + tests)
./scripts/quality.sh

# Individual commands
uv run black backend/ main.py          # Format code
uv run isort backend/ main.py           # Sort imports
uv run flake8 backend/ main.py          # Lint code
uv run mypy backend/ main.py            # Type checking
```

### Environment Setup

Create `.env` file in root directory:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture

### Backend Structure (`/backend/`)

- **`app.py`**: FastAPI application with CORS middleware, serves API endpoints and static frontend
- **`rag_system.py`**: Main orchestrator coordinating all RAG components
- **`vector_store.py`**: ChromaDB integration for semantic search and document storage
- **`document_processor.py`**: Processes course documents into searchable chunks
- **`ai_generator.py`**: Anthropic Claude integration with tool-calling capabilities
- **`search_tools.py`**: Search tools for the AI to use during query processing
- **`session_manager.py`**: Manages conversation history and user sessions
- **`models.py`**: Pydantic models for Course, Lesson, and CourseChunk data structures
- **`config.py`**: Configuration management with environment variable loading

### Key Configuration (config.py)

- Chunk size: 800 characters with 100 character overlap
- Max search results: 5
- Conversation history: 2 messages
- ChromaDB path: `./chroma_db`
- Claude model: `claude-sonnet-4-20250514`

### Frontend Structure (`/frontend/`)

- **`index.html`**: Main web interface
- **`script.js`**: Frontend JavaScript for API communication
- **`style.css`**: Application styling

### Data Flow

1. Course documents in `/docs/` are processed into chunks on startup
2. Documents are embedded using sentence-transformers and stored in ChromaDB
3. User queries trigger semantic search via search tools
4. AI generates responses using retrieved context and conversation history
5. Frontend displays answers with source attribution

## API Endpoints

- `POST /api/query`: Submit questions and receive AI-generated answers
- `GET /api/courses`: Get course statistics and available titles
- `GET /docs`: Auto-generated API documentation

## Application Access

- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- always use uv to run server, do not use pip directly
- make sure to use uv to manage all dependencies

- use uv to run Python files