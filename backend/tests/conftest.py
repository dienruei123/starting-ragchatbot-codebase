"""
Shared test fixtures and configuration for the RAG system test suite.

This module provides common fixtures, mocks, and utilities used across
multiple test files to ensure consistent testing setup and teardown.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rag_system import RAGSystem


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir: str) -> Config:
    """Create a test configuration with temporary paths."""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma_db")
    config.ANTHROPIC_API_KEY = "test_key"
    config.MAX_RESULTS = 5  # Ensure valid configuration
    return config


@pytest.fixture
def test_docs_dir(temp_dir: str) -> str:
    """Create test course documents for loading."""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Python course content
    python_content = """Course Title: Python Programming Fundamentals
Course Link: https://example.com/python-fundamentals
Course Instructor: Alice Johnson

Lesson 1: Introduction to Python
Lesson Link: https://example.com/python-fundamentals/lesson-1
Python is a high-level, interpreted programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991.

Lesson 2: Python Variables and Data Types
Lesson Link: https://example.com/python-fundamentals/lesson-2
In Python, variables are used to store data values. Python has several built-in data types including integers, floating-point numbers, strings, and booleans.

Lesson 3: Control Flow and Functions
Lesson Link: https://example.com/python-fundamentals/lesson-3
Python uses control flow statements like if, elif, and else for conditional execution. Functions in Python are defined using the def keyword.
"""
    
    # Machine Learning course content
    ml_content = """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-intro
Course Instructor: Bob Smith

Lesson 1: What is Machine Learning
Lesson Link: https://example.com/ml-intro/lesson-1
Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance.

Lesson 2: Supervised Learning Algorithms
Lesson Link: https://example.com/ml-intro/lesson-2
Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common algorithms include linear regression and decision trees.
"""
    
    with open(os.path.join(docs_dir, "python_course.txt"), "w") as f:
        f.write(python_content)
        
    with open(os.path.join(docs_dir, "ml_course.txt"), "w") as f:
        f.write(ml_content)
    
    return docs_dir


@pytest.fixture
def rag_system(test_config: Config) -> RAGSystem:
    """Create a RAG system instance with test configuration."""
    return RAGSystem(test_config)


@pytest.fixture
def rag_system_with_data(test_config: Config, test_docs_dir: str) -> RAGSystem:
    """Create a RAG system instance with test data pre-loaded."""
    system = RAGSystem(test_config)
    system.add_course_folder(test_docs_dir, clear_existing=True)
    return system


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing AI interactions."""
    mock_client = Mock()
    
    # Default mock response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test AI response"
    mock_response.content[0].type = "text"
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    return mock_client


class MockAnthropicContentBlock:
    """Mock content block for simulating Anthropic API responses."""
    
    def __init__(self, block_type: str, text: str = None, name: str = None, 
                 input_data: Dict[str, Any] = None, block_id: str = None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input_data or {}
        self.id = block_id or "mock_block"


class MockAnthropicResponse:
    """Mock response object for simulating Anthropic API responses."""
    
    def __init__(self, content, stop_reason: str = "end_turn"):
        if isinstance(content, str):
            # Simple text response
            self.content = [MockAnthropicContentBlock("text", text=content)]
        elif isinstance(content, list):
            # List of content blocks
            self.content = content
        else:
            # Single content block
            self.content = [content]
        
        self.stop_reason = stop_reason
        self.usage = Mock()
        self.usage.input_tokens = 100
        self.usage.output_tokens = 50


@pytest.fixture
def mock_anthropic_with_tool_calling(mock_anthropic_client):
    """Configure mock Anthropic client for tool calling scenarios."""
    
    def create_tool_response(query: str, course_name: str = None):
        """Create a realistic tool calling response."""
        # First response: tool use
        tool_use = MockAnthropicContentBlock(
            "tool_use",
            name="search_course_content",
            input_data={"query": query, "course_name": course_name},
            block_id="test_tool_call"
        )
        
        # Second response: final answer
        final_answer = f"Based on the search results, here's information about {query}."
        
        return [
            MockAnthropicResponse([tool_use], stop_reason="tool_use"),
            MockAnthropicResponse(final_answer)
        ]
    
    mock_anthropic_client.create_tool_response = create_tool_response
    return mock_anthropic_client


@pytest.fixture
def test_app():
    """Create a FastAPI test application without static file mounting issues."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    
    # Create a minimal test app with just the API endpoints
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    test_rag_system = Mock()
    test_rag_system.query.return_value = (
        "Test answer about Python programming concepts.", 
        ["Test source 1", "Test source 2"]
    )
    test_rag_system.session_manager.create_session.return_value = "test_session_123"
    test_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Programming Fundamentals", "Introduction to Machine Learning"]
    }
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = test_rag_system.session_manager.create_session()
            
            answer, sources = test_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = test_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    # Store reference to mock for test access
    app.state.test_rag_system = test_rag_system
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return {
        "query": "What are Python variables?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with session ID for testing."""
    return {
        "query": "Explain machine learning algorithms",
        "session_id": "existing_session_456"
    }


@pytest.fixture
def expected_course_stats():
    """Expected course statistics response for testing."""
    return {
        "total_courses": 2,
        "course_titles": ["Python Programming Fundamentals", "Introduction to Machine Learning"]
    }


# Test markers for categorizing tests
pytest_markers = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions", 
    "api": "API endpoint tests",
    "slow": "Tests that take longer to run"
}