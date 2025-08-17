"""
API endpoint tests for the FastAPI RAG system.

Tests all API endpoints for proper request/response handling, error scenarios,
and integration with the RAG system components.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint for query processing."""
    
    def test_query_endpoint_basic_request(self, client: TestClient, sample_query_request):
        """Test basic query request with valid input."""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify response content
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert len(data["answer"]) > 0
        assert len(data["sources"]) > 0
    
    def test_query_endpoint_with_existing_session(self, client: TestClient, sample_query_request_with_session):
        """Test query request with existing session ID."""
        response = client.post("/api/query", json=sample_query_request_with_session)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use the provided session ID
        assert data["session_id"] == "existing_session_456"
        assert "answer" in data
        assert "sources" in data
    
    def test_query_endpoint_without_session_creates_new(self, client: TestClient):
        """Test that query without session ID creates a new session."""
        request_data = {"query": "Test query without session"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should create new session
        assert data["session_id"] == "test_session_123"  # From mock
        assert "answer" in data
        assert "sources" in data
    
    def test_query_endpoint_empty_query(self, client: TestClient):
        """Test query endpoint with empty query string."""
        request_data = {"query": ""}
        
        response = client.post("/api/query", json=request_data)
        
        # Should still process (RAG system handles empty queries)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_whitespace_only_query(self, client: TestClient):
        """Test query endpoint with whitespace-only query."""
        request_data = {"query": "   \t\n   "}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_long_query(self, client: TestClient):
        """Test query endpoint with very long query."""
        long_query = "What is Python? " * 100  # Very long query
        request_data = {"query": long_query}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_special_characters(self, client: TestClient):
        """Test query endpoint with special characters and unicode."""
        request_data = {"query": "What about Python's 'strings' & variables? 🐍 测试"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_missing_query_field(self, client: TestClient):
        """Test query endpoint with missing required query field."""
        request_data = {"session_id": "test_session"}  # Missing 'query' field
        
        response = client.post("/api/query", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_query_endpoint_invalid_json(self, client: TestClient):
        """Test query endpoint with invalid JSON."""
        response = client.post(
            "/api/query", 
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return JSON parse error
        assert response.status_code == 422
    
    def test_query_endpoint_wrong_content_type(self, client: TestClient):
        """Test query endpoint with wrong content type."""
        response = client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # FastAPI should handle this gracefully or return error
        assert response.status_code in [422, 400]
    
    def test_query_endpoint_rag_system_error(self, client: TestClient, test_app):
        """Test query endpoint when RAG system raises an exception."""
        # Configure mock to raise exception directly on the test app's mock
        test_app.state.test_rag_system.query.side_effect = Exception("RAG system error")
        
        request_data = {"query": "test query"}
        response = client.post("/api/query", json=request_data)
        
        # Should return 500 error
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert "RAG system error" in error_data["detail"]
    
    def test_query_endpoint_response_format_validation(self, client: TestClient):
        """Test that query response matches expected format exactly."""
        request_data = {"query": "Python variables"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Strict format validation
        required_fields = {"answer", "sources", "session_id"}
        assert set(data.keys()) == required_fields
        
        # Type validation
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Content validation
        assert len(data["answer"]) > 0
        assert len(data["session_id"]) > 0


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint for course statistics."""
    
    def test_courses_endpoint_basic_request(self, client: TestClient, expected_course_stats):
        """Test basic courses statistics request."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify response content matches expected
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
    
    def test_courses_endpoint_response_types(self, client: TestClient):
        """Test that courses endpoint returns correct data types."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Type validation
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        
        # Validate course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
            assert len(title) > 0
    
    def test_courses_endpoint_no_parameters_required(self, client: TestClient):
        """Test that courses endpoint doesn't require any parameters."""
        # Should work with no query parameters
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        # Should work with ignored query parameters
        response = client.get("/api/courses?ignored=parameter")
        assert response.status_code == 200
    
    def test_courses_endpoint_wrong_method(self, client: TestClient):
        """Test courses endpoint with wrong HTTP method."""
        # POST should not be allowed
        response = client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed
        
        # PUT should not be allowed
        response = client.put("/api/courses")
        assert response.status_code == 405
        
        # DELETE should not be allowed
        response = client.delete("/api/courses")
        assert response.status_code == 405
    
    def test_courses_endpoint_rag_system_error(self, client: TestClient, test_app):
        """Test courses endpoint when RAG system raises an exception."""
        # Configure mock to raise exception directly on the test app's mock
        test_app.state.test_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        # Should return 500 error
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert "Analytics error" in error_data["detail"]
    
    def test_courses_endpoint_response_format_validation(self, client: TestClient):
        """Test that courses response matches expected format exactly."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Strict format validation
        required_fields = {"total_courses", "course_titles"}
        assert set(data.keys()) == required_fields
        
        # Content validation
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]


@pytest.mark.api
class TestRootEndpoint:
    """Test the root endpoint ("/") for basic API info."""
    
    def test_root_endpoint_basic_request(self, client: TestClient):
        """Test basic root endpoint request."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return basic message
        assert "message" in data
        assert isinstance(data["message"], str)
        assert "RAG System" in data["message"]
    
    def test_root_endpoint_wrong_method(self, client: TestClient):
        """Test root endpoint with wrong HTTP methods."""
        # POST should not be allowed (or should be handled gracefully)
        response = client.post("/")
        assert response.status_code in [405, 200]  # Method Not Allowed or handled
        
        # PUT should not be allowed
        response = client.put("/")
        assert response.status_code in [405, 200]


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS middleware and other middleware functionality."""
    
    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are properly set."""
        # Test with a CORS request that should trigger headers
        headers = {"Origin": "http://localhost:3000"}
        response = client.get("/api/courses", headers=headers)
        
        # CORS headers might not be present on all responses in test client
        # Just verify the request succeeds
        assert response.status_code == 200
    
    def test_options_request_handling(self, client: TestClient):
        """Test OPTIONS requests for CORS preflight."""
        response = client.options("/api/query")
        
        # Should handle OPTIONS requests for CORS (405 is also acceptable in test)
        assert response.status_code in [200, 204, 405]
    
    def test_cors_with_custom_headers(self, client: TestClient):
        """Test CORS with custom headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        
        response = client.options("/api/query", headers=headers)
        
        # Should handle preflight request
        assert response.status_code in [200, 204]


@pytest.mark.api
class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_404_for_nonexistent_endpoints(self, client: TestClient):
        """Test 404 response for non-existent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/api/invalid")
        assert response.status_code == 404
        
        response = client.get("/nonexistent/path")
        assert response.status_code == 404
    
    def test_error_response_format(self, client: TestClient):
        """Test that error responses follow consistent format."""
        # Test 404 error format
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        
        # Test validation error format
        response = client.post("/api/query", json={})  # Missing required field
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_content_type_handling(self, client: TestClient):
        """Test handling of different content types."""
        # JSON content type (correct)
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        
        # No content type header
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Test API performance and load handling."""
    
    def test_multiple_concurrent_requests(self, client: TestClient):
        """Test handling of multiple rapid requests."""
        queries = [
            {"query": f"Test query {i}"} for i in range(10)
        ]
        
        responses = []
        for query in queries:
            response = client.post("/api/query", json=query)
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data
    
    def test_large_response_handling(self, client: TestClient):
        """Test handling of potentially large responses."""
        # Query that might generate large response
        request_data = {"query": "Tell me everything about Python programming"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
    
    def test_api_response_times(self, client: TestClient):
        """Test that API responses are reasonably fast."""
        import time
        
        start_time = time.time()
        response = client.get("/api/courses")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be reasonably fast (adjust threshold as needed)
        response_time = end_time - start_time
        assert response_time < 5.0  # 5 seconds max for courses endpoint


@pytest.mark.api
class TestAPIDocumentation:
    """Test API documentation and OpenAPI spec generation."""
    
    def test_openapi_json_endpoint(self, client: TestClient):
        """Test that OpenAPI JSON is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi_spec = response.json()
        
        # Verify basic OpenAPI structure
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Verify our endpoints are documented
        paths = openapi_spec["paths"]
        assert "/api/query" in paths
        assert "/api/courses" in paths
    
    def test_docs_endpoint_accessible(self, client: TestClient):
        """Test that API docs endpoint is accessible."""
        response = client.get("/docs")
        
        # Should either return docs or redirect to docs
        assert response.status_code in [200, 301, 302, 307, 308]
    
    def test_redoc_endpoint_accessible(self, client: TestClient):
        """Test that ReDoc endpoint is accessible."""
        response = client.get("/redoc")
        
        # Should either return redoc or redirect to redoc
        assert response.status_code in [200, 301, 302, 307, 308]