"""
Comprehensive tests for CourseSearchTool.execute() method.

This test suite evaluates the CourseSearchTool to identify why it's
returning "query failed" responses for content-related questions.
"""

import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute(unittest.TestCase):
    """Test suite for CourseSearchTool.execute() method"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search_with_results(self):
        """Test execute with successful search returning results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=[
                "This is content about Python variables",
                "More content about data types",
            ],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson/1"
        )

        # Execute search
        result = self.search_tool.execute("python variables")

        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="python variables", course_name=None, lesson_number=None
        )

        # Verify result format contains expected content
        self.assertIn("Python Basics", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("This is content about Python variables", result)
        self.assertIn("More content about data types", result)

        # Verify sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(
            self.search_tool.last_sources[0]["display"], "Python Basics - Lesson 1"
        )

    def test_execute_search_with_error(self):
        """Test execute when vector store returns an error"""
        # Mock error result
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("test query")

        # Verify error is returned directly
        self.assertEqual(result, "ChromaDB connection failed")

    def test_execute_search_with_empty_results(self):
        """Test execute when no results are found - CRITICAL TEST for query failed issue"""
        # Mock empty results (this is what happens when MAX_RESULTS=0)
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("nonexistent topic")

        # Verify "no content found" message
        self.assertEqual(result, "No relevant content found.")

    def test_execute_search_with_course_filter(self):
        """Test execute with course name filter"""
        mock_results = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Execute with course filter
        result = self.search_tool.execute(
            "advanced topics", course_name="Advanced Python"
        )

        # Verify course filter was passed to vector store
        self.mock_vector_store.search.assert_called_once_with(
            query="advanced topics", course_name="Advanced Python", lesson_number=None
        )

        # Verify formatted result contains course info
        self.assertIn("Advanced Python", result)

    def test_execute_search_with_lesson_filter(self):
        """Test execute with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 5}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson/5"
        )

        # Execute with lesson filter
        result = self.search_tool.execute("lesson content", lesson_number=5)

        # Verify lesson filter was passed
        self.mock_vector_store.search.assert_called_once_with(
            query="lesson content", course_name=None, lesson_number=5
        )

        # Verify formatted result contains lesson info
        self.assertIn("Lesson 5", result)

    def test_execute_search_with_both_filters(self):
        """Test execute with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson/2"
        )

        # Execute with both filters
        result = self.search_tool.execute(
            "ML algorithms", course_name="Data Science", lesson_number=2
        )

        # Verify both filters were passed
        self.mock_vector_store.search.assert_called_once_with(
            query="ML algorithms", course_name="Data Science", lesson_number=2
        )

        # Verify formatted result
        self.assertIn("Data Science", result)
        self.assertIn("Lesson 2", result)

    def test_execute_empty_results_with_filters(self):
        """Test execute when no results found with filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute with filters
        result = self.search_tool.execute(
            "nonexistent", course_name="Python Basics", lesson_number=1
        )

        # Verify specific no results message includes filter info
        expected = "No relevant content found in course 'Python Basics' in lesson 1."
        self.assertEqual(result, expected)

    def test_format_results_with_lesson_links(self):
        """Test that results formatting includes lesson links when available"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Web Dev", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/web-dev/lesson-1"
        )

        # Execute search
        result = self.search_tool.execute("web development")

        # Verify lesson link was requested
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Web Dev", 1)

        # Verify source includes link
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(
            self.search_tool.last_sources[0]["link"],
            "https://example.com/web-dev/lesson-1",
        )

    def test_format_results_without_lesson_links(self):
        """Test results formatting when no lesson links available"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Web Dev", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Execute search
        result = self.search_tool.execute("web development")

        # Verify source has no link
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertIsNone(self.search_tool.last_sources[0]["link"])

    def test_format_results_without_lesson_number(self):
        """Test results formatting for content without lesson numbers"""
        mock_results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Machine Learning"}],  # No lesson_number
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("ML overview")

        # Verify formatting handles missing lesson number
        self.assertIn("Machine Learning", result)
        self.assertNotIn("Lesson", result)

        # Verify source display without lesson
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(
            self.search_tool.last_sources[0]["display"], "Machine Learning"
        )

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted for Anthropic"""
        definition = self.search_tool.get_tool_definition()

        # Verify required fields
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertEqual(schema["required"], ["query"])

        # Verify properties
        properties = schema["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)

        # Verify property types
        self.assertEqual(properties["query"]["type"], "string")
        self.assertEqual(properties["course_name"]["type"], "string")
        self.assertEqual(properties["lesson_number"]["type"], "integer")

    def test_sources_tracking_and_reset(self):
        """Test that sources are properly tracked and can be reset"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course 1", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
        ]

        # Execute search
        self.search_tool.execute("test query")

        # Verify sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(
            self.search_tool.last_sources[0]["display"], "Course 1 - Lesson 1"
        )
        self.assertEqual(
            self.search_tool.last_sources[1]["display"], "Course 2 - Lesson 2"
        )

        # Reset sources
        self.search_tool.last_sources = []
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_critical_max_results_zero_scenario(self):
        """CRITICAL TEST: Simulate the MAX_RESULTS=0 configuration issue"""
        # This simulates what happens when config.MAX_RESULTS = 0
        # Vector store should return empty results, causing "query failed"

        mock_results = SearchResults(
            documents=[],  # Empty because MAX_RESULTS=0
            metadata=[],
            distances=[],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search - this should reveal the issue
        result = self.search_tool.execute("python variables")

        # This is what causes "query failed" - empty results due to MAX_RESULTS=0
        self.assertEqual(result, "No relevant content found.")

        # Verify that vector store was called (so the issue isn't in calling)
        self.mock_vector_store.search.assert_called_once()


class TestToolManagerIntegration(unittest.TestCase):
    """Test ToolManager integration with CourseSearchTool"""

    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_register_tool(self):
        """Test tool registration in ToolManager"""
        # Register the search tool
        self.tool_manager.register_tool(self.search_tool)

        # Verify tool is registered correctly
        self.assertIn("search_course_content", self.tool_manager.tools)
        self.assertEqual(
            self.tool_manager.tools["search_course_content"], self.search_tool
        )

    def test_get_tool_definitions(self):
        """Test getting tool definitions for AI"""
        # Register tool and get definitions
        self.tool_manager.register_tool(self.search_tool)
        definitions = self.tool_manager.get_tool_definitions()

        # Verify format
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "search_course_content")
        self.assertIn("description", definitions[0])
        self.assertIn("input_schema", definitions[0])

    def test_execute_tool_success(self):
        """Test successful tool execution through manager"""
        # Mock successful search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Register tool
        self.tool_manager.register_tool(self.search_tool)

        # Execute through manager
        result = self.tool_manager.execute_tool(
            "search_course_content", query="test query", course_name="Test Course"
        )

        # Verify execution
        self.assertIn("Test Course", result)
        self.mock_vector_store.search.assert_called_once()

    def test_execute_tool_with_error(self):
        """Test tool execution when search tool returns error"""
        # Mock error result
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Vector store connection failed",
        )
        self.mock_vector_store.search.return_value = mock_results

        # Register tool
        self.tool_manager.register_tool(self.search_tool)

        # Execute through manager
        result = self.tool_manager.execute_tool(
            "search_course_content", query="test query"
        )

        # Verify error is propagated
        self.assertEqual(result, "Vector store connection failed")

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        result = self.tool_manager.execute_tool("nonexistent_tool", query="test")
        self.assertEqual(result, "Tool 'nonexistent_tool' not found")

    def test_get_last_sources(self):
        """Test retrieving sources from tools"""
        # Register tool with mock sources
        self.tool_manager.register_tool(self.search_tool)
        self.search_tool.last_sources = [
            {
                "display": "Test Course - Lesson 1",
                "link": "https://example.com/lesson1",
            },
            {"display": "Test Course - Lesson 2", "link": None},
        ]

        # Get sources
        sources = self.tool_manager.get_last_sources()

        # Verify sources
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["display"], "Test Course - Lesson 1")
        self.assertEqual(sources[1]["display"], "Test Course - Lesson 2")

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        # Register tool with mock sources
        self.tool_manager.register_tool(self.search_tool)
        self.search_tool.last_sources = [{"display": "Test Course", "link": None}]

        # Reset sources
        self.tool_manager.reset_sources()

        # Verify sources are reset
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_multiple_tools_source_management(self):
        """Test source management with multiple tools"""
        # Create second mock tool
        mock_tool2 = Mock()
        mock_tool2.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool2.last_sources = [{"display": "Tool 2 Source", "link": None}]

        # Register both tools
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(mock_tool2)

        # Set sources on search tool
        self.search_tool.last_sources = [
            {"display": "Search Tool Source", "link": None}
        ]

        # Get sources (should return from tool with sources)
        sources = self.tool_manager.get_last_sources()

        # Should return sources from first tool that has them
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["display"], "Search Tool Source")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
