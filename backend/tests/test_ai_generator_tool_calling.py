"""
Comprehensive tests for AI Generator tool calling functionality.

This test suite evaluates whether the AI Generator correctly calls
CourseSearchTool and handles the tool execution flow properly.
"""

import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


class MockAnthropicContentBlock:
    """Mock content block for Anthropic API responses"""

    def __init__(
        self, block_type, text=None, name=None, input_data=None, block_id=None
    ):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input_data or {}
        self.id = block_id or "mock_tool_id"


class MockAnthropicResponse:
    """Mock Anthropic API response"""

    def __init__(self, content, stop_reason="end_turn"):
        if isinstance(content, list):
            self.content = content
        elif isinstance(content, str):
            self.content = [MockAnthropicContentBlock("text", content)]
        else:
            self.content = [content]
        self.stop_reason = stop_reason


class TestAIGeneratorBasicFunctionality(unittest.TestCase):
    """Test basic AI generator functionality without tools"""

    def setUp(self):
        """Set up test fixtures"""
        self.ai_generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockAnthropicResponse("This is a direct response without tools")
        mock_client.messages.create.return_value = mock_response

        # Create new AI generator to use mocked client
        ai_gen = AIGenerator("test_key", "test_model")

        # Generate response without tools
        result = ai_gen.generate_response("What is machine learning?")

        # Verify response
        self.assertEqual(result, "This is a direct response without tools")

        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(
            call_args["messages"][0]["content"], "What is machine learning?"
        )
        self.assertNotIn("tools", call_args)

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tools_but_no_tool_use(self, mock_anthropic_class):
        """Test response when tools are available but AI doesn't use them"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # AI responds directly without using tools
        mock_response = MockAnthropicResponse("General knowledge answer without search")
        mock_client.messages.create.return_value = mock_response

        # Create AI generator and tool manager
        ai_gen = AIGenerator("test_key", "test_model")
        mock_tool_manager = Mock()
        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        # Generate response with tools available
        result = ai_gen.generate_response(
            query="What is Python?",  # General question that might not need search
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify response
        self.assertEqual(result, "General knowledge answer without search")

        # Verify tools were provided to API
        call_args = mock_client.messages.create.call_args[1]
        self.assertIn("tools", call_args)
        self.assertEqual(call_args["tools"], mock_tools)


class TestAIGeneratorToolCalling(unittest.TestCase):
    """Test AI generator tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_tool_manager = Mock()
        self.mock_vector_store = Mock()

        # Set up real tool manager with search tool for realistic tool definitions
        self.real_tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.real_tool_manager.register_tool(self.search_tool)
        self.tool_definitions = self.real_tool_manager.get_tool_definitions()

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation that triggers tool use"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: AI decides to use search tool
        tool_use_content = [
            MockAnthropicContentBlock(
                "text", "I need to search for information about this topic."
            ),
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={
                    "query": "Python variables",
                    "course_name": "Python Basics",
                },
                block_id="tool_123",
            ),
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")

        # Second response: AI provides final answer after tool use
        final_response = MockAnthropicResponse(
            "Python variables are used to store data values. They are containers that hold different types of data like strings, numbers, and booleans."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock tool execution result
        self.mock_tool_manager.execute_tool.return_value = "[Python Basics - Lesson 2]\nPython variables are containers for storing data values..."

        # Create new AI generator to use mocked client
        ai_gen = AIGenerator("test_key", "test_model")

        # Generate response with tools
        result = ai_gen.generate_response(
            query="How do Python variables work?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
        )

        # Verify final response
        self.assertIn("Python variables are used to store data", result)

        # Verify tool was executed with correct parameters
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python variables",
            course_name="Python Basics",
        )

        # Verify two API calls were made (initial + follow-up)
        self.assertEqual(mock_client.messages.create.call_count, 2)

    @patch("anthropic.Anthropic")
    def test_tool_execution_flow_messages(self, mock_anthropic_class):
        """Test the complete tool execution message flow"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "machine learning basics"},
                block_id="tool_456",
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")

        # Final response
        final_response = MockAnthropicResponse(
            "Based on the course materials, machine learning is..."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock tool result
        tool_result = "[ML Course - Lesson 1]\nMachine learning is a subset of AI that enables computers to learn..."
        self.mock_tool_manager.execute_tool.return_value = tool_result

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute
        result = ai_gen.generate_response(
            query="What is machine learning?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
        )

        # Verify tool execution
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning basics"
        )

        # Verify second API call includes proper message structure
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]

        # Should have: user query, assistant tool use, user tool result
        self.assertEqual(len(messages), 3)

        # Check message roles
        self.assertEqual(messages[0]["role"], "user")  # Original query
        self.assertEqual(messages[1]["role"], "assistant")  # Tool use
        self.assertEqual(messages[2]["role"], "user")  # Tool result

        # Verify tool result content structure
        tool_result_content = messages[2]["content"]
        self.assertEqual(len(tool_result_content), 1)
        self.assertEqual(tool_result_content[0]["type"], "tool_result")
        self.assertEqual(tool_result_content[0]["tool_use_id"], "tool_456")
        self.assertEqual(tool_result_content[0]["content"], tool_result)

    @patch("anthropic.Anthropic")
    def test_multiple_tool_calls_in_response(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Response with multiple tool calls
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "Python basics", "course_name": "Python 101"},
                block_id="tool_1",
            ),
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={
                    "query": "Python advanced",
                    "course_name": "Advanced Python",
                },
                block_id="tool_2",
            ),
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse(
            "Combining information from both Python courses..."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock tool execution results
        self.mock_tool_manager.execute_tool.side_effect = [
            "[Python 101 - Lesson 1]\nBasic Python concepts...",
            "[Advanced Python - Lesson 1]\nAdvanced Python techniques...",
        ]

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute
        result = ai_gen.generate_response(
            query="Compare basic and advanced Python concepts",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
        )

        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)

        # Verify call arguments
        calls = self.mock_tool_manager.execute_tool.call_args_list

        # First call
        self.assertEqual(calls[0][0][0], "search_course_content")
        self.assertEqual(calls[0][1]["query"], "Python basics")
        self.assertEqual(calls[0][1]["course_name"], "Python 101")

        # Second call
        self.assertEqual(calls[1][0][0], "search_course_content")
        self.assertEqual(calls[1][1]["query"], "Python advanced")
        self.assertEqual(calls[1][1]["course_name"], "Advanced Python")

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test query"},
                block_id="tool_error",
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse(
            "I apologize, but I encountered an error while searching..."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock tool error - simulating the MAX_RESULTS=0 issue
        self.mock_tool_manager.execute_tool.return_value = "No relevant content found."

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute
        result = ai_gen.generate_response(
            query="Find information about a topic",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
        )

        # Verify tool error was passed to AI in follow-up call
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result_content = second_call_args["messages"][2]["content"][0]
        self.assertEqual(tool_result_content["content"], "No relevant content found.")

        # This is what could lead to "query failed" responses
        self.assertIn("error", result.lower())

    @patch("anthropic.Anthropic")
    def test_conversation_history_with_tools(self, mock_anthropic_class):
        """Test tool calling with conversation history context"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "Python functions"},
                block_id="tool_history",
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse(
            "Building on our previous discussion about variables..."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]
        self.mock_tool_manager.execute_tool.return_value = (
            "[Python Basics - Lesson 3]\nFunctions in Python..."
        )

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute with conversation history
        history = "User: What are Python variables?\nAssistant: Python variables are containers for storing data."
        result = ai_gen.generate_response(
            query="How do Python functions work?",
            conversation_history=history,
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager,
        )

        # Verify history is included in system prompt
        first_call_args = mock_client.messages.create.call_args_list[0][1]
        system_content = first_call_args["system"]
        self.assertIn("Previous conversation", system_content)
        self.assertIn("Python variables are containers", system_content)

    @patch("anthropic.Anthropic")
    def test_no_tool_manager_graceful_handling(self, mock_anthropic_class):
        """Test behavior when tool_manager is None but tools are provided"""
        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response (shouldn't happen with no tool manager, but test graceful handling)
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test"},
                block_id="tool_no_manager",
            )
        ]
        response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        mock_client.messages.create.return_value = response

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute with tools but no tool manager - should not crash
        try:
            result = ai_gen.generate_response(
                query="Test query",
                tools=self.tool_definitions,
                tool_manager=None,  # No tool manager provided
            )
            # Should handle gracefully - exact behavior depends on implementation
            # Main thing is it shouldn't crash
        except Exception as e:
            self.fail(f"Should handle missing tool_manager gracefully, but got: {e}")


class TestAIGeneratorRealToolIntegration(unittest.TestCase):
    """Integration tests with real tool components"""

    def setUp(self):
        """Set up with real tool manager and search tool"""
        from vector_store import SearchResults

        self.mock_vector_store = Mock()
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(self.search_tool)

    @patch("anthropic.Anthropic")
    def test_real_tool_execution_flow(self, mock_anthropic_class):
        """Test with real tool manager and mocked vector store"""
        from vector_store import SearchResults

        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={
                    "query": "database design principles",
                    "course_name": "Database Systems",
                },
                block_id="real_tool",
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse(
            "Database design follows several key principles..."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock vector store search result
        mock_search_result = SearchResults(
            documents=[
                "Database normalization reduces redundancy and improves data integrity."
            ],
            metadata=[{"course_title": "Database Systems", "lesson_number": 3}],
            distances=[0.15],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_search_result
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/db-lesson3"
        )

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute with real tool manager
        result = ai_gen.generate_response(
            query="What are the principles of database design?",
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
        )

        # Verify vector store was called with correct parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="database design principles",
            course_name="Database Systems",
            lesson_number=None,
        )

        # Verify sources were tracked in the tool
        sources = self.tool_manager.get_last_sources()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["display"], "Database Systems - Lesson 3")
        self.assertEqual(sources[0]["link"], "https://example.com/db-lesson3")

    @patch("anthropic.Anthropic")
    def test_real_tool_with_empty_results_scenario(self, mock_anthropic_class):
        """Test real tool behavior with empty results (MAX_RESULTS=0 scenario)"""
        from vector_store import SearchResults

        # Mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "machine learning algorithms"},
                block_id="empty_results",
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse(
            "I couldn't find relevant information about that topic."
        )

        mock_client.messages.create.side_effect = [first_response, final_response]

        # Mock empty search result (simulating MAX_RESULTS=0 issue)
        mock_search_result = SearchResults(
            documents=[],  # Empty due to MAX_RESULTS=0
            metadata=[],
            distances=[],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_search_result

        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")

        # Execute
        result = ai_gen.generate_response(
            query="Explain machine learning algorithms",
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
        )

        # Verify search was attempted
        self.mock_vector_store.search.assert_called_once()

        # Get the tool result that was passed to AI
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result_content = second_call_args["messages"][2]["content"][0]["content"]

        # This should be the "No relevant content found." message from CourseSearchTool
        self.assertEqual(tool_result_content, "No relevant content found.")

        # This is what leads to "query failed" type responses
        self.assertIn("couldn't find", result.lower())


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
