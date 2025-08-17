"""
Tests for AI Generator sequential tool calling functionality.

This test suite verifies the new sequential tool calling behavior where Claude
can make up to 2 tool calls in separate API rounds for complex queries.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class MockAnthropicContentBlock:
    """Mock content block for Anthropic API responses"""
    def __init__(self, block_type, text=None, name=None, input_data=None, block_id=None):
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


class TestSequentialToolCalling(unittest.TestCase):
    """Test sequential tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tool_manager = Mock()
        self.mock_vector_store = Mock()
        
        # Set up real tool manager with search tool for realistic tool definitions
        self.real_tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.real_tool_manager.register_tool(self.search_tool)
        self.tool_definitions = self.real_tool_manager.get_tool_definitions()
        
    @patch('anthropic.Anthropic')
    def test_single_tool_call_backward_compatibility(self, mock_anthropic_class):
        """Test that single tool call behavior still works (backward compatibility)"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First response: AI uses one tool
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use", 
                name="search_course_content", 
                input_data={"query": "Python variables"}, 
                block_id="tool_1"
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        
        # Second response: AI provides final answer (no more tools)
        final_response = MockAnthropicResponse("Python variables store data values.")
        
        mock_client.messages.create.side_effect = [first_response, final_response]
        self.mock_tool_manager.execute_tool.return_value = "Variable info from course"
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="What are Python variables?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify single tool execution
        self.mock_tool_manager.execute_tool.assert_called_once()
        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertIn("Python variables store data", result)
        
    @patch('anthropic.Anthropic')
    def test_sequential_two_round_tool_calling(self, mock_anthropic_class):
        """Test sequential tool calling across 2 rounds"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: AI uses first tool
        round1_tool_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="get_course_outline", 
                input_data={"course_name": "Python Basics"}, 
                block_id="tool_round1"
            )
        ]
        round1_response = MockAnthropicResponse(round1_tool_content, stop_reason="tool_use")
        
        # Round 2: AI uses second tool based on first results
        round2_tool_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "lesson 4 content", "course_name": "Python Basics"}, 
                block_id="tool_round2"
            )
        ]
        round2_response = MockAnthropicResponse(round2_tool_content, stop_reason="tool_use")
        
        # Final response: AI provides comprehensive answer
        final_response = MockAnthropicResponse("Based on the course outline and lesson content...")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock tool execution results
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1: Variables, Lesson 2: Functions, Lesson 3: Classes, Lesson 4: Advanced Topics",
            "Lesson 4 covers advanced Python features like decorators and generators"
        ]
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="What does lesson 4 of Python Basics cover?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify sequential tool execution (2 rounds)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(mock_client.messages.create.call_count, 3)
        
        # Verify tool execution order and parameters
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0][0], "get_course_outline")
        self.assertEqual(calls[1][0][0], "search_course_content")
        
        self.assertIn("Based on the course outline", result)
        
    @patch('anthropic.Anthropic')
    def test_early_termination_no_tools_round1(self, mock_anthropic_class):
        """Test early termination when AI doesn't use tools in round 1"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # AI responds directly without using tools
        direct_response = MockAnthropicResponse("This is general knowledge that doesn't require search.")
        mock_client.messages.create.return_value = direct_response
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify no tool execution and single API call
        self.mock_tool_manager.execute_tool.assert_not_called()
        self.assertEqual(mock_client.messages.create.call_count, 1)
        self.assertIn("general knowledge", result)
        
    @patch('anthropic.Anthropic')
    def test_termination_after_max_rounds(self, mock_anthropic_class):
        """Test termination after reaching maximum 2 rounds"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: AI uses tool
        round1_tool_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content", 
                input_data={"query": "machine learning"}, 
                block_id="tool_r1"
            )
        ]
        round1_response = MockAnthropicResponse(round1_tool_content, stop_reason="tool_use")
        
        # Round 2: AI uses another tool  
        round2_tool_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="get_course_outline",
                input_data={"course_name": "ML Course"}, 
                block_id="tool_r2"
            )
        ]
        round2_response = MockAnthropicResponse(round2_tool_content, stop_reason="tool_use")
        
        # Final response: No tools should be available (max rounds reached)
        final_response = MockAnthropicResponse("Final comprehensive answer without more tools.")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        self.mock_tool_manager.execute_tool.side_effect = ["ML content", "Course outline"]
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="Explain machine learning concepts",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify 2 tool executions and 3 API calls (2 rounds + final)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(mock_client.messages.create.call_count, 3)
        
        # Verify final API call has no tools parameter
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", final_call_args)
        
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test graceful handling of tool execution errors"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Tool use response
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test query"}, 
                block_id="tool_error"
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse("I encountered an error but can still help...")
        
        mock_client.messages.create.side_effect = [first_response, final_response]
        
        # Mock tool execution error
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="Find information",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify error was handled gracefully
        self.mock_tool_manager.execute_tool.assert_called_once()
        
        # Check that error message was passed to AI
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        tool_result_content = messages[2]["content"][0]
        self.assertIn("Tool execution failed", tool_result_content["content"])
        self.assertIn("Database connection failed", tool_result_content["content"])
        
    @patch('anthropic.Anthropic')
    def test_api_call_error_handling(self, mock_anthropic_class):
        """Test handling of API call errors during tool execution"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First call succeeds, second call fails
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test"}, 
                block_id="api_error"
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        
        mock_client.messages.create.side_effect = [
            first_response,
            Exception("API rate limit exceeded")
        ]
        self.mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="Test query",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify error handling
        self.assertIn("Error in round 1", result)
        self.assertIn("API rate limit exceeded", result)
        
    @patch('anthropic.Anthropic')
    def test_conversation_context_preservation(self, mock_anthropic_class):
        """Test that conversation context is preserved across rounds"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1 and 2 responses
        round1_response = MockAnthropicResponse([
            MockAnthropicContentBlock("tool_use", name="search_course_content", 
                                    input_data={"query": "functions"}, block_id="r1")
        ], stop_reason="tool_use")
        
        round2_response = MockAnthropicResponse([
            MockAnthropicContentBlock("tool_use", name="get_course_outline", 
                                    input_data={"course_name": "Python"}, block_id="r2")
        ], stop_reason="tool_use")
        
        final_response = MockAnthropicResponse("Complete answer with context")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        self.mock_tool_manager.execute_tool.side_effect = ["Function content", "Course outline"]
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute with conversation history
        history = "User: What are variables?\nAssistant: Variables store data."
        result = ai_gen.generate_response(
            query="Now explain functions",
            conversation_history=history,
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify conversation history is included in all API calls
        for call_args in mock_client.messages.create.call_args_list:
            system_content = call_args[1]["system"]
            self.assertIn("Previous conversation", system_content)
            self.assertIn("Variables store data", system_content)
            
    @patch('anthropic.Anthropic')
    def test_multiple_tools_in_single_round(self, mock_anthropic_class):
        """Test handling multiple tools within a single round"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: AI uses multiple tools
        multi_tool_content = [
            MockAnthropicContentBlock("tool_use", name="search_course_content",
                                    input_data={"query": "Python basics"}, block_id="t1"),
            MockAnthropicContentBlock("tool_use", name="get_course_outline",
                                    input_data={"course_name": "Python"}, block_id="t2")
        ]
        round1_response = MockAnthropicResponse(multi_tool_content, stop_reason="tool_use")
        
        # Round 2: Single tool call
        round2_response = MockAnthropicResponse([
            MockAnthropicContentBlock("tool_use", name="search_course_content",
                                    input_data={"query": "advanced topics"}, block_id="t3")
        ], stop_reason="tool_use")
        
        final_response = MockAnthropicResponse("Comprehensive analysis complete")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        self.mock_tool_manager.execute_tool.side_effect = [
            "Basic Python content", 
            "Course structure", 
            "Advanced content"
        ]
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="Compare basic and advanced Python concepts",
            tools=self.tool_definitions,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify all 3 tools were executed (2 in round 1, 1 in round 2)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 3)
        
        # Verify message structure includes all tool results
        round2_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = round2_call_args["messages"]
        
        # The second call (index 1) contains messages after both rounds
        round2_call_args = mock_client.messages.create.call_args_list[1][1] 
        messages = round2_call_args["messages"]
        
        # Should have: user query, assistant round1, user round1 results, assistant round2, user round2 results
        self.assertEqual(len(messages), 5)
        
        # Round 1 tool results should contain both tools (message index 2)
        round1_tool_results = messages[2]["content"]
        self.assertEqual(len(round1_tool_results), 2)
        
        # Round 2 tool results should contain single tool (message index 4)
        round2_tool_results = messages[4]["content"]
        self.assertEqual(len(round2_tool_results), 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tool_manager = Mock()
        
    @patch('anthropic.Anthropic')
    def test_empty_response_handling(self, mock_anthropic_class):
        """Test handling of empty or malformed responses"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with empty content
        empty_response = Mock()
        empty_response.content = []
        mock_client.messages.create.return_value = empty_response
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute
        result = ai_gen.generate_response(
            query="Test query",
            tools=[],
            tool_manager=self.mock_tool_manager
        )
        
        # Should handle gracefully
        self.assertEqual(result, "No response generated")
        
    @patch('anthropic.Anthropic') 
    def test_no_tool_manager_with_tools(self, mock_anthropic_class):
        """Test behavior when tools are provided but no tool_manager"""
        # Mock Anthropic client  
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # AI responds directly (should not attempt tool use)
        direct_response = MockAnthropicResponse("Direct response without tools")
        mock_client.messages.create.return_value = direct_response
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Execute with tools but no tool manager
        result = ai_gen.generate_response(
            query="Test query",
            tools=[{"name": "test_tool"}],
            tool_manager=None
        )
        
        # Should return direct response
        self.assertIn("Direct response", result)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)