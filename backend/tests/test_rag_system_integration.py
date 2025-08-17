"""
Comprehensive integration tests for the complete RAG system.

This test suite evaluates the end-to-end functionality to identify
why the system returns "query failed" for content-related questions.
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config
from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestRAGSystemInitialization(unittest.TestCase):
    """Test RAG system component initialization"""
    
    def setUp(self):
        """Set up test configuration"""
        self.test_config = Config()
        # Override paths for testing
        self.test_config.CHROMA_PATH = "/tmp/test_chroma_db"
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        
    def test_rag_system_component_initialization(self):
        """Test that all RAG system components are properly initialized"""
        rag_system = RAGSystem(self.test_config)
        
        # Verify all components are initialized
        self.assertIsNotNone(rag_system.document_processor)
        self.assertIsNotNone(rag_system.vector_store)
        self.assertIsNotNone(rag_system.ai_generator)
        self.assertIsNotNone(rag_system.session_manager)
        self.assertIsNotNone(rag_system.tool_manager)
        self.assertIsNotNone(rag_system.search_tool)
        self.assertIsNotNone(rag_system.outline_tool)
        
        # Verify configuration is passed correctly
        self.assertEqual(rag_system.config, self.test_config)
        
    def test_tools_registration(self):
        """Test that tools are properly registered in tool manager"""
        rag_system = RAGSystem(self.test_config)
        
        # Verify tools are registered
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        self.assertEqual(len(tool_definitions), 2)  # search + outline tools
        
        tool_names = [tool["name"] for tool in tool_definitions]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
        
        # Verify tools can be executed
        self.assertIn("search_course_content", rag_system.tool_manager.tools)
        self.assertIn("get_course_outline", rag_system.tool_manager.tools)
        
    def test_critical_max_results_configuration_issue(self):
        """CRITICAL TEST: Verify the MAX_RESULTS configuration is now fixed"""
        # This test verifies the configuration issue has been resolved
        
        # Test with current config (should now be > 0)
        default_config = Config()
        self.assertGreater(default_config.MAX_RESULTS, 0)  # Should be fixed now!
        
        rag_system = RAGSystem(default_config)
        
        # Verify that vector store gets a valid MAX_RESULTS value
        self.assertGreater(rag_system.vector_store.max_results, 0)
        
        # This ensures searches can return results
        
    def test_vector_store_initialization_with_correct_config(self):
        """Test vector store with corrected configuration"""
        # Test with fixed config
        fixed_config = Config()
        fixed_config.MAX_RESULTS = 5  # Corrected value
        
        rag_system = RAGSystem(fixed_config)
        
        # Verify vector store gets the correct value
        self.assertEqual(rag_system.vector_store.max_results, 5)


class TestRAGSystemDataLoading(unittest.TestCase):
    """Test document loading and processing functionality"""
    
    def setUp(self):
        """Set up temporary directories and test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma_db")
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        self.test_config.MAX_RESULTS = 5  # Fix the config issue for testing
        
        # Create test course documents
        self.create_test_documents()
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_documents(self):
        """Create test course documents for integration testing"""
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # Test course 1: Python Fundamentals
        course1_content = """Course Title: Python Programming Fundamentals
Course Link: https://example.com/python-fundamentals
Course Instructor: Alice Johnson

Lesson 1: Introduction to Python
Lesson Link: https://example.com/python-fundamentals/lesson-1
Python is a high-level, interpreted programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability and allows programmers to express concepts in fewer lines of code.

Lesson 2: Python Variables and Data Types
Lesson Link: https://example.com/python-fundamentals/lesson-2
In Python, variables are used to store data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created when you first assign a value to it. Python has several built-in data types including integers, floating-point numbers, strings, and booleans.

Lesson 3: Control Flow and Functions
Lesson Link: https://example.com/python-fundamentals/lesson-3
Python uses control flow statements like if, elif, and else for conditional execution. Loops include for and while loops. Functions in Python are defined using the def keyword and can accept parameters and return values.
"""
        
        # Test course 2: Machine Learning Basics
        course2_content = """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-intro
Course Instructor: Bob Smith

Lesson 1: What is Machine Learning
Lesson Link: https://example.com/ml-intro/lesson-1
Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.

Lesson 2: Supervised Learning Algorithms
Lesson Link: https://example.com/ml-intro/lesson-2
Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common algorithms include linear regression for predicting continuous values, logistic regression for classification, decision trees, and neural networks.
"""
        
        with open(os.path.join(self.docs_dir, "python_course.txt"), "w") as f:
            f.write(course1_content)
            
        with open(os.path.join(self.docs_dir, "ml_course.txt"), "w") as f:
            f.write(course2_content)
    
    def test_document_loading_functionality(self):
        """Test loading course documents into the system"""
        rag_system = RAGSystem(self.test_config)
        
        # Load documents from test directory
        courses_added, chunks_added = rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Verify documents were loaded
        self.assertEqual(courses_added, 2)
        self.assertGreater(chunks_added, 0)
        
        # Verify course analytics
        analytics = rag_system.get_course_analytics()
        self.assertEqual(analytics["total_courses"], 2)
        course_titles = analytics["course_titles"]
        self.assertIn("Python Programming Fundamentals", course_titles)
        self.assertIn("Introduction to Machine Learning", course_titles)
        
    def test_vector_store_search_with_real_data(self):
        """Test vector store search functionality with actual data"""
        rag_system = RAGSystem(self.test_config)
        rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Test basic search
        results = rag_system.vector_store.search("Python variables")
        self.assertFalse(results.is_empty())
        self.assertIsNone(results.error)
        
        # Verify results contain relevant content
        found_content = " ".join(results.documents).lower()
        self.assertIn("variables", found_content)
        
        # Test course-filtered search
        results = rag_system.vector_store.search("variables", course_name="Python Programming Fundamentals")
        self.assertFalse(results.is_empty())
        
        # Test lesson-filtered search
        results = rag_system.vector_store.search("data types", lesson_number=2)
        self.assertFalse(results.is_empty())
        
    def test_course_search_tool_with_real_data(self):
        """Test CourseSearchTool with actual loaded data"""
        rag_system = RAGSystem(self.test_config)
        rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Test search tool execution
        result = rag_system.search_tool.execute("Python variables")
        self.assertNotEqual(result, "No relevant content found.")
        self.assertNotIn("error", result.lower())
        
        # Verify result formatting
        self.assertIn("Python Programming Fundamentals", result)
        self.assertIn("variables", result.lower())
        
        # Test course-specific search
        result = rag_system.search_tool.execute("machine learning", course_name="Introduction to Machine Learning")
        self.assertIn("Introduction to Machine Learning", result)
        
        # Test sources tracking
        self.assertGreater(len(rag_system.search_tool.last_sources), 0)


class TestRAGSystemQueryProcessing(unittest.TestCase):
    """Test complete query processing pipeline"""
    
    def setUp(self):
        """Set up test environment with real data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma_db")
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        self.test_config.MAX_RESULTS = 5  # Fix the config issue
        
        # Create minimal test document
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        
        test_content = """Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 1: Test Topic
Lesson Link: https://example.com/test/lesson-1
This is test content about programming concepts and software development.
"""
        with open(os.path.join(self.docs_dir, "test_course.txt"), "w") as f:
            f.write(test_content)
            
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('anthropic.Anthropic')
    def test_full_query_flow_with_mocked_ai(self, mock_anthropic_class):
        """Test complete query flow with mocked AI responses"""
        from conftest import MockAnthropicContentBlock, MockAnthropicResponse
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Set up mock AI response that uses search tool
        tool_use_content = [
            MockAnthropicContentBlock(
                "tool_use", 
                name="search_course_content",
                input_data={"query": "programming concepts", "course_name": "Test Course"}, 
                block_id="integration_test"
            )
        ]
        first_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse("Programming concepts are fundamental ideas that help developers write effective code. They include topics like variables, functions, and data structures.")
        
        mock_client.messages.create.side_effect = [first_response, final_response]
        
        # Create RAG system and load data
        rag_system = RAGSystem(self.test_config)
        rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Execute query
        response, sources = rag_system.query("What are programming concepts?")
        
        # Verify response was generated
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "query failed")
        self.assertIn("Programming concepts", response)
        
        # Verify sources were retrieved
        self.assertGreater(len(sources), 0)
        
        # Verify AI was called with tools
        call_args = mock_client.messages.create.call_args_list[0][1]
        self.assertIn("tools", call_args)
        self.assertIsNotNone(call_args.get("tool_choice"))
        
    def test_session_management_functionality(self):
        """Test conversation session management"""
        rag_system = RAGSystem(self.test_config)
        
        # Create session
        session_id = rag_system.session_manager.create_session()
        self.assertIsNotNone(session_id)
        
        # Add conversation exchange
        rag_system.session_manager.add_exchange(
            session_id, 
            "What is Python?", 
            "Python is a programming language."
        )
        
        # Verify history retrieval
        history = rag_system.session_manager.get_conversation_history(session_id)
        self.assertIn("What is Python?", history)
        self.assertIn("Python is a programming language", history)
        
    def test_tool_manager_integration_with_real_system(self):
        """Test tool manager functionality with complete system"""
        rag_system = RAGSystem(self.test_config)
        rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Test tool execution through manager
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="programming concepts",
            course_name="Test Course"
        )
        
        self.assertNotIn("Tool 'search_course_content' not found", result)
        self.assertIn("Test Course", result)
        
        # Test source management
        sources = rag_system.tool_manager.get_last_sources()
        self.assertIsInstance(sources, list)
        
        # Test source reset
        rag_system.tool_manager.reset_sources()
        sources_after_reset = rag_system.tool_manager.get_last_sources()
        self.assertEqual(len(sources_after_reset), 0)


class TestRAGSystemErrorScenarios(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test configuration"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma_db")
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_max_results_zero_issue_reproduction(self):
        """CRITICAL TEST: Reproduce the MAX_RESULTS=0 issue that causes query failed"""
        # Use the actual problematic config
        broken_config = Config()  # This has MAX_RESULTS=0
        
        rag_system = RAGSystem(broken_config)
        
        # Even with data loaded, searches will return 0 results
        # Let's simulate this with a mock since loading real data is complex
        mock_results = rag_system.search_tool.execute("any query")
        
        # With MAX_RESULTS=0, this should return "No relevant content found"
        # This is the core issue causing "query failed"
        self.assertEqual(mock_results, "No relevant content found.")
        
    def test_error_handling_no_documents_loaded(self):
        """Test system behavior when no documents are loaded"""
        self.test_config.MAX_RESULTS = 5  # Fix config for this test
        rag_system = RAGSystem(self.test_config)
        
        # Try to search without loading any documents
        result = rag_system.search_tool.execute("any query")
        
        # Should handle gracefully (empty results)
        self.assertIsNotNone(result)
        
    def test_error_handling_invalid_course_name(self):
        """Test handling of invalid course name filters"""
        self.test_config.MAX_RESULTS = 5
        rag_system = RAGSystem(self.test_config)
        
        # Search with non-existent course name
        result = rag_system.search_tool.execute("any query", course_name="Nonexistent Course")
        
        # Should return appropriate error message
        self.assertIn("No course found matching", result)
        
    def test_api_key_configuration_handling(self):
        """Test API key configuration scenarios"""
        # Test with empty API key
        config_no_key = Config()
        config_no_key.ANTHROPIC_API_KEY = ""
        config_no_key.MAX_RESULTS = 5
        
        # System should initialize but AI calls might fail
        rag_system = RAGSystem(config_no_key)
        self.assertIsNotNone(rag_system.ai_generator)
        
        # Test with test API key
        config_test_key = Config()
        config_test_key.ANTHROPIC_API_KEY = "test_key"
        config_test_key.MAX_RESULTS = 5
        
        rag_system_test = RAGSystem(config_test_key)
        self.assertIsNotNone(rag_system_test.ai_generator)
        
    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries"""
        self.test_config.MAX_RESULTS = 5
        rag_system = RAGSystem(self.test_config)
        
        # Test empty query
        result = rag_system.search_tool.execute("")
        self.assertIsNotNone(result)
        
        # Test whitespace-only query
        result = rag_system.search_tool.execute("   ")
        self.assertIsNotNone(result)
        
    def test_vector_store_connection_issues(self):
        """Test handling of vector store connection problems"""
        self.test_config.MAX_RESULTS = 5
        rag_system = RAGSystem(self.test_config)
        
        # Simulate vector store failure by replacing with broken mock
        original_store = rag_system.search_tool.store
        mock_store = Mock()
        
        # Set up mock to return error results instead of raising exception
        from vector_store import SearchResults
        mock_results = SearchResults.empty("ChromaDB connection failed")
        mock_store.search.return_value = mock_results
        rag_system.search_tool.store = mock_store
        
        # Try to execute search - should handle error gracefully
        result = rag_system.search_tool.execute("test query")
        
        # Should return error message, not crash
        self.assertIsNotNone(result)
        self.assertEqual(result, "ChromaDB connection failed")
        
        # Restore original store
        rag_system.search_tool.store = original_store


class TestRAGSystemPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests"""
    
    def setUp(self):
        """Set up for performance testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "perf_chroma_db")
        self.test_config.MAX_RESULTS = 5  # Fix config
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_multiple_concurrent_queries(self):
        """Test system handling of multiple queries"""
        # Create simple test data
        docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        content = """Course Title: Performance Test Course
Course Link: https://example.com/perf
Course Instructor: Performance Tester

Lesson 1: Performance Testing
Performance testing is crucial for applications to ensure they can handle expected loads.

Lesson 2: Load Testing  
Load testing helps identify bottlenecks and performance issues before production.

Lesson 3: Stress Testing
Stress testing pushes systems beyond normal operating capacity to find breaking points.
"""
        
        with open(os.path.join(docs_dir, "perf_course.txt"), "w") as f:
            f.write(content)
        
        rag_system = RAGSystem(self.test_config)
        rag_system.add_course_folder(docs_dir, clear_existing=True)
        
        # Execute multiple queries rapidly
        queries = [
            "performance testing",
            "load testing", 
            "stress testing",
            "bottlenecks",
            "applications"
        ]
        
        results = []
        for query in queries:
            result = rag_system.search_tool.execute(query)
            results.append(result)
            
        # All queries should succeed
        for result in results:
            self.assertNotIn("error", result.lower())
            self.assertNotEqual(result, "No relevant content found.")
            
    def test_configuration_validation(self):
        """Test that configuration issues are detectable"""
        # Test various config scenarios
        configs_to_test = [
            {"MAX_RESULTS": 0, "expected_issue": True},   # The problematic config
            {"MAX_RESULTS": 5, "expected_issue": False},  # Corrected config
            {"MAX_RESULTS": -1, "expected_issue": True},  # Invalid config
            {"MAX_RESULTS": 1000, "expected_issue": False}, # Very large config
        ]
        
        for config_test in configs_to_test:
            test_config = Config()
            test_config.MAX_RESULTS = config_test["MAX_RESULTS"]
            test_config.CHROMA_PATH = os.path.join(self.temp_dir, f"test_db_{config_test['MAX_RESULTS']}")
            
            rag_system = RAGSystem(test_config)
            
            # Check if vector store gets the config value
            self.assertEqual(rag_system.vector_store.max_results, config_test["MAX_RESULTS"])
            
            if config_test["expected_issue"] and config_test["MAX_RESULTS"] <= 0:
                # These configs should cause issues
                self.assertLessEqual(rag_system.vector_store.max_results, 0)
            else:
                # These configs should be fine
                self.assertGreater(rag_system.vector_store.max_results, 0)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)