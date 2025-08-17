#!/usr/bin/env python3
"""
Comprehensive test runner for RAG System debugging.

This script runs all test suites to identify and diagnose issues
causing "query failed" responses in the RAG chatbot system.

Usage:
    python run_rag_tests.py [--verbose] [--suite <suite_name>] [--fix]
    
Test suites:
    - course_search_tool: Tests CourseSearchTool.execute() method
    - ai_generator: Tests AI generator tool calling integration  
    - integration: End-to-end RAG system tests
    - all: Run all test suites (default)
    
Options:
    --verbose: Show detailed test output
    --suite: Run specific test suite only
    --fix: Show detailed fix recommendations
"""

import sys
import os
import unittest
import argparse
from io import StringIO
import traceback
from typing import Dict, List, Tuple, Any

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RAGTestResult:
    """Container for test results with analysis"""
    
    def __init__(self, suite_name: str, test_result, output: str):
        self.suite_name = suite_name
        self.test_result = test_result
        self.output = output
        self.tests_run = test_result.testsRun if hasattr(test_result, 'testsRun') else 0
        self.failures = test_result.failures if hasattr(test_result, 'failures') else []
        self.errors = test_result.errors if hasattr(test_result, 'errors') else []
        self.success = test_result.wasSuccessful() if hasattr(test_result, 'wasSuccessful') else False
        
    @property
    def total_issues(self) -> int:
        return len(self.failures) + len(self.errors)
        
    @property
    def success_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return ((self.tests_run - self.total_issues) / self.tests_run) * 100


class RAGTestAnalyzer:
    """Analyzes test results and categorizes failures"""
    
    FAILURE_CATEGORIES = {
        'config_issues': [
            'max_results', 'MAX_RESULTS', 'config', 'configuration'
        ],
        'vector_store_issues': [
            'chroma', 'vector', 'search', 'embedding', 'ChromaDB', 'vector_store'
        ],
        'tool_calling_issues': [
            'anthropic', 'tool_use', 'api', 'tool_manager', 'execute_tool'
        ],
        'data_loading_issues': [
            'document', 'loading', 'course', 'add_course', 'process_course'
        ],
        'integration_issues': [
            'rag_system', 'query', 'session', 'response'
        ]
    }
    
    def __init__(self):
        self.categorized_failures: Dict[str, List] = {
            category: [] for category in self.FAILURE_CATEGORIES.keys()
        }
        self.categorized_failures['other'] = []
        
    def analyze_failure(self, test_name: str, failure_msg: str, error_type: str = 'failure') -> str:
        """Categorize a single failure and return the category"""
        failure_lower = failure_msg.lower()
        test_name_lower = test_name.lower()
        
        for category, keywords in self.FAILURE_CATEGORIES.items():
            if any(keyword.lower() in failure_lower or keyword.lower() in test_name_lower 
                   for keyword in keywords):
                self.categorized_failures[category].append({
                    'test': test_name,
                    'message': failure_msg,
                    'type': error_type
                })
                return category
                
        # If no category matches, add to 'other'
        self.categorized_failures['other'].append({
            'test': test_name,
            'message': failure_msg,
            'type': error_type
        })
        return 'other'
    
    def analyze_results(self, results: List[RAGTestResult]) -> Dict[str, Any]:
        """Analyze all test results and provide comprehensive analysis"""
        total_tests = sum(r.tests_run for r in results)
        total_failures = sum(len(r.failures) for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        # Categorize all failures and errors
        for result in results:
            for failure in result.failures:
                test_name = str(failure[0])
                failure_msg = failure[1]
                self.analyze_failure(test_name, failure_msg, 'failure')
                
            for error in result.errors:
                test_name = str(error[0])
                error_msg = error[1]
                self.analyze_failure(test_name, error_msg, 'error')
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0,
            'categorized_failures': self.categorized_failures,
            'suite_results': {r.suite_name: {
                'tests': r.tests_run,
                'failures': len(r.failures),
                'errors': len(r.errors),
                'success_rate': r.success_rate,
                'success': r.success
            } for r in results}
        }


class RAGTestRunner:
    """Main test runner for RAG system diagnostics"""
    
    def __init__(self):
        self.analyzer = RAGTestAnalyzer()
        
    def run_test_suite(self, suite_name: str, verbose: bool = False) -> RAGTestResult:
        """Run a specific test suite and return results"""
        
        # Configure test runner
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if verbose else 1,
            buffer=True
        )
        
        # Load appropriate test suite
        try:
            if suite_name == "course_search_tool":
                from test_course_search_tool import TestCourseSearchToolExecute, TestToolManagerIntegration
                suite = unittest.TestSuite()
                loader = unittest.TestLoader()
                suite.addTest(loader.loadTestsFromTestCase(TestCourseSearchToolExecute))
                suite.addTest(loader.loadTestsFromTestCase(TestToolManagerIntegration))
                
            elif suite_name == "ai_generator":
                from test_ai_generator_tool_calling import (
                    TestAIGeneratorBasicFunctionality, 
                    TestAIGeneratorToolCalling,
                    TestAIGeneratorRealToolIntegration
                )
                suite = unittest.TestSuite()
                loader = unittest.TestLoader()
                suite.addTest(loader.loadTestsFromTestCase(TestAIGeneratorBasicFunctionality))
                suite.addTest(loader.loadTestsFromTestCase(TestAIGeneratorToolCalling))
                suite.addTest(loader.loadTestsFromTestCase(TestAIGeneratorRealToolIntegration))
                
            elif suite_name == "integration":
                from test_rag_system_integration import (
                    TestRAGSystemInitialization,
                    TestRAGSystemDataLoading,
                    TestRAGSystemQueryProcessing,
                    TestRAGSystemErrorScenarios,
                    TestRAGSystemPerformanceAndStress
                )
                suite = unittest.TestSuite()
                loader = unittest.TestLoader()
                suite.addTest(loader.loadTestsFromTestCase(TestRAGSystemInitialization))
                suite.addTest(loader.loadTestsFromTestCase(TestRAGSystemDataLoading))
                suite.addTest(loader.loadTestsFromTestCase(TestRAGSystemQueryProcessing))
                suite.addTest(loader.loadTestsFromTestCase(TestRAGSystemErrorScenarios))
                suite.addTest(loader.loadTestsFromTestCase(TestRAGSystemPerformanceAndStress))
                
            else:
                raise ValueError(f"Unknown test suite: {suite_name}")
                
        except ImportError as e:
            # Create a dummy failed result for import errors
            class DummyResult:
                def __init__(self, error_msg):
                    self.testsRun = 0
                    self.failures = [("import_error", error_msg)]
                    self.errors = []
                def wasSuccessful(self):
                    return False
                    
            return RAGTestResult(suite_name, DummyResult(str(e)), str(e))
        
        # Run the tests
        result = runner.run(suite)
        output = stream.getvalue()
        stream.close()
        
        return RAGTestResult(suite_name, result, output)
    
    def run_all_suites(self, verbose: bool = False) -> List[RAGTestResult]:
        """Run all test suites"""
        suites = ["course_search_tool", "ai_generator", "integration"]
        results = []
        
        for suite_name in suites:
            print(f"\n{'='*20} Running {suite_name} tests {'='*20}")
            try:
                result = self.run_test_suite(suite_name, verbose)
                results.append(result)
                
                if verbose:
                    print(result.output)
                else:
                    status = "✅ PASSED" if result.success else "❌ FAILED"
                    print(f"{status} - {result.tests_run} tests, {len(result.failures)} failures, {len(result.errors)} errors")
                    
            except Exception as e:
                print(f"💥 ERROR running {suite_name}: {e}")
                if verbose:
                    traceback.print_exc()
                    
        return results
    
    def print_summary_analysis(self, results: List[RAGTestResult]):
        """Print comprehensive analysis of test results"""
        analysis = self.analyzer.analyze_results(results)
        
        print("\n" + "="*80)
        print("RAG SYSTEM TEST ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total tests run: {analysis['total_tests']}")
        print(f"   Total failures: {analysis['total_failures']}")
        print(f"   Total errors: {analysis['total_errors']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        
        # Suite-by-suite breakdown
        print(f"\n📋 SUITE BREAKDOWN:")
        for suite_name, suite_data in analysis['suite_results'].items():
            status_icon = "✅" if suite_data['success'] else "❌"
            print(f"   {status_icon} {suite_name}:")
            print(f"      Tests: {suite_data['tests']}, "
                  f"Failures: {suite_data['failures']}, "
                  f"Errors: {suite_data['errors']}, "
                  f"Success: {suite_data['success_rate']:.1f}%")
        
        # Categorized failure analysis
        print(f"\n🔍 FAILURE ANALYSIS BY CATEGORY:")
        categorized = analysis['categorized_failures']
        
        for category, failures in categorized.items():
            if failures:
                category_name = category.replace('_', ' ').title()
                print(f"\n   🚨 {category_name} ({len(failures)} issues):")
                
                for i, failure in enumerate(failures[:3]):  # Show first 3 of each category
                    test_name = failure['test'].split('.')[-1] if '.' in failure['test'] else failure['test']
                    print(f"      {i+1}. {test_name}")
                    
                    # Show first line of error for brevity
                    first_line = failure['message'].split('\n')[0] if failure['message'] else "No details"
                    if len(first_line) > 100:
                        first_line = first_line[:97] + "..."
                    print(f"         {first_line}")
                
                if len(failures) > 3:
                    print(f"      ... and {len(failures) - 3} more issues")
    
    def print_fix_recommendations(self, results: List[RAGTestResult]):
        """Print specific fix recommendations based on test results"""
        analysis = self.analyzer.analyze_results(results)
        categorized = analysis['categorized_failures']
        
        print("\n" + "="*80)
        print("RECOMMENDED FIXES FOR RAG SYSTEM")
        print("="*80)
        
        fixes_recommended = False
        
        # Configuration Issues (CRITICAL)
        if categorized.get('config_issues'):
            fixes_recommended = True
            print(f"\n🔧 CRITICAL: Configuration Issues")
            print(f"   Problem: MAX_RESULTS=0 in config.py causes all searches to return empty results")
            print(f"   Fix: Change line 21 in backend/config.py:")
            print(f"        From: MAX_RESULTS: int = 0")
            print(f"        To:   MAX_RESULTS: int = 5")
            print(f"   Impact: This will immediately fix the 'query failed' issue")
            
        # Vector Store Issues
        if categorized.get('vector_store_issues'):
            fixes_recommended = True
            print(f"\n🔧 Vector Store Issues")
            print(f"   - Check ChromaDB installation: uv add chromadb")
            print(f"   - Verify vector store initialization in RAGSystem")
            print(f"   - Check if course documents are being properly loaded")
            print(f"   - Verify embedding model (sentence-transformers) is working")
            print(f"   - Check file permissions for ChromaDB storage directory")
            
        # Tool Calling Issues  
        if categorized.get('tool_calling_issues'):
            fixes_recommended = True
            print(f"\n🔧 Tool Calling Issues")
            print(f"   - Verify ANTHROPIC_API_KEY is set in .env file")
            print(f"   - Check tool definitions match Anthropic's expected format")
            print(f"   - Verify tool_manager is properly passed to ai_generator")
            print(f"   - Check tool execution flow in _handle_tool_execution method")
            print(f"   - Test API connectivity with a simple Anthropic API call")
            
        # Data Loading Issues
        if categorized.get('data_loading_issues'):
            fixes_recommended = True
            print(f"\n🔧 Data Loading Issues")
            print(f"   - Verify docs/ folder exists and contains course files")
            print(f"   - Check document format matches expected structure")
            print(f"   - Verify document_processor parsing logic")
            print(f"   - Check course metadata extraction")
            print(f"   - Ensure file permissions allow reading course documents")
            
        # Integration Issues
        if categorized.get('integration_issues'):
            fixes_recommended = True
            print(f"\n🔧 Integration Issues")
            print(f"   - Check component initialization order in RAGSystem")
            print(f"   - Verify tool registration happens correctly")
            print(f"   - Check query processing pipeline end-to-end")
            print(f"   - Verify session management is working")
            
        # Other Issues
        if categorized.get('other'):
            fixes_recommended = True
            print(f"\n🔧 Other Issues")
            print(f"   - Check basic Python environment and dependencies")
            print(f"   - Verify all imports are working correctly")
            print(f"   - Check for any missing configuration")
            print(f"   - Review error logs for additional context")
            
        if fixes_recommended:
            print(f"\n📋 IMMEDIATE ACTION PLAN:")
            print(f"   1. 🚨 URGENT: Fix MAX_RESULTS=0 in config.py")
            print(f"   2. 🔍 Verify: Run individual test suites with --verbose")
            print(f"   3. ✅ Test: Run quick manual test after config fix")
            print(f"   4. 🔄 Validate: Re-run full test suite to confirm fixes")
            
            print(f"\n💡 QUICK VERIFICATION TEST:")
            print(f"   After fixing config.py, run this manual test:")
            print(f"   ```python")
            print(f"   from rag_system import RAGSystem")
            print(f"   from config import config")
            print(f"   rag = RAGSystem(config)")
            print(f"   result = rag.search_tool.execute('test query')")
            print(f"   print(f'Result: {{result}}')  # Should not be 'No relevant content found.'")
            print(f"   ```")
        else:
            print(f"\n✅ No major issues detected in test results!")
            print(f"   All test suites appear to be passing successfully.")
            
    def run_quick_diagnostic(self):
        """Run a quick diagnostic to identify the most critical issues"""
        print("🔍 Running Quick RAG System Diagnostic...")
        print("="*50)
        
        # Check configuration first
        try:
            from config import config
            print(f"📋 Configuration Check:")
            print(f"   MAX_RESULTS: {config.MAX_RESULTS}")
            if config.MAX_RESULTS == 0:
                print(f"   🚨 CRITICAL ISSUE FOUND: MAX_RESULTS=0")
                print(f"   This is the likely cause of 'query failed' responses!")
            else:
                print(f"   ✅ MAX_RESULTS looks good")
                
            print(f"   ANTHROPIC_API_KEY: {'Set' if config.ANTHROPIC_API_KEY else 'Not set'}")
            print(f"   CHROMA_PATH: {config.CHROMA_PATH}")
            
        except ImportError as e:
            print(f"   ❌ Cannot import config: {e}")
            
        # Check if dependencies are available
        print(f"\n📦 Dependency Check:")
        deps = ['anthropic', 'chromadb', 'sentence_transformers', 'fastapi']
        for dep in deps:
            try:
                __import__(dep)
                print(f"   ✅ {dep}: Available")
            except ImportError:
                print(f"   ❌ {dep}: Missing")
                
        print(f"\n💡 Run 'python run_rag_tests.py --verbose' for full analysis")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="RAG System Test Runner and Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rag_tests.py                    # Run all tests
  python run_rag_tests.py --verbose          # Run with detailed output  
  python run_rag_tests.py --suite integration # Run integration tests only
  python run_rag_tests.py --fix              # Show detailed fix recommendations
  python run_rag_tests.py --quick            # Quick diagnostic only
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed test output")
    parser.add_argument("--suite", "-s", 
                       choices=["course_search_tool", "ai_generator", "integration", "all"],
                       default="all", help="Test suite to run")
    parser.add_argument("--fix", "-f", action="store_true",
                       help="Show detailed fix recommendations") 
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick diagnostic only")
    
    args = parser.parse_args()
    
    runner = RAGTestRunner()
    
    if args.quick:
        runner.run_quick_diagnostic()
        return
    
    print("RAG System Comprehensive Test Suite")
    print("="*50)
    print("This test suite will identify why the RAG chatbot")
    print("returns 'query failed' for content-related questions.")
    print()
    
    # Run tests
    if args.suite == "all":
        results = runner.run_all_suites(args.verbose)
    else:
        print(f"Running {args.suite} tests...")
        result = runner.run_test_suite(args.suite, args.verbose)
        results = [result]
        
        if args.verbose:
            print(result.output)
        else:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"{status} - {result.tests_run} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    
    # Print analysis
    if results:
        runner.print_summary_analysis(results)
        
        if args.fix or any(not r.success for r in results):
            runner.print_fix_recommendations(results)
            
        # Return appropriate exit code
        if all(r.success for r in results):
            print(f"\n🎉 All tests passed! RAG system appears to be working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some tests failed. Please review the analysis and apply recommended fixes.")
            sys.exit(1)
    else:
        print("No test results to analyze.")
        sys.exit(1)


if __name__ == "__main__":
    main()