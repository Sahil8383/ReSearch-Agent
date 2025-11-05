"""Tools for the agent to interact with external services"""

import os
import sys
import io
import contextlib
from typing import Optional

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


class WebSearchTool:
    """Web search tool using Tavily API"""
    
    def __init__(self):
        """Initialize Tavily search client"""
        if TavilyClient is None:
            raise ImportError(
                "tavily-python package not installed. "
                "Install with: pip install tavily-python"
            )
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment. "
                "Get your API key at https://tavily.com"
            )
        self.client = TavilyClient(api_key=api_key)
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Perform web search and return formatted results"""
        try:
            print(f"🔍 Searching the web for: '{query}'")
            response = self.client.search(query=query, max_results=max_results)
            
            results = response.get('results', [])
            if not results:
                return "No results found."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                snippet = result.get('content', 'No description')
                url = result.get('url', '')
                formatted_results.append(
                    f"{i}. {title}\n   {snippet}\n   URL: {url}"
                )
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing search: {str(e)}"


class CodeExecutionTool:
    """Python code execution tool with safety features"""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize the code execution tool
        
        Args:
            timeout: Maximum execution time in seconds (default: 10)
        """
        self.timeout = timeout
    
    def execute(self, code: str) -> str:
        """
        Execute Python code and return the output
        
        Args:
            code: Python code string to execute
            
        Returns:
            String containing the output, error message, or execution result
        """
        try:
            print(f"🐍 Executing Python code...")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Create a namespace for code execution (isolated from main namespace)
            exec_namespace = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
            }
            
            # Try to capture the last expression result
            result = None
            
            try:
                with contextlib.redirect_stdout(stdout_capture), \
                     contextlib.redirect_stderr(stderr_capture):
                    
                    # Compile the code first to catch syntax errors
                    try:
                        compiled_code = compile(code, '<string>', 'exec')
                    except SyntaxError as e:
                        return f"Syntax Error: {str(e)}\nLine {e.lineno}: {e.text}"
                    
                    # Execute the code
                    exec(compiled_code, exec_namespace)
                    
                    # Try to get the result of the last expression if it's a single expression
                    # This is a bit tricky - we'll just show stdout/stderr for now
                    # If the code is a single expression, the user should use print() or assign to a variable
                    
            except KeyboardInterrupt:
                return "Execution interrupted by timeout or user."
            except Exception as e:
                error_msg = f"Runtime Error: {type(e).__name__}: {str(e)}"
                stderr_content = stderr_capture.getvalue()
                if stderr_content:
                    error_msg += f"\nStderr: {stderr_content}"
                return error_msg
            
            # Collect outputs
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            # Format the result
            output_parts = []
            
            if stdout_content:
                output_parts.append(f"Output:\n{stdout_content}")
            
            if stderr_content:
                output_parts.append(f"Errors:\n{stderr_content}")
            
            if not output_parts:
                # If no output, check if we can find any variables that might have been assigned
                # This is a simple heuristic - if code ends with a variable name, try to show it
                code_lines = code.strip().split('\n')
                last_line = code_lines[-1].strip()
                
                # If last line looks like a variable assignment or expression, show it
                if last_line and not last_line.startswith('#') and not last_line.startswith('print'):
                    # Try to evaluate the last expression if it's a simple expression
                    try:
                        # Only try this for simple expressions (not statements)
                        if '=' not in last_line and not last_line.startswith(('if', 'for', 'while', 'def', 'class', 'import', 'from')):
                            eval_namespace = exec_namespace.copy()
                            with contextlib.redirect_stdout(io.StringIO()), \
                                 contextlib.redirect_stderr(io.StringIO()):
                                result = eval(last_line, eval_namespace)
                                if result is not None:
                                    output_parts.append(f"Result: {result}")
                    except:
                        pass  # If eval fails, just show nothing (exec already ran)
            
            if output_parts:
                return "\n".join(output_parts)
            else:
                return "Code executed successfully (no output)."
                
        except Exception as e:
            return f"Error executing code: {str(e)}"
