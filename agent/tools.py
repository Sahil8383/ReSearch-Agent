"""Tools for the agent to interact with external services"""

import os

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
