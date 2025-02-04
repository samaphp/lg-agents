
from agents.tools.searchweb import scrape_web, search_web_with_query, use_browser
from crewai.tools import BaseTool
import asyncio  

class WebSearchTool(BaseTool):
    name: str ="Web Search Tools"
    description: str = ("Search the web for websites that match the user's query.")

    def _run(self, query: str) -> str:
        return search_web_with_query(query,10)
    
class ScrapeWebTool(BaseTool):
    name: str ="Scrape Web Tool"
    description: str = ("Scrape a web page and return the text.")
    def _run(self, url: str) -> str:
        result = asyncio.run(scrape_web(url))
        return result
    

class BrowserUseTool(BaseTool):
    name: str ="Browser Use Tool"
    description: str = ("Use a browser to search the web and navigate urls to achieve a goal.")
    def _run(self, goal: str) -> str:
        result = asyncio.run(use_browser(goal))
        return result