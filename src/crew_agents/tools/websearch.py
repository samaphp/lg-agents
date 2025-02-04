
from agents.tools.searchweb import scrape_web, search_web_with_query
from crewai.tools import BaseTool

class WebSearchTool(BaseTool):
    name: str ="Web Search Tools"
    description: str = ("Search the web for websites that match the user's query.")

    def _run(self, query: str) -> str:
        return search_web_with_query(query,10)
    
class ScrapeWebTool(BaseTool):
    name: str ="Scrape Web Tool"
    description: str = ("Scrape a web page and return the text.")
    async def _run(self, url: str) -> str:
        result = await scrape_web(url)
        return result