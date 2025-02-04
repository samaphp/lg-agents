from agents.tools.searchweb import scrape_web, search_web_with_query, use_browser
from crewai.tools import BaseTool
import asyncio  
from crew_agents.schemas import HomeMatches

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
    

class HomeFinderTool(BaseTool):
    name: str ="Home Finder Tool"
    description: str = ("Use a browser to search redfin.com for home listings that match the user's query.")
    def _run(self, goal: str ) -> str:
        result = asyncio.run(
            use_browser(
                f"""Go to redfin.com and search for homes that match the user's query: {goal}.  
                Return 3 results per city that most closely match the user's query.  
                Return the link to the home not the link to the search page.""",HomeMatches,25))
        return result
