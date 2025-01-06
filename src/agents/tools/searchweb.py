from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from agents.llmtools import get_llm

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class SearchResult(BaseModel):
    link:str = Field(None, description="Link to the search result.")
    content:str = Field(None, description="Content of the search result.")

def search_web(instructions: str, max_results: int = 3):
    """ Retrieve docs from web search after generating a search query from llm"""

    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([instructions])
    print("SEARCH QUERY: ", search_query.search_query)
    return search_web_with_query(search_query.search_query, max_results)


def search_web_with_query(query: str, max_results: int = 3):
    
    """ Retrieve docs from web search """

    tavily_search = TavilySearchResults(max_results=max_results)

    # Search
    search_docs = tavily_search.invoke(query)

    return [SearchResult(link=doc["url"], content=doc["content"]) for doc in search_docs]
  