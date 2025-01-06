from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from agents.llmtools import get_llm

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def search_web(instructions: str, max_results: int = 3):
    """ Retrieve docs from web search after generating a search query from llm"""

    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([instructions])
    
    return search_web_with_query(search_query.search_query, max_results)


def search_web_with_query(query: str, max_results: int = 3):
    
    """ Retrieve docs from web search """

    tavily_search = TavilySearchResults(max_results=max_results)

    # Search
    search_docs = tavily_search.invoke(query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"docresults": [formatted_search_docs]} 