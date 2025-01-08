from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults,TavilyAnswer
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field

from agents.llmtools import get_llm

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class SearchResult(BaseModel):
    link:str = Field(None, description="Link to the search result.")
    content:str = Field(None, description="Content of the search result.")

def search_web(instructions: str, max_results: int = 3)->List[SearchResult]:
    """ Retrieve docs from web search after generating a search query from llm"""

    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([instructions])
    print("SEARCH QUERY: ", search_query.search_query)
    return search_web_with_query(search_query.search_query, max_results)


def search_web_with_query(query: str, max_results: int = 3)->List[SearchResult]:
    
    """ Retrieve docs from web search """

    tavily_search = TavilySearchResults(max_results=max_results,include_answer=True,include_raw_content=True)

    # Search
    search_docs = tavily_search.invoke(query)

    return [SearchResult(link=doc["url"], content=doc["content"]) for doc in search_docs]


def search_web_get_answer(query: str)->str:
    
    """ Retrieve docs from web search and answer the query """

    tavily_answer = TavilyAnswer()

    # Search
    answer = tavily_answer.invoke(query)

    return answer

def scrape_web(url: str)->str:
    """ Scrape the web page """

    loader = WebBaseLoader(url)

    # Load the page
    docs = loader.load()

    resultdoc = docs[0]
    #print(resultdoc.page_content)
    #print(resultdoc.metadata)

    return resultdoc