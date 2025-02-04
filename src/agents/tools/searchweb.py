from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults,TavilyAnswer
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from agents.llmtools import get_llm

#Another scrape to consider https://github.com/dendrite-systems/dendrite-python-sdk

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

    import os
    tavily_search = TavilySearchResults(
        max_results=max_results,
        include_answer=False,
        include_raw_content=True,
        api_key=os.getenv("TAVILY_API_KEY")
    )
    # Search
    search_docs = tavily_search.invoke(query)
    # Check if search_docs is a string (likely an error)
    if isinstance(search_docs, str):
        print(f"Error in search results: {search_docs}")
        return []  # Return empty list to avoid downstream errors
   
    return [SearchResult(link=doc["url"], content=doc["content"]) for doc in search_docs]


def search_web_get_answer(query: str)->str:
    
    """ Retrieve docs from web search and answer the query """

    tavily_answer = TavilyAnswer()

    # Search
    answer = tavily_answer.invoke(query)

    return answer

async def scrape_web(url: str) -> str:
    """Scrape the web page asynchronously"""
    loader = WebBaseLoader(url)
    # Load the page asynchronously
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    #print(docs[0].page_content[:100])
    return docs[0]

async def scrape_web_agent(url: str, query: str, output_model: type[BaseModel]) -> BaseModel:
    llm = get_llm()
    doc = await scrape_web(url)
    structured_llm = llm.with_structured_output(output_model)
    result = await structured_llm.ainvoke(
        [query + "\n\n" + doc.page_content],
        config={"temperature": 0.3}
    )
    return result

async def use_browser(query: str, output_model: type[BaseModel], max_steps: int = 10) -> BaseModel:
        llm = get_llm()
        controller = Controller()

        @controller.registry.action('Done with task', param_model=output_model)
        async def done(params: output_model):
            #print(f"[CONTROLLER] Done with task: {params.model_dump_json()}")
            result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
            return result

        browser_agent = Agent(
            task=query,
            llm=llm,
            browser=Browser(
                config=BrowserConfig(
                    headless=False,
                    proxy=None,
                )
            ),
            controller=controller,
            save_conversation_path="logs/conversation.json"
        )
            
        result = await browser_agent.run(max_steps=max_steps)
        final_result = result.final_result()
        return final_result