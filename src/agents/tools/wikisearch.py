from pydantic import BaseModel, Field
from langchain_community.document_loaders import WikipediaLoader
from agents.llmtools import get_llm

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def search_wikipedia(instructions: str, max_results: int = 3):
    """ Retrieve docs from wikipedia search after generating a search query from llm"""

    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([instructions])
    
    return search_wikipedia_with_query(search_query.search_query, max_results)


def search_wikipedia_with_query(query: str, max_results: int = 3):
    
    """ Retrieve docs from wikipedia search """

    # Search
    search_docs = WikipediaLoader(query=query, 
                                  load_max_docs=max_results).load()
     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"docresults": [formatted_search_docs]} 