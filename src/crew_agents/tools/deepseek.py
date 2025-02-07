from typing import Optional
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class QueryInput(BaseModel):
    query: str = Field(description="The query to process using the DeepSeek LLM")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt to guide the model's behavior")
    temperature: float = Field(default=0.2, description="Controls randomness in the output (0.0 = deterministic, 1.0 = creative)")

def get_groq_llm():
    return ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)

class DeepSeekTool(BaseTool):
    name: str = "DeepSeek Query Tool"
    description: str = "A tool that processes queries using the DeepSeek LLM. Use this for general question answering, analysis, and text generation tasks."
    args_schema: type[BaseModel] = QueryInput

    def _run(self, query: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        Process a query using the Groq DeepSeek LLM and return the result as a string.
        
        Args:
            query: The query to process
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Controls randomness in the output (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            str: The processed result from the LLM
        """
        llm = get_groq_llm()
        
        if system_prompt:
            messages = [
                HumanMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
        else:
            messages = [HumanMessage(content=query)]
        
        response = llm.invoke(messages)
        return response.content


