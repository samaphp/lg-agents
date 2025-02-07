
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
groq_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)

def get_llm():
    return llm

## Too many limitations right now
def get_groq_llm():
    return groq_llm
