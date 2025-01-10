from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from typing import  Annotated

from agents.tools.searchweb import SearchResult

class Persona(BaseModel):
    name: str = Field(
        description="Name of the persona"
    )
    description: str = Field(
        description="Description of the persona focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nDescription: {self.description}\n"

class Competitor(BaseModel):
    name: str = Field(
        description="Name of the competitor"
    )
    description: str = Field(
        description="Description of the persona focus, concerns, and motives.",
    )
    url: str = Field(
        description="URL of the competitor",
    )

class MarketingInput(TypedDict):
    appName: str # App Name
    appUrl: str  # App URL
    max_personas: int = 2 # Number of analysts
    competitor_hint: str | None # Competitor hint

class MarketingPlanState(TypedDict):
    appName: str # App Name
    appUrl: str | None # App URL
    competitor_hint: str | None # Competitor hint
    appDescription: str # App Description
    keyfeatures: List[str] # Key features
    value_proposition: str # Value proposition
    max_personas: int # Number of analysts
    human_feedback: str # Human feedback
    personas: List[Persona] # Analyst asking questions
    competitors: List[Competitor] # Competitors
    keywords: Annotated[List[str], operator.add] # Keywords
    tagline: str # Tagline
    subreddits: Annotated[List[str], operator.add] # Subreddits, the operator add seems to cause duplicates
    marketing_suggestions: List[str] # Marketing suggestions
    search_results: Annotated[List[SearchResult], operator.add] # Search results