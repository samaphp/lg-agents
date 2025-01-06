from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from typing import  Annotated

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



class MarketingPlanState(TypedDict):
    appName: str # App Name
    max_personas: int # Number of analysts
    human_feedback: str # Human feedback
    personas: List[Persona] # Analyst asking questions
    competitors: List[Competitor] # Competitors
    search_results: Annotated[list, operator.add] # Source docs
