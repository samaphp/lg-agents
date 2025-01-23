# Define data models
import operator
from pydantic import BaseModel
from typing import Annotated, Sequence, TypeVar, List, Union, Literal
from typing_extensions import TypedDict

class Player(BaseModel):
    name: str
    height: str | None = None
    hometown: str | None = None
    position: str | None = None
    handedness: str | None = None
    velocity: str | None = None
    pg_grade: str | None = None
    links: List[str] = []

class Team(BaseModel):
    team_name: str
    players: Annotated[List[Player], operator.add]

class RosterAgentInput(BaseModel):
    college_name: str

class TeamRosterState(TypedDict):
    """State for the team roster agent."""
    college_name: str
    roster_url: str | None
    team: Team | None
    summary: str | None