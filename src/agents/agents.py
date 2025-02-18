from dataclasses import dataclass
from typing import Literal, Union, Callable

from langgraph.graph.state import CompiledStateGraph

from agents.college_finder_agent.team_roster_agent import team_roster_agent
from agents.college_finder_agent.college_agent import college_finder_agent
from agents.marketing_agent.marketing_agent import marketing_agent
from agents.privateagents.private.bargpt_agent.bargpt_trending_flow import BarGPTTrendingPostFlow
from api_schema import AgentInfo
from core.crew_agent import CrewAgent
from crew_agents.vacation_house_agent.vacation_house_agent import VacationHouseAgent

DEFAULT_AGENT = "marketing-agent"


@dataclass
class Agent:
    description: str
    type: Literal["LANGGRAPH", "CREW"]
    graph: Union[CompiledStateGraph, CrewAgent, Callable[[], CrewAgent]] | None = None


def get_vacation_house_agent():
    return VacationHouseAgent()

def get_bargpt_trending_agent():
    return BarGPTTrendingPostFlow()


all_agents: dict[str, Agent] = {
    #ADD Agents HERE
   "marketing-agent": Agent(description="A marketing agent.", graph=marketing_agent, type="LANGGRAPH"),
   "college-agent": Agent(description="A college agent.", graph=college_finder_agent, type="LANGGRAPH"),
   "team-roster-agent": Agent(description="A team roster agent.", graph=team_roster_agent, type="LANGGRAPH"),
   "vacation-house-agent": Agent(description="An agent to help find vacation houses.", graph=get_vacation_house_agent(), type="CREW"),
   ## Private Agents
   "bargpt-trending-agent": Agent(description="An agent to help find trending topics.", graph=get_bargpt_trending_agent(), type="CREW"),
}


def get_agent(agent_id: str) -> Union[CompiledStateGraph, CrewAgent]:
    agent = all_agents[agent_id].graph
    if callable(agent):
        return agent()
    return agent


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in all_agents.items()
    ]
