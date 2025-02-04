from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.college_finder_agent.team_roster_agent import team_roster_agent
from agents.college_finder_agent.college_agent import college_finder_agent
from agents.marketing_agent.marketing_agent import marketing_agent
from api_schema import AgentInfo

DEFAULT_AGENT = "marketing-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


all_agents: dict[str, Agent] = {
    #ADD Agents HERE
   "marketing-agent": Agent(description="A marketing agent.", graph=marketing_agent),
   "college-agent": Agent(description="A college agent.", graph=college_finder_agent),
   "team-roster-agent": Agent(description="A team roster agent.", graph=team_roster_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return all_agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in all_agents.items()
    ]
