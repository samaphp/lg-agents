import asyncio
from uuid import uuid4
import json
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from agents.marketing_agent.marketing_schema import MarketingPlanState

load_dotenv()

from agents import DEFAULT_AGENT, all_agents  # noqa: E402

agent = all_agents[DEFAULT_AGENT]


async def main() -> None:
    #inputs = {"messages": [("user", "Create a marketing plan for a new app called SkyAssistant a engagement tool for BlueSky")]}
    initial_state: MarketingPlanState = {
        "appName": "Saas Fights a SaaS battle for the best SaaS tools",
        "max_personas": 1,
    }
    thread = {"configurable": {"thread_id": "1"}}

    # Convert the AddableValuesDict to a regular dict and then to JSON
    async for event in agent.graph.astream(initial_state, thread, stream_mode="values"):
        # Review
       print("[EVENT]", event)


if __name__ == "__main__":
    asyncio.run(main())
