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
        "appUrl": "https://www.productfights.com",
        "max_personas": 1,
    }
    result = await agent.graph.ainvoke(
        initial_state,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )
    # Convert the AddableValuesDict to a regular dict and then to JSON

    def serialize_obj(obj):
        if hasattr(obj, 'model_dump'):  # Handle Pydantic models
            return obj.model_dump()
        elif isinstance(obj, (list, tuple)):  # Handle lists/tuples
            return [serialize_obj(item) for item in obj]
        elif isinstance(obj, dict):  # Handle dictionaries
            return {k: serialize_obj(v) for k, v in obj.items()}
        return str(obj)  # Fallback to string representation

    json_result = json.dumps(dict(result), default=serialize_obj)
    print("RESULT in JSON:")
    print(json_result)

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
