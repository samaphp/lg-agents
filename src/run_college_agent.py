import asyncio
from uuid import uuid4
import json
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from agents.college_finder_agent.college_agent_schema import CollegeFinderInput, CollegeFinderState
from agents.marketing_agent.marketing_schema import MarketingInput, MarketingPlanState

load_dotenv()

from agents import DEFAULT_AGENT, all_agents  # noqa: E402

agent = all_agents["college-agent"]


async def main() -> None:
    #inputs = {"messages": [("user", "Create a marketing plan for a new app called SkyAssistant a engagement tool for BlueSky")]}
    initial_state: CollegeFinderInput = {
        "major": "any major",
        "location_preference": "Virginia",
        "min_acceptance_rate": 30,
        "max_colleges": 10,
        "search_query": "division 3 baseball schools campus size greater than 1500 students",
        "sat_score": 1200,
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

    # Remove messages from the result before serializing
    if "messages" in result:
        del result["messages"]

    json_result = json.dumps(dict(result), default=serialize_obj,indent=2)
    print("RESULT in JSON:")
    print(json_result)


if __name__ == "__main__":
    asyncio.run(main())
