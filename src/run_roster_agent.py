import asyncio
import os
from uuid import uuid4
import json
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableConfig

from agents.college_finder_agent.team_roster_agent import create_team_roster_graph, RosterAgentInput



async def main() -> None:
    load_dotenv()
    dotenv_path = find_dotenv()
    print(f"Found .env at: {dotenv_path}")
    if not dotenv_path:
        print("No .env file found!")
    #print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    #print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))
    # Create the agent
    agent = create_team_roster_graph()

    # Initialize state
    initial_state = RosterAgentInput(
        college_name="University of Scranton"
    )

    # Run the agent
    result = await agent.ainvoke(
        initial_state,
        config=RunnableConfig(configurable={"thread_id": uuid4()}),
    )

    def serialize_obj(obj):
        if hasattr(obj, 'model_dump'):  # Handle Pydantic models
            return obj.model_dump()
        elif isinstance(obj, (list, tuple)):  # Handle lists/tuples
            return [serialize_obj(item) for item in obj]
        elif isinstance(obj, dict):  # Handle dictionaries
            return {k: serialize_obj(v) for k, v in obj.items()}
        return str(obj)  # Fallback to string representation

    # Convert result to JSON
    json_result = json.dumps(dict(result), default=serialize_obj, indent=2)
    print("\nRESULT in JSON:")
    print(json_result)

if __name__ == "__main__":
    asyncio.run(main())
