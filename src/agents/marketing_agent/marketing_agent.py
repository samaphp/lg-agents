from typing import Annotated, Sequence, TypeVar, List
from typing_extensions import TypedDict
from langgraph.graph import Graph, StateGraph
from agents.llmtools import get_llm
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from agents.marketing_agent.marketing_schema import Competitor, MarketingPlanState, Persona
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from agents.tools.searchweb import search_web_with_query
from agents.tools.wikisearch import search_wikipedia_with_query

class PersonaList(BaseModel):
    personas: List[Persona]

class CompetitorList(BaseModel):
    competitors: List[Competitor]

controller = Controller()


@controller.registry.action('Done with task', param_model=CompetitorList)
async def done(params: Competitor):
    print(f"[CONTROLLER] Done with task: {params.model_dump_json()}")
    result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
    return result

def create_marketing_graph() -> CompiledStateGraph:
    
    # Define state type
    workflow_state = TypeVar("workflow_state", bound=MarketingPlanState)

    # Create personas node
    def create_personas(state: workflow_state) -> workflow_state:
        prompt = f"""
        Create {state['max_personas']} buyer personas for {state['appName']}.
        Focus on their key characteristics, needs, and pain points.
        
        Return a list of personas, each with a name and description.
        """
        
        llm = get_llm()
        structured_llm = llm.with_structured_output(PersonaList)
        response = structured_llm.invoke(prompt)
        
        # Take only up to max_personas
        state["personas"] = response.personas[:state['max_personas']]
        print("ANALYSTS", state["personas"])
        return state
    
    async def search_web_for_competitors(state: workflow_state):
        results = search_wikipedia_with_query(f"Websites that are alternatives for {state['appName']}")
        print("SEARCH RESULTS", results)
        return {"search_results": results["docresults"]}

    # Research competitors node using Browser Use
    async def research_competitors(state: workflow_state) -> workflow_state:
        results = []

        llm = get_llm()

        for persona in state["personas"]:
            browser_agent = Agent(
                task=f"""Research competitors for {state['appName']} focusing on:
                Persona: {persona.persona}
                
                1. Find 3 main competitors using Google Search (avoid using G2)
                2. Analyze their features relevant to this persona and add that to the description
                """,
                llm=llm,
                browser=Browser(
                    config=BrowserConfig(
                        headless=True,
                        proxy=None,
                    )
                ),
                controller=controller
            )
            
            # Run browser agent to research competitors with max 15 steps
            # result = await browser_agent.run(max_steps=15)
            # print("RESULT FROM BROWSER USE", result)
            # final_result = result.final_result()
            # if final_result:
            #     parsed = CompetitorList.model_validate_json(final_result)
            #     results.extend(parsed.competitors)

       # state["competitors"]=results

        state["competitors"] = [
            Competitor(
                name="CompetitorA",
                description="A leading SaaS comparison platform focused on enterprise solutions",
                url="https://competitora.com"
            ),
            Competitor(
                name="CompetitorB", 
                description="Popular SaaS review site targeting small businesses",
                url="https://competitorb.com"
            ),
            Competitor(
                name="CompetitorC",
                description="AI-powered SaaS recommendation engine for startups",
                url="https://competitorc.com"
            )
        ]
        return state

    # Get human feedback node
    def get_feedback(state: workflow_state) -> workflow_state:
        # Here you could implement actual user interaction
        # For now we'll just store it in state
        state["human_feedback"] = "Feedback received"
        return state

    # End node to properly signal completion
    def end(state: workflow_state) -> workflow_state:
        # You can add any final state cleanup or validation here
        print("Marketing analysis completed:", state)
        return state

    # Create the graph
    workflow = StateGraph(MarketingPlanState)

    # Add nodes
    workflow.add_node("create_personas", create_personas)
    workflow.add_node("search_web_for_competitors", search_web_for_competitors)
    workflow.add_node("research_competitors", research_competitors)
    workflow.add_node("get_feedback", get_feedback)
    workflow.add_node("__END__", end)

    # Create edges
    workflow.add_edge("create_personas", "search_web_for_competitors")
    workflow.add_edge("search_web_for_competitors", "research_competitors")
    workflow.add_edge("research_competitors", "get_feedback")
    workflow.add_edge("get_feedback", "__END__")
    
    # Set entry point
    workflow.set_entry_point("create_personas")

    # Compile graph
    return workflow.compile(checkpointer=MemorySaver())

#define the graph
marketing_agent = create_marketing_graph()





# Helper function to run the graph
async def run_marketing_analysis(
    app_name: str,
    max_personas: int = 3
) -> MarketingPlanState:
    
    # Initialize state
    initial_state: MarketingPlanState = {
        "appName": app_name,
        "max_personas": max_personas,
        "human_feedback": "",
        "personas": []
    }
    
    # Create and run graph
    graph = create_marketing_graph()
    final_state = await graph.arun(initial_state)
    
    return final_state


# async def main():
#     llm = ChatOpenAI(model="gpt-4")
    
#     result = await run_marketing_analysis(
#         app_name="My Cool App",
#         llm=llm,
#         max_personas=3
#     )
    
#     # Access results
#     print("Personas:")
#     for persona in result["analysts"]:
#         print(persona.persona)
        
#     print("\nCompetitor Analysis:")
#     for analysis in result["competitor_analysis"]:
#         print(analysis)
