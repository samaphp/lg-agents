from typing import Annotated, Sequence, TypeVar, List
from typing_extensions import TypedDict
from langgraph.graph import Graph, StateGraph
from agents.llmtools import get_llm
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from agents.marketing_agent.marketing_schema import Competitor, MarketingPlanState, Persona
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from agents.tools.searchweb import search_web, search_web_with_query
from agents.tools.wikisearch import search_wikipedia_with_query

class PersonaList(BaseModel):
    personas: List[Persona]

class CompetitorList(BaseModel):
    competitors: List[Competitor]

class SiteInfo(BaseModel):
    appName: str
    description: str
    keyfeatures: List[str]
    value_proposition: str


class KeywordList(BaseModel):
    keywords: List[str]

def create_marketing_graph() -> CompiledStateGraph:
    
    # Define state type
    workflow_state = TypeVar("workflow_state", bound=MarketingPlanState)

    # Create personas node
    def create_personas(state: workflow_state) -> workflow_state:
        prompt = f"""
        Create {state['max_personas']} buyer personas for {state['appName']}.
        App description: {state['appDescription']}
        Key features: {state['keyfeatures']}
        Value proposition: {state['value_proposition']}

        Focus on their key characteristics, needs, and pain points.
        
        Return a list of personas, each with a name and description.
        """
        
        llm = get_llm()
        structured_llm = llm.with_structured_output(PersonaList)
        response = structured_llm.invoke(prompt)
        
        # Take only up to max_personas
        state["personas"] = response.personas[:state['max_personas']]
        #print("ANALYSTS", state["personas"])
        return state
    
    async def search_web_for_competitors(state: workflow_state):
        results = search_web(f"Find website with a similar value proposition: {state['value_proposition']}")
        #print("SEARCH RESULTS", results)
        return {"search_results": results}
    
    async def search_web_for_competitors_by_hint(state: workflow_state):
        if state['competitor_hint']:
            results = search_web_with_query(f"Find website similar to {state['competitor_hint']}")
            #print("SEARCH RESULTS", results)
            return {"search_results": results}
        else:
            return state

    # Research competitors node using Browser Use
    async def analyze_site(state: workflow_state) -> workflow_state:
        results = []

        llm = get_llm()
        controller = Controller()

        @controller.registry.action('Done with task', param_model=SiteInfo)
        async def done(params: SiteInfo):
            print(f"[CONTROLLER] Done with task: {params.model_dump_json()}")
            result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
            return result

        browser_agent = Agent(
            task=f"""Visit website {state['appUrl']} focusing on:
             what the website is about, what it does, and what it offers.
             Visit 1 to 3 pages and extract the following information:
             - Description of the website
             - Key features of the website
             - Value proposition of the website
             - app name
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
            
        result = await browser_agent.run(max_steps=10)
        final_result = result.final_result()
        if final_result:
            parsed = SiteInfo.model_validate_json(final_result)
            state["appDescription"] = parsed.description
            state["keyfeatures"] = parsed.keyfeatures
            state["value_proposition"] = parsed.value_proposition
            state["appName"] = parsed.appName
        return state

    async def extract_keywords(state: workflow_state) -> workflow_state:
        prompt = f"""
        Thinking like a social media manager what are some keywords you would monitor for {state['appName']} whose value proposition is: {state['value_proposition']}

        The personas are: {state['personas']}
        
        Return a list of 5-10 keywords or phrases to search for to find posts to monitor and respond to.
        """
        llm = get_llm()
        structured_llm = llm.with_structured_output(KeywordList)
        response = structured_llm.invoke(prompt)
        state["keywords"] = response.keywords
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

    async def finalize_competitors(state: workflow_state) -> workflow_state:
        """
        Analyze search results to create a final list of competitors.
        Uses browser to visit each site and extract relevant information.
        """
        llm = get_llm()
        controller = Controller()
        results = []

        @controller.registry.action('Done with task', param_model=CompetitorList)
        async def done(params: CompetitorList):
            print(f"[CONTROLLER] Done with task: {params.model_dump_json()}")
            result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
            return result

        # Process each search result
        for search_result in state.get('search_results', []):
            try:
                browser_agent = Agent(
                    task=f"""Visit website {search_result.link} and look for a list of competitors to {state['appName']} or {state['competitor_hint']}.  
                    Do not visit {state['appUrl']} or {state['competitor_hint']}.
                    
                    Extract competitors to {state['appName']} or {state['competitor_hint']}:
                    - Name of the company/product
                    - Brief Description of what they do
                    - URL of the website
                    
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
                
                result = await browser_agent.run(max_steps=15)
                if result and result.final_result():
                    competitors = CompetitorList.model_validate_json(result.final_result())
                    results.append(competitors)
            except Exception as e:
                print(f"Error analyzing competitor {search_result.link}: {e}")
                continue

        # Update state with competitors
        state["competitors"] = results
        return state

    # Create the graph
    workflow = StateGraph(MarketingPlanState)

    # Add nodes
    workflow.add_node("analyze_site", analyze_site)
    workflow.add_node("create_personas", create_personas)
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("search_web_for_competitors", search_web_for_competitors)
    workflow.add_node("search_web_for_competitors_by_hint", search_web_for_competitors_by_hint)
    workflow.add_node("get_feedback", get_feedback)
    workflow.add_node("__END__", end)
    workflow.add_node("finalize_competitors", finalize_competitors)

    # Create edges
    workflow.add_edge("analyze_site", "create_personas")
    workflow.add_edge("create_personas", "extract_keywords")
    workflow.add_edge("extract_keywords", "search_web_for_competitors_by_hint")
    workflow.add_edge("search_web_for_competitors_by_hint", "finalize_competitors")
    workflow.add_edge("finalize_competitors", "__END__")
    
    # Set entry point
    workflow.set_entry_point("analyze_site")

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
