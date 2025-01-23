from typing import Annotated, Sequence, TypeVar, List, Union, Literal
from typing_extensions import TypedDict
from langgraph.graph import Graph, StateGraph, START, END
from agents.college_finder_agent.team_roster_schema import Player, RosterAgentInput, Team, TeamRosterState
from agents.llmtools import get_llm
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from agents.tools.searchweb import search_web_get_answer, search_web_with_query, SearchResult, scrape_web_agent
from langgraph.constants import Send


def search_web_for_roster(query: str) -> List[SearchResult]:
    """Search the web for team roster information using a search engine."""
    print(f"Searching the web for: {query}")
    results = search_web_with_query(query, max_results=3)
    print(f"Found {len(results)} results from the web")
    return results


def get_roster_url(query: str) -> str:
    """Get a direct answer about a team's roster URL."""
    print(f"Getting roster URL for: {query}")
    answer = search_web_get_answer(query)
    return answer

class PlayerState(TypedDict):
    """State for the player agent."""
    player: Player


def find_player_links(state: PlayerState) -> PlayerState:
    """Find the player links for the roster."""
    VALID_SITES = [
        "perfectgame.org",
        "ps-baseball.com",
        "prepbaseballreport.com"
    ]
    print(f"Finding player links for: {state['player'].name}")
    query = f"{state['player'].name} baseball"

    results = search_web_with_query(query,max_results=10)
    #print("\nDirect Search Results:")
    for result in results:
        if any(site.lower() in result.link.lower() for site in VALID_SITES):
            state["player"].links.append(result.link)

    return state

async def extract_player_info(state: PlayerState) -> TeamRosterState:
    """Extract the player info from the links."""
    print(f"Extracting player info from: {state}")

    class FastballVelocity(BaseModel):
        velocity: str = Field(description="The top fastball velocity of the player")

    for link in state["player"].links:
        # Only check velocity if player is a pitcher
        if state["player"].position and state["player"].position.lower() in ["p", "lhp", "rhp", "pitcher"]:
            # Extract roster information using scrape_web_agent
            velo = await scrape_web_agent(
                link,
                """From the data provided find the top fastball velocity of the player""",
                FastballVelocity
            )
            if velo.velocity:
                state["player"].velocity = velo.velocity
                break
    return {"team": {"players": [state["player"]]}}


def processPlayers(state: TeamRosterState):
    team = state["team"]
    print(f"Processing {len(team.players)} players from {team.team_name}")
    return [Send("process_player_info", {"player": p}) for p in team.players]


def create_team_roster_graph():
    """Create the team roster agent graph."""

    # Create the model node
    model = get_llm()

    def find_roster_url(state: RosterAgentInput) -> TeamRosterState:

        college_name = state.college_name
        class RosterURL(BaseModel):
            url: str = Field(description="The official roster URL for the college baseball team")


        results = search_web_with_query(f"What is the offical .edu url of the {college_name} baseball team 2024 or 2025 roster",max_results=5)
        #print("\nDirect Search Results:")
        for result in results:
            print(f"\nURL: {result.link}")
            print(f"Content: {result.content[:200]}...")

        context = "\n\n".join([f"URL: {r.link}\nContent: {r.content}" for r in results])

        llm = get_llm()
        structured_llm = llm.with_structured_output(RosterURL)
        roster_info = structured_llm.invoke(
            [f"Based on the following search results, what is the official roster URL for {college_name} baseball team? "
            "Look for .edu domains and official athletics pages. Provide your confidence level and reasoning.\n\n"
            f"{context}"],
            config={"temperature": 0.3}
        )

        print("\nExtracted Roster URL Info:")
        print(f"URL: {roster_info.url}")
        
        # Update state with new message
        return {college_name: college_name, "roster_url": roster_info.url}

    async def extract_roster(state: TeamRosterState) -> TeamRosterState:
        """Extract roster information from the URL."""

        
        if not state.get("roster_url"):
            print("No roster URL found")
            return state
        
        print("\nExtracting roster information from: ", state["roster_url"])
        college_name= state["college_name"]
        try:
            # Use scrape_web_agent to extract roster information
            roster = await scrape_web_agent(
                state["roster_url"],
                """Extract the full roster information from this webpage for the {college_name} baseball team. Include:
                - Each player's full name
                - Position
                - Handedness (L/L or R/R or L/R or R/L)
                - Height if available
                - Hometown if available
                Format as a structured team roster.""",
                Team
            )
            
            return {"team": roster}
            
        except Exception as e:
            print(f"Error extracting roster: {e}")
            return state

    def summarize_roster(state: TeamRosterState) -> TeamRosterState:
        """Summarize the roster information."""
        print(f"Summarizing roster information for: {state['team'].team_name}")
        return state
    
    # Create a subgraph for processing a players info
    player_graph = StateGraph(PlayerState)
    player_graph.add_node("find_player_links", find_player_links)
    player_graph.add_node("extract_player_info", extract_player_info)

    player_graph.add_edge("find_player_links", "extract_player_info")
    player_graph.add_edge("extract_player_info", END)
    player_graph.set_entry_point("find_player_links")

    
    
    # Create the graph
    workflow = StateGraph(TeamRosterState,input=RosterAgentInput)

    # Add nodes
    workflow.add_node("find_roster_url", find_roster_url)
    workflow.add_node("extract_roster", extract_roster)
    workflow.add_node("process_player_info", player_graph.compile())
    workflow.add_node("summarize_roster", summarize_roster)
    # Add edges
    workflow.add_edge("find_roster_url", "extract_roster")

    workflow.add_conditional_edges("extract_roster", processPlayers, ["process_player_info"])

    workflow.add_edge("process_player_info", "summarize_roster")
    workflow.add_edge("summarize_roster", END)

    # Set entry point
    workflow.set_entry_point("find_roster_url")

    graph = workflow.compile()

    # Save graph visualization to disk and display it
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open("roster_graph.png", "wb") as f:
        f.write(graph_image)

    # Compile
    return graph


team_roster_agent = create_team_roster_graph()