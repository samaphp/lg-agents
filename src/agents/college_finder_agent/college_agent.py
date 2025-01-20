from typing import Annotated, Sequence, TypeVar, List, Union, Literal
from typing_extensions import TypedDict
from langgraph.graph import Graph, StateGraph, START, END
from agents.llmtools import get_llm
from agents.college_finder_agent.college_agent_schema import College, CollegeFinderInput, CollegeFinderState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from IPython.display import Image, display
from agents.tools.searchweb import search_web_get_answer, search_web_with_query, use_browser, SearchResult
from agents.tools.wikisearch import search_wikipedia_with_query
from langgraph.constants import Send
from operator import add

class CollegeList(BaseModel):
    colleges: List[College]

class RecommendationList(BaseModel):
    recommendations: List[str]

# Define tools using the @tool decorator
@tool
def search_web_for_colleges(query: str) -> List[SearchResult]:
    """Search the web for college information using a search engine."""
    print(f"Searching the web for: {query}")
    results = search_web_with_query(query, max_results=5)
    print(f"Found {len(results)} results from the web")
    return [AIMessage(content=str(result)) for result in results]

@tool
def search_wikipedia_for_colleges(query: str) -> List[str]:
    """Search Wikipedia for college information."""
    print(f"Searching Wikipedia for: {query}")
    results = search_wikipedia_with_query(query, max_results=3)
    print(f"Found {len(results)} results from Wikipedia")
    #print("Wikipedia results: ", results)
    return [AIMessage(content=str(results))]

@tool
def get_web_answer(query: str) -> str:
    """Get a direct answer from web search for a specific question about colleges."""
    print(f"Getting web answer for: {query}")
    answer = search_web_get_answer(query)
    #print(f"Found answer: {answer}")
    return AIMessage(content=answer)

@tool
def ask_llm_for_colleges(query: str,exclude_colleges: str=None) -> List[College]:
    """Ask the LLM to find colleges from a text query."""
    print(f"Asking LLM to extract colleges from: {query} and exclude: {exclude_colleges}")
    llm = get_llm()
    prompt = f"""
        Find colleges (the more the better) that match this query: {query}
        Order them by relevance to the query and their prestige
        Do not include these colleges: {exclude_colleges}
    """
    response = llm.invoke(prompt)
    #print(f"LLM response: {response.content}")
    return [AIMessage(content=str(response.content))]



def create_college_finder_graph() -> CompiledStateGraph:
    # Define tools
    tools = [search_web_for_colleges, search_wikipedia_for_colleges, get_web_answer,ask_llm_for_colleges]
    
    # Create tool executor node
    tool_node = ToolNode(tools)

    # Create the model node
    model = get_llm().bind_tools(tools)

    def should_continue(state: CollegeFinderState) -> Union[Literal["continue"], Literal["end"]]:
        """Determine if we should continue running the agent."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # Add progress print
        print(f"\nFound {len(state.get('colleges', []))} colleges so far. Target: {state['max_colleges']}")
        
        if (len(state.get("colleges", [])) < state["max_colleges"] and 
            isinstance(last_message, AIMessage) and 
            last_message.tool_calls):
            print("Continuing search for more colleges...")
            return "continue"
        print("Search complete. Cleaning up data next...")
        return "end"

    def call_model(state: CollegeFinderState) -> CollegeFinderState:
        """Call the model to get the next action."""
        print("\nQuerying AI model for next action...")
        messages = state.get("messages", [])
        

        context = f""" Use your own knowledge and the available tools to search for colleges:
        
        Find colleges, not listed below, based on these criteria:
        - Major: {state['major']}
        - Location preference: {state['location_preference'] if state['location_preference'] else 'Any'}
        - Maximum tuition: ${state['max_tuition'] if state['max_tuition'] else 'Not specified'}
        - Minimum acceptance rate: {state['min_acceptance_rate']}% if specified
        - Number of colleges needed: {state['max_colleges']}
        - {state['search_query']}
        
        Currently found colleges(do not include these): 
        {', '.join(college.name for college in state.get('colleges', []))}
        
        First try to answer the question yourself, if you can't, then use the available tools:
        1. ask_llm_for_colleges: Best for finding college names that match the criteria
        1. search_web_for_colleges: Best for finding current information about colleges
        2. search_wikipedia_for_colleges: Best for general college information and history
        3. get_web_answer: Best for getting a direct answer from web search for a specific question about a specific college (use this to get data points like acceptance rate, tuition, dorm percentage, sat scores, etc.
        )
        Start by searching for colleges that match these criteria.  If colleges are missing information like acceptance rate or dorm percentage, use get_web_answer to get more specific information about it like acceptance rate, tuition, dorm percentage, sat scores, etc.
        """

        #print(f"Context: {context}")
        messages = [HumanMessage(content=context)]
        
        # Get model response
        response = model.invoke([HumanMessage(content=context)])

        #print(f"Model response: {response}")
        
        # Update state with new message
        new_messages = messages + [response]
        return {**state, "messages": new_messages}

    def process_tool_results(state: CollegeFinderState) -> CollegeFinderState:
        """Process tool results and extract college information."""
        print("\nProcessing search results...")
        messages = state.get("messages", [])
        # print("\nAll messages in state:")
        # for i, msg in enumerate(messages):
        #     print(f"\nMessage {i+1}:")
        #     print(f"Type: {type(msg).__name__}")
        #     print(f"Content: {msg.content}")
        #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #         print(f"Tool calls: {msg.tool_calls}")
        # print("\n")
        tool_outputs = [msg for msg in messages if isinstance(msg, ToolMessage)]

        #print(f"Tool outputs: {tool_outputs}")
        
        if not tool_outputs:
            print("No new search results to process")
            return state
            
        print(f"Found {len(tool_outputs)} new search results to analyze")
        
        # Process up to 10 tool outputs
        new_colleges = []
        for output in tool_outputs[:10]:
            # Extract college information from tool results
            prompt = f"""Extract college information from this content:
            - Look for colleges that offer programs in {state['major']}
            - For each college, extract:
              - Name
              - Location
              - Description
              - Acceptance rate (if available)
              - SAT scores (if available)
              - Dorm percentage (if available)
              - Tuition (if available)
              - Notable programs/majors
              - Website URL
            
            Content:
            {output}
            """
            
            llm = get_llm()
            structured_llm = llm.with_structured_output(CollegeList)
            response = structured_llm.invoke(prompt)
            new_colleges.extend(response.colleges)
        
        # Filter colleges based on criteria
        filtered_colleges = []
        for college in new_colleges:
            if state["location_preference"] and state["location_preference"].lower() not in college.location.lower():
                continue
            if state["max_tuition"] and college.tuition:
                try:
                    tuition = float(''.join(filter(str.isdigit, college.tuition)))
                    if tuition > state["max_tuition"]:
                        continue
                except ValueError:
                    pass
            if state["min_acceptance_rate"] and college.acceptance_rate:
                try:
                    rate = float(''.join(filter(str.isdigit, college.acceptance_rate)))
                    if rate < state["min_acceptance_rate"]:
                        continue
                except ValueError:
                    pass
            filtered_colleges.append(college)
        
        # Add unique colleges
        current_colleges = state.get("colleges", [])
        existing_names = {c.name for c in current_colleges}
        unique_new_colleges = [c for c in filtered_colleges if c.name not in existing_names]
        updated_colleges = current_colleges + unique_new_colleges[:state["max_colleges"] - len(current_colleges)]
        
        # Add a message summarizing the findings
        summary = f"Found {len(unique_new_colleges)} new colleges matching your criteria."
        messages.append(AIMessage(content=summary, name="process_results"))
        
        # After processing colleges
        print(f"Added {len(unique_new_colleges)} new unique colleges to the list")
        
        return {**state, "colleges": updated_colleges, "messages": messages}
    
    def gather_college_info(state: dict) -> dict:
        """Gather more information about a college."""
        # Convert dict to College model if needed
        college = state["college"] if isinstance(state["college"], College) else College(**state["college"])
        
        #print(f"Gathering more information about: {college.name}")
        # Build query based on missing fields
        query_parts = []
        if not college.tuition:
            query_parts.append("tuition cost")
        if not college.acceptance_rate:
            query_parts.append("acceptance rate") 
        if not college.dorm_percentage:
            query_parts.append("percentage of students living in dorms")
        if not college.sat_scores:
            query_parts.append("average SAT scores")
        if not college.programs:
            query_parts.append(f"notable programs and majors")
        if not college.url:
            query_parts.append("official website url")
        
        if query_parts:
            query = f"What is the {', '.join(query_parts)} for {college.name} college?"
            print(f"Query: {query}")
            answer = search_web_get_answer(query)
            #print(f"Additional info for {college.name}: {answer}")
        if answer:
            # Map the search results back to the College object using LLM
            prompt = f"""
            Based on this information about {college.name}:
            {answer}
            
            Update only the following fields if the information is present:
            - tuition
            - acceptance_rate 
            - dorm_percentage
            - sat_scores
            - programs
            - url
            
            Keep any existing values if no new information is found.
            Existing info:
            {college}
            """
            
            llm = get_llm()
            structured_llm = llm.with_structured_output(College)
            updated_college = structured_llm.invoke(
                prompt,
                config={"temperature": 0.1}
            )
            
            # Preserve existing values if new ones weren't found
            if not updated_college.tuition and college.tuition:
                updated_college.tuition = college.tuition
            if not updated_college.acceptance_rate and college.acceptance_rate:
                updated_college.acceptance_rate = college.acceptance_rate
            if not updated_college.dorm_percentage and college.dorm_percentage:
                updated_college.dorm_percentage = college.dorm_percentage
            if not updated_college.sat_scores and college.sat_scores:
                updated_college.sat_scores = college.sat_scores
            if not updated_college.url and college.url:
                updated_college.url = college.url
            if not updated_college.programs and college.programs:
                updated_college.programs = college.programs
            elif not updated_college.programs:
                updated_college.programs = []  # Ensure programs is never None
               
            #print("Updated college info", updated_college)
                
            # Instead of returning just the college, return a state update
            return {
                "colleges": [updated_college]  # This will be properly merged due to the Annotated[List[College], operator.add]
            }
        
        # If no answer was found, return original college with empty programs list if needed
        if not college.programs:
            college.programs = []
        return {
            "colleges": [college]
        }

    def gather_all_college_data(state: CollegeFinderState):
        print("Gathering all college data...")
        return [Send("gather_college_info", {"college": c}) for c in state["colleges"]]
    
    def data_cleanup(state: CollegeFinderState):
        return {}
    

    def generate_recommendations(state: CollegeFinderState) -> CollegeFinderState:
        """Generate final recommendations based on found colleges."""
        print("\nGenerating final recommendations...")
        if not state.get("colleges"):
            return state
            
        prompt = f"""Based on these colleges and criteria, provide 5-10 specific recommendations:
        
        Student's interests:
        - Major: {state['major']}
        - Location preference: {state['location_preference'] if state['location_preference'] else 'Any'}
        - Maximum tuition: ${state['max_tuition'] if state['max_tuition'] else 'Not specified'}
        - Minimum acceptance rate: {state['min_acceptance_rate']}% if specified
        
        Found colleges:
        {state['colleges']}
        
        Provide specific recommendations about:
        1. Which colleges might be the best fit and why
        2. What aspects of these colleges align with the student's criteria
        3. Any additional considerations or next steps
        """
        
        llm = get_llm()
        structured_llm = llm.with_structured_output(RecommendationList)
        response = structured_llm.invoke(prompt)
        
        # Add recommendations to messages for a complete conversation history
        messages = state.get("messages", [])
        messages.append(AIMessage(content="\n".join(response.recommendations)))
        
        return {**state, "recommendations": response.recommendations, "messages": messages}

    # Create the graph
    workflow = StateGraph(CollegeFinderState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_results", process_tool_results)
    workflow.add_node("gather_college_info", gather_college_info)
    workflow.add_node("data_cleanup", data_cleanup)
    workflow.add_node("generate_recommendations", generate_recommendations)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": "data_cleanup"
        }
    )
    workflow.add_conditional_edges(
        "data_cleanup",
        gather_all_college_data,
        ["gather_college_info"]
    )
    workflow.add_edge("gather_college_info", "generate_recommendations")
    workflow.add_edge("tools", "process_results")
    workflow.add_edge("process_results", "agent")

    workflow.add_edge("generate_recommendations", END)

    # Compile graph
    graph = workflow.compile(checkpointer=MemorySaver())
    
    # Save graph visualization to disk and display it
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open("college_finder_graph.png", "wb") as f:
        f.write(graph_image)
    
    # Optionally still display in notebook
    # display(Image(graph_image))

    return graph

# Create the agent instance
college_finder_agent = create_college_finder_graph()

# Helper function to run the graph
async def run_college_finder(
    major: str,
    location_preference: str = None,
    max_tuition: int = None,
    min_acceptance_rate: float = None,
    max_colleges: int = 3
) -> CollegeFinderState:
    
    # Initialize state
    initial_state: CollegeFinderState = {
        "major": major,
        "location_preference": location_preference,
        "max_tuition": max_tuition,
        "min_acceptance_rate": min_acceptance_rate,
        "max_colleges": max_colleges,
        "search_query": "",
        "search_results": [],
        "colleges": [],
        "recommendations": [],
        "messages": []  # Initialize empty messages list for ToolNode
    }
    
    # Run the graph
    final_state = await college_finder_agent.arun(initial_state)
    
    return final_state
