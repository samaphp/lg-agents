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

        #print(f"State in should_continue: {state}")

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
        # Initialize messages and colleges if they don't exist
        messages = state.get("messages", [])

        
        context = f""" Use your own knowledge and the available tools to search for colleges:
        
        Find the best colleges, not listed below, based on these criteria:
        - Major: {state['major']}
        - Location preference: {state['location_preference'] if state['location_preference'] else 'Any'}
        - Maximum tuition: ${state.get('max_tuition', 'Not specified')}
        - Minimum acceptance rate: {state['min_acceptance_rate']}% if specified
        - Number of colleges needed: {state['max_colleges']}
        - Sat score average near {state['sat_score']}, if provided
        - {state['search_query']}
        
        Currently found colleges(do not include these): 
        {', '.join(college.name for college in state.get('colleges', []))}
        
        First try to answer the question yourself, if you can't, then use the available tools:
        1. ask_llm_for_colleges: Best for finding college names that match the criteria
        1. search_web_for_colleges: Best for finding current information about colleges
        2. search_wikipedia_for_colleges: Best for general college information and history
        3. get_web_answer: Best for getting a direct answer from web search for a specific question about a specific college (use this to get data points like acceptance rate, tuition, ernollment, dorm percentage, sat scores, etc.
        )
        Start by searching for colleges that match these criteria.  If colleges are missing information like acceptance rate or dorm percentage, use get_web_answer to get more specific information about it like acceptance rate, tuition, dorm percentage, sat scores, etc.
        """

        #print(f"Context: {context}")
        messages = [HumanMessage(content=context)]
        
        # Get model response
        response = model.invoke([HumanMessage(content=context)])

        #print(f"Model response: {response}")

        # Initialize colleges list if it doesn't exist
        if "colleges" not in state:
            state["colleges"] = []
        if "recommendations" not in state:
            state["recommendations"] = []
        
        # Update state with new message
        # new_messages = messages + [response]
        return {**state, "messages": [response]}

    def process_tool_results(state: CollegeFinderState) -> CollegeFinderState:
        """Process tool results and extract college information."""
        print("\nProcessing search results...")
        messages = state.get("messages", [])
        print("\nAll messages in state:")

        # for i, msg in enumerate(messages):
        #      print(f"\nMessage {i+1}:")
        #      print(f"Type: {type(msg).__name__}")
        #      print(f"Content: {msg.content}")
        #      if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #          print(f"Tool calls: {msg.tool_calls}")
        # print("\n")
        
        tool_outputs = [msg for msg in messages if isinstance(msg, ToolMessage)]

        #print(f"Tool outputs: {tool_outputs}")
        
        if not tool_outputs:
            print("No new search results to process")
            return state
            
        print(f"Found {len(tool_outputs)} tool ouptputs to process")
        
        # Process all tool outputs
        new_colleges = []
        for output in tool_outputs:
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
              - Enrollment (if available)
              - Website URL
            
            Content:
            {output}
            """
            print("Calling LLM to extract colleges from tool output")
            llm = get_llm()
            structured_llm = llm.with_structured_output(CollegeList)
            response = structured_llm.invoke(prompt)
            print(f"Colleges Found: {len(response.colleges)}")
            new_colleges.extend(response.colleges)
        
        # Filter colleges based on criteria
        filtered_colleges = []
        for college in new_colleges:
            # manually remove colleges that don't match the criteria 
            # Not using for now
            filtered_colleges.append(college)
        
        # Add unique colleges
        current_colleges = state.get("colleges", [])
        existing_names = {c.name for c in current_colleges}
        unique_new_colleges = [c for c in filtered_colleges if c.name not in existing_names]
        updated_colleges = current_colleges + unique_new_colleges[:state["max_colleges"] - len(current_colleges)]
        
        # After processing colleges
        print(f"Added {len(unique_new_colleges)} new unique colleges to the list")
        print(f"Updated colleges: {len(updated_colleges)}")

        # Add a message summarizing the findings
        summary = f"Found {len(unique_new_colleges)} new colleges matching your criteria."
        
        return {**state, "colleges": updated_colleges, "messages": [AIMessage(content=summary, name="process_results")]}
    
    def gather_college_info(state: dict) -> dict:
        """Gather more information about a college."""
        # Convert dict to College model if needed
        college = state["college"] if isinstance(state["college"], College) else College(**state["college"])
        
        #print(f"Gathering more information about: {college.name}")
        # Build query based on missing fields
        query_parts = []
        if not college.tuition or college.tuition is None:
            query_parts.append("tuition cost")
        if not college.acceptance_rate or college.acceptance_rate is None:
            query_parts.append("acceptance rate")
        if not college.dorm_percentage or college.dorm_percentage is None:
            query_parts.append("percentage of students living on campus")
        if not college.sat_scores or college.sat_scores is None:
            query_parts.append("average SAT scores")
        if not college.programs or college.programs is None:
            query_parts.append(f"notable programs and majors")
        if not college.url or college.url is None:
            query_parts.append("official website url")
        if not college.enrollment or college.enrollment is None:
            query_parts.append("undergraduate enrollment")
        
        if query_parts:
            query = f"What is the {', '.join(query_parts)} for {college.name} college?"
            print(f"Query: {query}")
            answer = search_web_get_answer(query)
            #print(f"Additional info for {college.name}: {answer}")
        if query_parts and answer:
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
            - enrollment
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
            if not updated_college.enrollment and college.enrollment:
                updated_college.enrollment = college.enrollment
            if not updated_college.programs and college.programs:
                updated_college.programs = college.programs
            elif not updated_college.programs:
                updated_college.programs = []  # Ensure programs is never None
               
            #print("Updated college info", updated_college)

            updated_college.has_missing_fields = any([
                    not updated_college.tuition or updated_college.tuition is None,
                    not updated_college.acceptance_rate or updated_college.acceptance_rate is None,
                    not updated_college.dorm_percentage or updated_college.dorm_percentage is None,
                    not updated_college.sat_scores or updated_college.sat_scores is None,
                    not updated_college.programs or updated_college.programs is None,
                    not updated_college.url or updated_college.url is None,
                    not updated_college.enrollment or updated_college.enrollment is None
            ])

            # Return state with updated college and has_missing_fields flag
            return {
                "colleges": [updated_college],  # This will be properly merged due to the Annotated[List[College], operator.add]
            }
        
        # If no answer was found, return original college with empty programs list if needed
        if not college.programs:
            college.programs = []
        return {
            "colleges": [college],
        }

    def gather_all_college_data(state: CollegeFinderState):
        print("Gathering all college data...")
        # Initialize data_gathering_attempts if not present
        if "data_gathering_attempts" not in state:
            state["data_gathering_attempts"] = 0
        return [Send("gather_college_info", {"college": c}) for c in state["colleges"]]
    
    def data_gathering(state: CollegeFinderState):
        # Increment attempt counter
        print(f"Data gathering attempt {state.get('data_gathering_attempts', 0) + 1}")
        return {"data_gathering_attempts": state.get("data_gathering_attempts", 0) + 1}

    def debug_state(state: CollegeFinderState):
        print(f"## DEBUG STATE ##")
        return {}

    def should_continue_gathering(state: CollegeFinderState) -> Union[Literal["continue_gathering"], Literal["finish"]]:
        """Determine if we should continue gathering data or move to recommendations."""
        attempts = state.get("data_gathering_attempts", 0)
        has_missing_fields = any(college_state.has_missing_fields == True for college_state in state.get("colleges", []))
        
        if attempts < 3 and has_missing_fields:
            print(f"Some fields still missing after attempt {attempts}, continuing data gathering...")
            return "continue_gathering"
        else:
            if attempts >= 3:
                print("Reached maximum data gathering attempts (3), moving to recommendations...")
            else:
                print("All fields gathered successfully, moving to recommendations...")
            return "finish"

    def generate_recommendations(state: CollegeFinderState) -> CollegeFinderState:
        """Generate final recommendations based on found colleges."""
        print("\nGenerating final recommendations...")
        if not state.get("colleges"):
            return state
            
        prompt = f"""Based on these colleges and criteria, provide 5-10 specific recommendations:
        
        Student's interests:
        - Major: {state['major']}
        - Location preference: {state.get('location_preference', 'Any')}
        - Maximum tuition: ${state.get('max_tuition', 'Not specified')}
        - Minimum acceptance rate: {f"{state['min_acceptance_rate']}%" if state.get('min_acceptance_rate') else 'Not specified'}
        
        Found colleges:
        {state['colleges']}
        
        Provide specific recommendations about:
        1. Which colleges might be the best fit and why
        2. What aspects of these colleges align with the student's criteria
        3. Any additional considerations or next steps

        Order the recommendations by relevance to the student's criteria and the colleges found.

        Do not use markdown or any formatting.
        
        """
        
        llm = get_llm()
        structured_llm = llm.with_structured_output(RecommendationList)
        response = structured_llm.invoke(prompt)
        
        return {**state, "recommendations": response.recommendations, "messages": [AIMessage(content="\n".join(response.recommendations))]}

    # Create the graph
    workflow = StateGraph(CollegeFinderState,input=CollegeFinderInput)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_results", process_tool_results)
    workflow.add_node("gather_college_info", gather_college_info)
    workflow.add_node("data_gathering", data_gathering)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("debug_state", debug_state)
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "process_results")
    workflow.add_edge("process_results", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": "data_gathering"
        }
    )
    # For each college, gather more information
    workflow.add_conditional_edges(
        "data_gathering",
        gather_all_college_data,
        ["gather_college_info"]
    )
    workflow.add_edge("gather_college_info", "debug_state")
    workflow.add_conditional_edges(
        "debug_state",
        should_continue_gathering,
        {
            "continue_gathering": "data_gathering",
            "finish": "generate_recommendations"
        }
    )
    


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

