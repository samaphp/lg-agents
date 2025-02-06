# https://github.com/bhancockio/nextjs-crewai-basic-tutorial/tree/main
# https://docs.crewai.com/concepts/langchain-tools
# https://docs.crewai.com/concepts/tools#available-crewai-tools
# https://crawl4ai.com/mkdocs/
# https://www.youtube.com/watch?v=Osl4NgAXvRk
# https://python.langchain.com/docs/integrations/providers/groq/#chat-models


from crewai import Agent, Crew, Process, Task

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


from agents.llmtools import get_llm
from core.crew_agent import CrewAgent
from crew_agents.tools.distancetool import DistanceCalculatorTool
from crew_agents.tools.websearch import ScrapeWebTool, WebSearchTool, HomeFinderTool
from crew_agents.schemas import CityInfo, ResultSummary, VacationHomes, CandidateCities, HomeMatches


CITY_LIMIT = 2
HOME_LIMIT = 2

class VacationHouseAgent(CrewAgent):
     
    def __init__(self):
        super().__init__()
        self.llm = get_llm()
        self.web_search_tool = WebSearchTool()
        self.scrape_web_tool = ScrapeWebTool()
        self.home_finder_tool = HomeFinderTool()
        self.distance_tool = DistanceCalculatorTool()
        self.status_callback = None

    def set_status_callback(self, callback):
        """Set the callback function for status updates."""
        self.status_callback = callback

    def append_event_callback(self, event: Any) -> None:
        """Callback for task events that updates status via the status callback if set."""
        if self.status_callback:
            # Create a status update with timestamp and event info
            update = {
                "timestamp": datetime.utcnow().isoformat(),
                "description": event.description if hasattr(event, 'description') else str(event),
                "output": event.raw if hasattr(event, 'raw') else str(event)
            }
            self.status_callback(update)

    def create_agents(self) -> List[Agent]:
        """Create and return the list of agents needed for the vacation house search."""
        return [
            self.city_researcher(),
            self.real_estate_agent(),
            self.local_expert()
        ]

    def create_tasks(self, query: str) -> List[Task]:
        """Create and return the list of tasks for the vacation house search."""
        city_researcher = self.city_researcher()
        real_estate_agent = self.real_estate_agent()
        local_expert = self.local_expert()

        city_research_task = self.find_candidate_cities_task(city_researcher, query)
        real_estate_task = self.find_vacation_homes_task(real_estate_agent, query, city_research_task)
        verify_listings_task = self.verify_listings(real_estate_agent, real_estate_task)
        local_business_task = self.find_local_businesses_task(local_expert, verify_listings_task)
        summarize_task = self.summarize_task(city_researcher, query, [city_research_task, verify_listings_task, local_business_task])

        return [
            city_research_task,
            real_estate_task,
            verify_listings_task,
            local_business_task,
            summarize_task
        ]

    def city_researcher(self) -> Agent:
        return Agent(
            role="City Researcher",
            goal=f"""
                You are hired to find the towns that best match the user's criteria for vacation houses.
                Given the request from the user find the best potential cities or towns that meet the user's criteria.
                You are not looking for a specific house, but a town that has the best match for the user's criteria.
                Focus on the location and price restrictions.
                For each city research if there are short term rental restrictions and describe them briefly.
                """,
            backstory="""As a City Researcher, you are responsible for aggregating all the researched information
                into a list.""",
            llm=self.llm,
            tools=[self.web_search_tool, self.scrape_web_tool],
            verbose=True,
            allow_delegation=False
        )
    
    def real_estate_agent(self) -> Agent:
        return Agent(
            role="Real Estate Agent",
            goal=f"""
                Find the best vacation homes for the user based on their input.
                Look for single family homes within budget and with the most rooms and bathrooms.
                """,
            backstory="""As a Real Estate Agent, you are responsible for finding the best vacation homes for the user.
                You will use the web tools to find the best homes for the user.""",
            llm=self.llm,
            tools=[self.web_search_tool, self.scrape_web_tool],
            verbose=True,
            allow_delegation=False
        )
    
    def local_expert(self) -> Agent:
        return Agent(
            role="Local Expert Agent",
            goal=f"""
               Find the best local businesses for the user based on the location of the home they are interested in.
               Focus on bars and restaurants and coffee shops.
               Also try to summarize how walkable the area is.
                """,
            backstory="""You are local expert trying to advice a potential buyer as to how the neighborhood is.""",
            llm=self.llm,
            tools=[self.web_search_tool, self.scrape_web_tool],
            verbose=True,
            allow_delegation=False
        )
    
    def find_candidate_cities_task(self, agent: Agent, query: str) -> Task:
        return Task(
            description=f"""
                Compile a list of cities that match the user's query for vacation houses: {query}
                Try to use multiple sources to find the cities that match.
                For each city research if there are short term rental restrictions and describe them briefly.
                Double check that the criteria in the query is met and the best fit cities are returned.
                Use both your knowledge and the web search tool to find and verify cities.
                Return at most {CITY_LIMIT} cities (or just 1 if an exact city was specified in the query).
                """,
            expected_output="""
                A JSON array of cities that match the user's query for vacation houses.
                For each city include:
                - Name and location
                - Why it matches the criteria
                - Approximate price ranges for vacation homes
                - Short term rental restrictions (if any)
                """,
            output_json=CandidateCities,
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool]
        )
    
    def find_vacation_homes_task(self, agent: Agent, query: str, task: Task) -> Task:
        return Task(
            description=f"""
                Use the web to find a few example vacation homes (not condos) in the cities mentioned in the context that match the user's query: {query}
                Perform each city search separately and only once.  Return {HOME_LIMIT} results per city.
                When searching use filters to narrow down the results.
                If you cannot return results for a given city, return an empty list.
                If a price is given try to find houses near that price, not exactly that price.
                Avoid zillow.com links, use sites like redfin.com or realtor.com.  
                """,
            expected_output="""
                A JSON array of homes that match the user's query.
                For each home include:
                - Name and location
                - Why it matches the criteria
                - Price
                - Link to the home
                """,
            output_json=CandidateCities,
            context=[task],
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool]
        )
    
    def verify_listings(self, agent: Agent, homes_task: Task) -> Task:
        return Task(
            description=f"""
                Verify the listings found in the homes_task are within the user's budget and have correct links.
                Verify links by opening the page and verifying the link is to the same address.  If it is not search for the address and try to find the correct link.
                The link should be to a for sale page for the property.
                Remove any listings using zillow.com links. Try to make links direct to the listing not a search results page if possible.
                """,
            expected_output="""
                An updated JSON array of candidate cities with the homes that match the user's query and have correct links.
                """,
            output_json=CandidateCities,
            context=[homes_task],
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool]
        )
    
    def find_local_businesses_task(self, agent: Agent, listings_task: Task) -> Task:
        return Task(
            description=f"""
                Find the best local businesses for the user based on the location of the homes they are interested in.
                For each home find the best bars and restaurants and coffee shops in the shortest distance to the homes address.
                Collect the full postal address of the business for distance calculations.
                """,
            expected_output="""
                Updated JSON array of ccities with the business information added to each home.  The business information should include:
                - Name
                - Postal address
                - Type of business
                - Distance from home specified in miles
                """,
            output_json=CandidateCities,
            context=[listings_task],
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool, self.distance_tool]
        )
    
    def summarize_task(self, agent: Agent, query: str, tasks: List[Task]) -> Task:
        return Task(
            description=f"""
                Analyze the results from the context to create a summary for the user based on the query: {query}
   
                Focus on:
                - How well the found cities match the user's requirements
                - The range and suitability of vacation homes found
                - Any notable patterns or insights across the results
                - The local businesses found for each house
                """,
            expected_output="""
                A JSON object following the supplied schema:
                - summary: A clear, concise summary of the vacation home search results that helps the user understand:
                - candidate_cities: Which cities were identified and why they're good matches
                - home_matches: The types and prices of homes found with links to the homes
                - With the home_matches list The local businesses found for each house including their distance if available
                """,
            context=tasks,
            output_json=ResultSummary,
            agent=agent,
            callback=self.append_event_callback
        )

    def run(self, input_data: Dict[str, Any]) -> str:
        """
        Run the vacation house search crew with the given input parameters.
        
        Args:
            input_data: Dictionary containing vacation house search parameters
            
        Returns:
            The results from running the crew
        """
        # Process input into a natural language query
        query = input_data.get("search_query", input_data.get("messages", ""))
        if isinstance(query, list):
            query = " ".join(query)
        
        # Create agents and tasks
        agents = self.create_agents()
        tasks = self.create_tasks(query)
        
        # Create and run the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        try:
            if self.status_callback:
                self.status_callback({
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Starting vacation house search",
                    "output": f"Initialized agent with query: {query}"
                })
            
            results = crew.kickoff()
            # print(f"###Results: {results}")
            if self.status_callback:
                self.status_callback({
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Completed vacation house search",
                    "output": results
                })
            
            return results
        except Exception as e:
            error_msg = f"Error running crew: {str(e)}"
            if self.status_callback:
                self.status_callback({
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Error in vacation house search",
                    "output": error_msg
                })
            return error_msg