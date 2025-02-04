# https://github.com/bhancockio/nextjs-crewai-basic-tutorial/tree/main
# https://docs.crewai.com/concepts/langchain-tools
# https://docs.crewai.com/concepts/tools#available-crewai-tools


from crewai import Agent, Crew, Task, Process

from typing import List
from pydantic import BaseModel

from agents.llmtools import get_llm
from crew_agents.tools.websearch import  ScrapeWebTool, WebSearchTool, HomeFinderTool
from crew_agents.schemas import CityInfo, VacationHomes, CandidateCities, HomeMatches


class VacationHouseAgent():
     
    def __init__(self):
        self.llm = get_llm()
        self.web_search_tool = WebSearchTool()
        self.scrape_web_tool = ScrapeWebTool()
        self.home_finder_tool = HomeFinderTool()

    def append_event_callback(self, task_output):
        print("### Callback called: ", task_output.description)
        #append_event(self.job_id, task_output.exported_output)

    ### AGENTS ###
    def city_reasearcher(self) -> Agent:
        return Agent(
            role="City Researcher",
            goal=f"""
                You are hired to find the towns that best match the user's criteria for vacation houses.
                Given the request from the user find the best potential cities or towns that meet the user's criteria.
                You are not looking for a specific house, but a town that has the best match for the user's criteria.
                Try to find 2 cities that most closely match the user's query.
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
                For the cities requested and details provided use Redfin.com to find a few example vacation homes (not condos).
                Search each city seperately and only once.  Return 3 results.
                """,
            backstory="""As a Real Estate Agent, you are responsible for finding the best vacation homes for the user.
                You will use Redfin to find the best homes for the user.""",
            llm=self.llm,
            tools=[self.home_finder_tool,self.web_search_tool, self.scrape_web_tool],
            verbose=True,
            allow_delegation=False
        )
    
    ### TASKS ###
    def find_candidate_cities_task(self, agent: Agent, query: str) -> Task:
        return Task(
            description=f"""
                Search the web for cities that match the user's query for vacation houses: {query}
                Try to use multiple sources to find the cities that match.
                For each city research if there are short term rental restrictions and describe them briefly.
                Double check that the criteria in the query is met and the best fit cities are returned.
                """,
            expected_output="""
                A JSON array of cities that match the user's query for vacation houses.
                For each city include:
                - Name and location
                - Why it matches the criteria
                - Approximate price ranges for vacation homes
                - Short term rental restrictions (if any)
                """,
            output_json_schema=CandidateCities.model_json_schema(),
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool]
        )
    
    def find_vacation_homes_task(self, agent: Agent, query: str, task: Task) -> Task:
        return Task(
            description=f"""
                Use redfin.com to find a few example vacation homes (not condos) in the cities mentioned in the context that match the user's query: {query}
                Perform each city search separately and only once.  Return 3 results.
                If you cannot return results for a given city, return an empty list.
                If a price is given try to find houses near that price, not exactly that price (but also not too far away).
                """,
            expected_output="""
                A JSON array of vacation homes that match the user's query.
                For each home include:
                - Name and location
                - Why it matches the criteria
                - Price
                - Link to the home
                """,
            output_json_schema=HomeMatches.model_json_schema(),
            context=[task],
            agent=agent,
            callback=self.append_event_callback,
            tools=[self.web_search_tool, self.scrape_web_tool, self.home_finder_tool]
        )
    
    def summarize_task(self, agent: Agent, query: str, city_task: Task, homes_task: Task) -> Task:
        return Task(
            description=f"""
                Analyze the results from the city research and home search tasks to create a summary for the user.
                Original query: {query}
                
                Focus on:
                - How well the found cities match the user's requirements
                - The range and suitability of vacation homes found
                - Any notable patterns or insights across the results
                """,
            expected_output="""
                A clear, concise summary of the vacation home search results that helps the user understand:
                - Which cities were identified and why they're good matches
                - The types and prices of homes found with links to the homes
                - Any recommendations or notable observations
                """,
            context=[city_task, homes_task],
            agent=agent,
            callback=self.append_event_callback
        )
    
    def create_crew(self, query: str) -> Crew:
        city_researcher = self.city_reasearcher()
        city_research_task = self.find_candidate_cities_task(city_researcher, query)
        real_estate_agent = self.real_estate_agent()
        real_estate_task = self.find_vacation_homes_task(real_estate_agent, query, city_research_task)
        summarize_task = self.summarize_task(city_researcher, query, city_research_task, real_estate_task)
        crew = Crew(
            agents=[city_researcher, real_estate_agent],
            tasks=[city_research_task, real_estate_task, summarize_task],
            verbose=True,
            process=Process.sequential
        )
        return crew
    
    def run_crew(self, query: str) -> str:
        crew = self.create_crew(query)
        try:
            results = crew.kickoff()
            print("Crew Results: ", results)
            return results
        except Exception as e:
            print(f"Crew Error: {e}")
            return f"Error running crew: {str(e)}"
