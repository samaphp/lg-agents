# https://github.com/bhancockio/nextjs-crewai-basic-tutorial/tree/main
# https://docs.crewai.com/concepts/langchain-tools
# https://docs.crewai.com/concepts/tools#available-crewai-tools


from crewai import Agent, Crew, Task, Process


from agents.llmtools import get_llm
from crew_agents.tools.websearch import ScrapeWebTool, WebSearchTool


class VacationHouseAgent():
     
    def __init__(self):
        self.llm = get_llm()
        self.web_search_tool = WebSearchTool()
        self.scrape_web_tool = ScrapeWebTool()

    ### AGENTS ###
    def city_reasearcher(self) -> Agent:
        return Agent(
            role="City Researcher",
            goal=f"""
                You are hired to find the best vacation house for the user.  
                Given the request from the user find the best potential cities or towns that meet the user's criteria.
                Focus on the location and price restrictions.
                """,
            backstory="""As a City Researcher, you are responsible for aggregating all the researched information
                into a list.""",
            llm=self.llm,
            tools=[self.web_search_tool, self.scrape_web_tool],
            verbose=True,
            allow_delegation=True
        )
    
    ### TASKS ###
    def find_candidate_cities_task(self, agent: Agent, query: str) -> Task:
        return Task(
            description=f"""
                Search the web for cities that match the user's query for vacation houses: {query}
                Try to use multiple sources to find the cities that match.
                """,
            expected_output="""
                A list of cities that match the user's query for vacation houses.
                For each city include:
                - Name and location
                - Why it matches the criteria
                - Approximate price ranges for vacation homes
                """,
            tools=[self.web_search_tool],
            agent=agent
        )

    def create_crew(self, query: str) -> Crew:
        city_researcher = self.city_reasearcher()
        city_research_task = self.find_candidate_cities_task(city_researcher, query)

        crew = Crew(
            agents=[city_researcher],
            tasks=[city_research_task],
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
