from dotenv import load_dotenv
from crew_agents.vacation_house_agent import VacationHouseAgent

def run_vacation_house_search(query: str) -> str:
    """
    Run the vacation house search agent with the given query.
    
    Args:
        query: Search query string with vacation house requirements
        
    Returns:
        str: Results from the crew search
    """
    agent = VacationHouseAgent()
    results = agent.run_crew(query)
    return results

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    search_query = "Looking for a vacation houses in a town similar to Key West but in North Carolina or South Carolina and a price under 850,000"
    results = run_vacation_house_search(search_query)
    print(f"Search Results: {results}")


