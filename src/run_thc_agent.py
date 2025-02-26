from dotenv import load_dotenv
import logging
from typing import Dict, Any

from agents.privateagents.private.thc_agent.thc_findproducts_flow import THCResearchFlowAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_thc_agent() -> Dict[str, Any]:
    """
    Run the THC research flow to find legal THC product companies.
    
    Returns:
        Dict[str, Any]: Result dictionary containing status and found companies
    """
    try:
        logger.info("Starting THC research flow...")
        flow = THCResearchFlowAgent()
        result = flow.run({"existing_companies": ["Crescent Canna"]})
        logger.info(f"Flow completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running THC research flow: {e}")
        raise

if __name__ == "__main__":
    load_dotenv()
    run_thc_agent()
