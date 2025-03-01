from dotenv import load_dotenv
import logging
from typing import Dict, Any
import argparse

from agents.privateagents.private.thc_agent.thc_findproducts_flow import THCResearchFlowAgent

from agents.privateagents.private.thc_agent.thc_findproducts_flow import VALID_ACTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_thc_agent(action: str = "find_company_details") -> Dict[str, Any]:
    """
    Run the THC research flow to find legal THC product companies.
    
    Args:
        action: The action to perform in the THC research flow
        
    Returns:
        Dict[str, Any]: Result dictionary containing status and found companies
    """
    try:
        logger.info(f"******\nStarting THC research flow with action: {action}...\n******")
        flow = THCResearchFlowAgent()
        result = flow.run({"action": action})
        logger.info(f"Flow completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running THC research flow: {e}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run THC research flow agent")
    parser.add_argument(
        "--action", 
        type=str, 
        default="find_company_details",
        help="Action to perform in the THC research flow"
    )
    args = parser.parse_args()
    if args.action not in VALID_ACTIONS:
        parser.error(f"Invalid action: {args.action}. Must be one of: {', '.join(VALID_ACTIONS)}")
    return args

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    run_thc_agent(args.action)
