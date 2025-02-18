from dotenv import load_dotenv
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.privateagents.private.bargpt_agent.bargpt_trending_flow import BarGPTTrendingPostFlow

def run_trending_flow() -> Dict[str, Any]:
    """
    Run the BarGPT trending post flow to generate and publish trending cocktail content.
    
    Returns:
        Dict[str, Any]: Result dictionary containing status, publish URL and selected topic
    """
    try:
        logger.info("Starting BarGPT trending post flow...")
        flow = BarGPTTrendingPostFlow()
        result = flow.run({})
        logger.info(f"Flow completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error running trending flow: {e}")
        raise

if __name__ == "__main__":
    load_dotenv()
    run_trending_flow()
