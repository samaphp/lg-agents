import logging
import uvicorn
from dotenv import load_dotenv
import signal
import sys

from core import settings

load_dotenv()

VERSION = "0.0.6"

def handle_shutdown(signum, frame):
    print("\nReceived shutdown signal. Exiting gracefully...")
    sys.exit(0)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Set default level
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # Handle termination signal
    
    print(f"RUNNING AGENT SERVICE {VERSION}")
    
    config = uvicorn.Config(
        "service:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.is_dev(),
        loop="asyncio"  # Explicitly set the event loop
    )
    
    server = uvicorn.Server(config)
    server.run()
         