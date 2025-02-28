#!/usr/bin/env python3
"""
Example script demonstrating the use of the OpenGraph tool.
This script extracts Open Graph metadata from a URL using the OpenGraphTool.
"""

import json
from crew_agents.tools.opengraph import OpenGraphTool


def main():
    """Run the OpenGraph tool example."""
    # Create an instance of the OpenGraphTool
    og_tool = OpenGraphTool()
    
    # Example URLs to extract Open Graph data from
    urls = [
        "https://hiseltzers.com/shop/seltzers/12oz-hi-seltzers/?attribute_pa_flavor=lemon-lime"
    ]
    
    # Extract and display Open Graph data for each URL
    for url in urls:
        print(f"\n\nExtracting Open Graph data from: {url}")
        print("-" * 80)
        
        # Run the tool
        result = og_tool.run(url)
        
        # Try to parse the result as JSON for pretty printing
        try:
            parsed_result = json.loads(result)
            print(json.dumps(parsed_result, indent=2))
        except json.JSONDecodeError:
            # If not valid JSON, print as is
            print(result)


if __name__ == "__main__":
    main() 