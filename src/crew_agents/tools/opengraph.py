from typing import Dict, Optional, List, Type
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class OpenGraphData(BaseModel):
    """Model for Open Graph data extracted from a webpage."""
    title: Optional[str] = Field(None, description="The title of the webpage")
    description: Optional[str] = Field(None, description="The description of the webpage")
    url: Optional[str] = Field(None, description="The canonical URL of the webpage")
    image: Optional[str] = Field(None, description="The URL of the image representing the webpage")
    site_name: Optional[str] = Field(None, description="The name of the website")
    type: Optional[str] = Field(None, description="The type of content (e.g., website, article)")
    locale: Optional[str] = Field(None, description="The locale of the content")
    additional_properties: Dict[str, str] = Field(default_factory=dict, description="Additional Open Graph properties")

class OpenGraphToolSchema(BaseModel):
    """Input for OpenGraphTool."""
    url: str = Field(..., description="The required URL to extract Open Graph metadata from")

class OpenGraphTool(BaseTool):
    """Tool for extracting Open Graph metadata from a URL."""
    name: str = "Open Graph Extractor"
    description: str = "Extract Open Graph metadata from a URL to get title, description, images, and other metadata. Especially useful for getting the primay image for a page."
    args_schema: Type[BaseModel] = OpenGraphToolSchema
    
    def _run(self, url: str) -> str:
        """
        Extract Open Graph metadata from the provided URL.
        
        Args:
            url: The URL to extract Open Graph metadata from
            
        Returns:
            A string representation of the extracted Open Graph data
        """
        try:
            # Send a GET request to the URL
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract Open Graph metadata
            og_data = OpenGraphData()
            
            # Process all meta tags with property attribute
            for meta in soup.find_all("meta"):
                if meta.get("property") and meta.get("property").startswith("og:"):
                    property_name = meta.get("property")[3:]  # Remove 'og:' prefix
                    content = meta.get("content")
                    
                    # Set known properties directly
                    if property_name == "title":
                        og_data.title = content
                    elif property_name == "description":
                        og_data.description = content
                    elif property_name == "url":
                        og_data.url = content
                    elif property_name == "image":
                        og_data.image = content
                    elif property_name == "site_name":
                        og_data.site_name = content
                    elif property_name == "type":
                        og_data.type = content
                    elif property_name == "locale":
                        og_data.locale = content
                    else:
                        # Store additional properties
                        og_data.additional_properties[property_name] = content
            
            # If no Open Graph title is found, try to use the HTML title
            if not og_data.title and soup.title:
                og_data.title = soup.title.string
                
            # If no Open Graph URL is found, use the provided URL
            if not og_data.url:
                og_data.url = url
                
            return og_data.model_dump_json(indent=2)
            
        except Exception as e:
            return f"Error extracting Open Graph data: {str(e)}"
