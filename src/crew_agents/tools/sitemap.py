from typing import Dict, List, Type
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SitemapData(BaseModel):
    """Model for sitemap data extracted from a sitemap XML."""
    urls: List[str] = Field(default_factory=list, description="List of URLs found in the sitemap")
    errors: List[str] = Field(default_factory=list, description="List of any errors encountered")

class SitemapToolSchema(BaseModel):
    """Input for SitemapTool."""
    url: str = Field(..., description="The required URL of the sitemap to process")

class SitemapTool(BaseTool):
    """Tool for extracting URLs from a sitemap."""
    name: str = "Sitemap URL Extractor"
    description: str = "Extract all URLs from a sitemap XML file, including crawling referenced sitemaps"
    args_schema: Type[BaseModel] = SitemapToolSchema

    def _process_sitemap(self, url: str, sitemap_data: SitemapData) -> None:
        """
        Process a sitemap URL and extract all URLs, handling nested sitemaps recursively.
        
        Args:
            url: The sitemap URL to process
            sitemap_data: The SitemapData object to update with found URLs
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xml,application/xhtml+xml,application/rss+xml,application/atom+xml,*/*;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "DNT": "1",
            }
            
            # Use a session to handle cookies and redirects properly
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()

            #print(f"Response content: {response.content}")
            
            soup = BeautifulSoup(response.content, "xml")
            
            # Handle sitemap index files
            sitemaps = soup.find_all("sitemap")
            #print(f"Sitemaps: {sitemaps}")
            if sitemaps:
                for sitemap in sitemaps:
                    loc = sitemap.find("loc")
                    if loc and loc.text:
                        self._process_sitemap(loc.text, sitemap_data)
            
            # Handle regular sitemaps
            urls = soup.find_all("url")
            for url_tag in urls:
                loc = url_tag.find("loc")
                if loc and loc.text:
                    sitemap_data.urls.append(loc.text)
                    
        except Exception as e:
            print(f"Error processing sitemap {url}: {str(e)}")
            sitemap_data.errors.append(f"Error processing sitemap {url}: {str(e)}")

    def _run(self, url: str) -> str:
        """
        Extract all URLs from the provided sitemap URL.
        
        Args:
            url: The sitemap URL to process
            
        Returns:
            A string representation of the extracted URLs and any errors
        """
        sitemap_data = SitemapData()
        self._process_sitemap(url, sitemap_data)
        result = sitemap_data.model_dump_json(indent=2)
        with open("agent_output/sitemap.json", "w") as f:
            f.write(result)
        return result