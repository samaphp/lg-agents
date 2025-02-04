import requests
from langchain_core.tools import BaseTool, tool
from typing import Tuple
from pydantic import BaseModel, Field

class AddressInput(BaseModel):
    address1: str = Field(..., description="First address to calculate distance from")
    address2: str = Field(..., description="Second address to calculate distance to")

def get_coordinates(address: str) -> Tuple[float, float]:
    """Get latitude and longitude for an address using Nominatim API"""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "DistanceCalculator/1.0"
    }
    
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    
    results = response.json()
    if not results:
        raise ValueError(f"Could not find coordinates for address: {address}")
        
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    return lat, lon

def calculate_distance(address1: str, address2: str) -> str:
    """Calculate the distance between two addresses.
    
    Uses the Nominatim API to geocode addresses and calculates the distance
    between them using the Haversine formula.
    
    Args:
        address1: First address to calculate distance from
        address2: Second address to calculate distance to
        
    Returns:
        str: Distance between the addresses in miles
    """
    from math import radians, sin, cos, sqrt, atan2
    
    try:
        lat1, lon1 = get_coordinates(address1)
        lat2, lon2 = get_coordinates(address2)
        
        R = 3959.87433  # Earth's radius in miles

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return f"{distance:.2f} miles"
        
    except Exception as e:
        raise ValueError(f"Error calculating distance: {str(e)}")
