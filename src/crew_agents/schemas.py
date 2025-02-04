from typing import List
from pydantic import BaseModel

class CityInfo(BaseModel):
    city: str
    state: str
    price_range: str
    why_it_matches: str
    short_term_rental_info: str

class BusinessInfo(BaseModel):
    name: str
    address: str
    type: str
    distanceFromHome: str

class VacationHomes(BaseModel):
    address: str
    price: str
    link: str
    why_it_matches: str
    walk_score: str
    bars_and_restaurants: List[BusinessInfo]
    coffee_shops: List[BusinessInfo]

class CandidateCities(BaseModel):
    cities: List[CityInfo]

class HomeMatches(BaseModel):
    homes: List[VacationHomes] 