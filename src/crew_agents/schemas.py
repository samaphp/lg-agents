from typing import List
from pydantic import BaseModel

class CityInfo(BaseModel):
    city: str
    state: str
    price_range: str
    why_it_matches: str
    short_term_rental_info: str

class VacationHomes(BaseModel):
    address: str
    price: str
    link: str
    why_it_matches: str

class CandidateCities(BaseModel):
    cities: List[CityInfo]

class HomeMatches(BaseModel):
    homes: List[VacationHomes] 