from agents.tools.distancetool import calculate_distance
from crewai.tools import BaseTool
from pydantic import BaseModel

class AddressInput(BaseModel):
    address1: str
    address2: str

class DistanceCalculatorTool(BaseTool):
    name: str = "Distance Calculator Tool"
    description: str = "Calculate the distance between two addresses in miles."

    def _run(self, address1: str, address2: str) -> str:
        return calculate_distance(address1, address2)
