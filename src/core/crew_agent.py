from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from crewai import Agent, Crew, Task, Process


class CrewAgent(ABC):
    """
    Abstract base class for crew-based agents.
    This class provides a standard interface for creating and running crew agents.
    """
    
    def __init__(self):
        """Initialize the agent with a default LLM."""
        pass
    


    def append_event_callback(self, task_output: Any) -> None:
        """Callback function for task events."""
        print("##########################")
        print("### Callback called: ", task_output.description)
        print("### Callback output: ", task_output.raw)
        print("##########################")

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> str:
        """
        Run the crew with the given input parameters.
        Each agent must implement this to handle its specific crew creation and execution.
        
        Args:
            input_data: Dictionary containing input parameters specific to the crew agent
            
        Returns:
            The results from running the crew
        """
        pass