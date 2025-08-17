
from abc import abstractmethod
from typing import List
from src.schemas.models import Document

class GeneratorService:
    """Abstract base class for response generator services"""

    @abstractmethod
    async def generate_response(self, query: str, context: List[Document]) -> str:
        """Generate a response for a query using the provided context"""
        pass
