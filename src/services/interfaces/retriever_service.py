from abc import ABC, abstractmethod
from typing import List
from src.schemas.models import Document, RetrievalResult

class RetrieverService(ABC):
    """Abstract base class for retriever services"""

    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for retrieval"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the collection exists in the vector store"""
        return False
