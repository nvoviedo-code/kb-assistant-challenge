from abc import ABC, abstractmethod
from typing import List
from src.schemas.models import Document


class DocumentLoaderService(ABC):
    """Abstract base class for document loader services"""

    @abstractmethod
    def load_documents(self) -> List[Document]:
        """Load documents from source"""
        pass
