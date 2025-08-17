import os
import uuid
from typing import List, Optional
from kbac.loaders.matrix_script_loader import MatrixScriptLoader
from src.schemas.models import Document
from ..interfaces.document_loader_service import DocumentLoaderService
from src.config.settings import BASE_DIR


class MatrixDocumentLoaderService(DocumentLoaderService):
    """Matrix script loader service (use kbac loader)"""

    def __init__(
        self, 
        script_path: Optional[str] = None,
    ):
        if script_path is None:
            script_path = os.path.join(
                BASE_DIR,
                "resources/movie-scripts/the-matrix-1999.pdf"
            )
        
        self.script_path = script_path
        self.loader = MatrixScriptLoader(source_path=self.script_path)

    def load_documents(self) -> List[Document]:
        """Load Matrix script documents"""
        base_documents = self.loader.load()

        documents = []
        for base_doc in base_documents:
            doc = Document(
                id=str(uuid.uuid4()),
                page_content=base_doc.text,
                metadata={
                    **base_doc.metadata,
                }
            )
            documents.append(doc)
            
        return documents
    