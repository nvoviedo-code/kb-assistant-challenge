import logging
from ..schemas.models import QueryResult
from .interfaces.document_loader_service import DocumentLoaderService
from .interfaces.retriever_service import RetrieverService
from .interfaces.generator_service import GeneratorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation (RAG) service for document-based question answering.
    """
    def __init__(
            self,
            loader: DocumentLoaderService,
            retriever: RetrieverService,
            generator: GeneratorService
    ):
        self.loader = loader
        self.retriever = retriever
        self.generator = generator
    
    def index(self) -> None:
        """Load and index documents if they haven't been indexed already"""
        
        # Check if collection already exists and contains documents
        if self.retriever.is_initialized():
            logger.info("Collection already exists and contains documents. Skipping indexing.")
            return
            
        logger.info("Loading documents...")
        documents = self.loader.load_documents()
        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Indexing documents...")
        self.retriever.index_documents(documents)
        logger.info("Indexing complete")

    async def query(self, question: str, top_k: int = 10, attach_documents: bool = False) -> QueryResult:
        """Query the RAG system"""
        logger.info(f"Processing query: '{question}' with top_k={top_k}")
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        retrieved_docs = [rr.document for rr in retrieval_results]
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate response using LLM with context
        result = await self.generator.generate_response(question, retrieved_docs)
        logger.info(f"Generated response with confidence: {result.confidence}")
        
        # Add retrieval information to result
        if attach_documents:
            result.retrieved_documents = retrieval_results
            logger.debug("Attached retrieval documents to result")

        return result