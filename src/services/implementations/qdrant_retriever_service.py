import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.schemas.models import Document, RetrievalResult
from ..interfaces.retriever_service import RetrieverService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantRetrieverService(RetrieverService):
    """Qdrant-based vector retriever with OpenAI embeddings"""
    
    def __init__(
        self, 
        config
    ):
        self.collection_name = config.collection_name
        self.embedding_dims = config.embedding_dims
        self.use_memory = config.qdrant_use_memory
        self.qdrant_host = config.qdrant_host
        self.qdrant_port = config.qdrant_port
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            dimensions=config.embedding_dims
        )
        
        # Initialize Qdrant client
        if self.use_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        if self._collection_exists():
            self._init_vector_store()
        else:
            self.vectorstore = None

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in Qdrant"""
        if not documents:
            return

        self._create_collection()

        if self.vectorstore is None:
            self._init_vector_store()
        
        self.vectorstore.add_documents(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        if not self.is_initialized():
            raise ValueError("No documents have been indexed yet. Call index_documents first.")
        
        search_results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # Get RetrievalResults
        results = []
        for i, (doc, score) in enumerate(search_results):
            result = RetrievalResult(
                document=Document(
                    id=doc.metadata.get("_id", f"result-{i}"),
                    page_content=doc.page_content,
                    metadata={k: v for k, v in doc.metadata.items() if k != "_id"}
                ),
                score=score,
                rank=i + 1
            )
            results.append(result)
        
        return results
    
    def is_initialized(self) -> bool:
        """Check if the collection has documents"""
        if not self._collection_exists():
            return False
            
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count > 0
    
                
    def _collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant"""
        return self.client.collection_exists(self.collection_name)


    def _init_vector_store(self) -> None:
        """Initialize the vector store for document retrieval"""
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )    
    
    def _create_collection(self) -> None:
        """Create the collection in Qdrant if it doesn't exist"""
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dims,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")
