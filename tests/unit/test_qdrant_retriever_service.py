import pytest
from unittest.mock import MagicMock, patch
import sys

# Path patching to make the mocks work correctly
with patch.dict(sys.modules, {
    'qdrant_client': MagicMock(),
    'langchain_openai': MagicMock(),
    'langchain_qdrant': MagicMock()
}):
    from src.services.implementations.qdrant_retriever_service import QdrantRetrieverService
from src.schemas.models import Document


class TestQdrantRetrieverService:
    """Test class for QdrantRetrieverService"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for the service"""
        config = MagicMock()
        config.collection_name = "test_collection"
        config.embedding_dims = 1536
        config.embedding_model = "text-embedding-3-small"
        config.qdrant_use_memory = True  # Use in-memory storage for tests
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        return config

    @pytest.fixture
    def mock_empty_collection_info(self):
        """Create a mock collection info with no points"""
        info = MagicMock()
        info.points_count = 0
        return info

    @pytest.fixture
    def mock_populated_collection_info(self):
        """Create a mock collection info with points"""
        info = MagicMock()
        info.points_count = 10
        return info

    @pytest.fixture
    def sample_documents(self):
        """Create a list of sample documents for testing"""
        # Use valid UUIDs for document IDs
        return [
            Document(
                id="12345678-1234-5678-1234-567812345678",
                page_content="This is a sample document for testing",
                metadata={"source": "test"}
            ),
            Document(
                id="87654321-4321-8765-4321-876543210000",
                page_content="Another document with different content",
                metadata={"source": "test"}
            )
        ]

    def test_init_with_memory_client(self, mock_config):
        """Test initialization with in-memory client"""
        mock_config.qdrant_use_memory = True
        
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client:
            mock_qdrant_client.return_value.collection_exists.return_value = False
            
            service = QdrantRetrieverService(mock_config)
            
            # Verify client initialization with memory
            mock_qdrant_client.assert_called_once_with(":memory:")
        assert service.vectorstore is None

    def test_init_with_remote_client(self, mock_config):
        """Test initialization with remote client"""
        mock_config.qdrant_use_memory = False
        
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client:
            mock_qdrant_client.return_value.collection_exists.return_value = False
            
            service = QdrantRetrieverService(mock_config)
            
            # Verify client initialization with host and port
            mock_qdrant_client.assert_called_once_with(
                host=mock_config.qdrant_host,
                port=mock_config.qdrant_port
            )
            assert service.vectorstore is None

    def test_init_with_existing_collection(self, mock_config):
        """Test initialization with existing collection"""
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client, \
             patch('src.services.implementations.qdrant_retriever_service.QdrantVectorStore') as mock_vector_store:
            
            # Setup mock to indicate collection exists
            mock_qdrant_client.return_value.collection_exists.return_value = True
            
            service = QdrantRetrieverService(mock_config)
            
            # Verify vector store initialization
            mock_vector_store.assert_called_once()
            assert service.vectorstore is not None

    def test_is_initialized_no_collection(self, mock_config):
        """Test is_initialized when collection doesn't exist"""
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client:
            mock_qdrant_client.return_value.collection_exists.return_value = False
            
            service = QdrantRetrieverService(mock_config)
            
            assert not service.is_initialized()

    def test_is_initialized_populated_collection(
        self, mock_config, mock_populated_collection_info
    ):
        """Test is_initialized with populated collection"""
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client:
            mock_qdrant_client.return_value.collection_exists.return_value = True
            mock_qdrant_client.return_value.get_collection.return_value = mock_populated_collection_info
            
            # Create service with initialization bypassed
            service = QdrantRetrieverService.__new__(QdrantRetrieverService)
            service.client = mock_qdrant_client.return_value
            service.collection_name = mock_config.collection_name
            
            assert service.is_initialized()

    def test_index_documents_new_collection(
        self, mock_config, sample_documents
    ):
        """Test indexing documents in a new collection"""
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client, \
             patch('src.services.implementations.qdrant_retriever_service.QdrantVectorStore') as mock_vector_store:
            
            # Setup mocks
            mock_qdrant_client.return_value.collection_exists.return_value = False
            mock_vector_store_instance = MagicMock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            service = QdrantRetrieverService(mock_config)
            
            # Reset mock to track calls after initialization
            mock_qdrant_client.reset_mock()
            
            # Setup the vectorstore mock
            service._init_vector_store = MagicMock()
            service.vectorstore = mock_vector_store_instance
            
            # Index documents
            service.index_documents(sample_documents)
            
            # Verify collection was created
            mock_qdrant_client.return_value.create_collection.assert_called_once()
            
            # Verify documents were added
            mock_vector_store_instance.add_documents.assert_called_once_with(sample_documents)
  
    def test_retrieve_success(
        self, mock_config
    ):
        """Test successful retrieval"""
        with patch('src.services.implementations.qdrant_retriever_service.QdrantClient') as mock_qdrant_client, \
             patch('src.services.implementations.qdrant_retriever_service.QdrantVectorStore') as mock_vector_store:
            
            # Setup mocks
            mock_qdrant_client.return_value.collection_exists.return_value = True
            mock_qdrant_client.return_value.get_collection.return_value.points_count = 10
            
            # Create a mock document for the search result
            mock_doc = MagicMock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"_id": "doc1", "source": "test"}
            
            # Setup the search results
            mock_vector_store_instance = MagicMock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
            
            service = QdrantRetrieverService(mock_config)
            
            # Mock initialization status
            service.is_initialized = MagicMock(return_value=True)
            service.vectorstore = mock_vector_store_instance
            
            # Perform retrieval
            results = service.retrieve("test query", top_k=3)
            
            # Verify the search was performed
            mock_vector_store_instance.similarity_search_with_score.assert_called_once_with(
                query="test query",
                k=3
            )
            
            # Verify the results
            assert len(results) == 1
            assert results[0].score == 0.95
            assert results[0].rank == 1
            assert results[0].document.id == "doc1"
            assert results[0].document.page_content == "Test content"
