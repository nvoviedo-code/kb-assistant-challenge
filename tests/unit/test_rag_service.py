import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.rag_service import RAGService
from src.schemas.models import Document, QueryResult, RetrievalResult


class TestRAGService:
    """Test class for RAGService"""
    
    @pytest.fixture
    def sample_documents(self):
        """Create a list of sample documents for testing"""
        return [
            Document(
                id="doc1",
                page_content="This is a sample document for testing",
                metadata={"source": "test"}
            ),
            Document(
                id="doc2",
                page_content="Another document with different content",
                metadata={"source": "test"}
            )
        ]

    @pytest.fixture
    def mock_loader(self, sample_documents):
        """Create a mock document loader service"""
        loader = MagicMock()
        loader.load_documents.return_value = sample_documents
        return loader

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever service"""
        retriever = MagicMock()
        return retriever

    @pytest.fixture
    def mock_retriever_with_docs(self, sample_documents):
        """Create a mock retriever that returns sample documents"""
        retriever = MagicMock()
        retrieval_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.95,
                rank=1
            ),
            RetrievalResult(
                document=sample_documents[1],
                score=0.85,
                rank=2
            )
        ]
        retriever.retrieve.return_value = retrieval_results
        return retriever

    @pytest.fixture
    def mock_generator(self):
        """Create a mock generator service"""
        generator = MagicMock()
        # Setup async mock for generate_response
        generator.generate_response = AsyncMock()
        generator.generate_response.return_value = QueryResult(
            query="test query",
            answer="This is a test response",
            confidence=0.92,
            sources_used=["doc1", "doc2"]
        )
        return generator

    def test_init(self, mock_loader, mock_retriever, mock_generator):
        """Test RAGService initialization"""
        service = RAGService(
            loader=mock_loader,
            retriever=mock_retriever,
            generator=mock_generator
        )
        
        # Verify service attributes
        assert service.loader == mock_loader
        assert service.retriever == mock_retriever
        assert service.generator == mock_generator

    def test_index_already_initialized(self, mock_loader, mock_retriever, mock_generator):
        """Test index method when collection is already initialized"""
        # Configure the retriever to report it's initialized
        mock_retriever.is_initialized.return_value = True
        
        service = RAGService(
            loader=mock_loader,
            retriever=mock_retriever,
            generator=mock_generator
        )
        
        # Call index method
        service.index()
        
        # Verify loader and indexing methods were not called
        mock_loader.load_documents.assert_not_called()
        mock_retriever.index_documents.assert_not_called()

    def test_index_not_initialized(self, mock_loader, mock_retriever, mock_generator, sample_documents):
        """Test index method when collection is not initialized"""
        # Configure the retriever to report it's not initialized
        mock_retriever.is_initialized.return_value = False
        
        service = RAGService(
            loader=mock_loader,
            retriever=mock_retriever,
            generator=mock_generator
        )
        
        # Call index method
        service.index()
        
        # Verify loader and indexing methods were called
        mock_loader.load_documents.assert_called_once()
        mock_retriever.index_documents.assert_called_once_with(sample_documents)

    @pytest.mark.asyncio
    async def test_query_without_documents(
        self, mock_loader, mock_retriever_with_docs, mock_generator, sample_documents
    ):
        """Test query method without attaching documents"""
        service = RAGService(
            loader=mock_loader,
            retriever=mock_retriever_with_docs,
            generator=mock_generator
        )
        
        # Call query method
        result = await service.query("test query", top_k=5, attach_documents=False)
        
        # Verify retriever was called
        mock_retriever_with_docs.retrieve.assert_called_once_with("test query", top_k=5)
        
        # Verify generator was called with the retrieved documents
        mock_generator.generate_response.assert_awaited_once_with(
            "test query", 
            [doc.document for doc in mock_retriever_with_docs.retrieve.return_value]
        )
        
        # Verify result
        assert result.query == "test query"
        assert result.answer == "This is a test response"
        assert result.confidence == 0.92
        assert len(result.retrieved_documents) == 0  # No documents attached

    @pytest.mark.asyncio
    async def test_query_with_documents(
        self, mock_loader, mock_retriever_with_docs, mock_generator, sample_documents
    ):
        """Test query method with documents attached"""
        service = RAGService(
            loader=mock_loader,
            retriever=mock_retriever_with_docs,
            generator=mock_generator
        )
        
        # Call query method with attach_documents=True
        result = await service.query("test query", top_k=5, attach_documents=True)
        
        # Verify retriever was called
        mock_retriever_with_docs.retrieve.assert_called_once_with("test query", top_k=5)
        
        # Verify generator was called with the retrieved documents
        mock_generator.generate_response.assert_awaited_once_with(
            "test query", 
            [doc.document for doc in mock_retriever_with_docs.retrieve.return_value]
        )
        
        # Verify result
        assert result.query == "test query"
        assert result.answer == "This is a test response"
        assert result.confidence == 0.92
        assert result.retrieved_documents is not None  # Documents are attached
        assert len(result.retrieved_documents) == 2
