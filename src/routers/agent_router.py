import logging
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends

from src.config.settings import settings, Settings
from src.schemas.models import MatrixQuery

from ..services.implementations.matrix_document_loader_service import MatrixDocumentLoaderService
from ..services.implementations.qdrant_retriever_service import QdrantRetrieverService
from ..services.implementations.matrix_generator_service import MatrixGeneratorService
from ..services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@lru_cache(maxsize=1)
def get_settings():
    """Dependency for settings"""
    logger.info("Loading settings...")
    return settings


@lru_cache(maxsize=1)
def get_rag_service(config = Depends(get_settings)):
    """Initialize RAG service and index data"""
    logger.info("Initializing RAG service...")
    
    loader_service = MatrixDocumentLoaderService()
    retriever_service = QdrantRetrieverService(config)
    generator_service = MatrixGeneratorService()
    
    rag_service = RAGService(
        loader=loader_service,
        retriever=retriever_service,
        generator=generator_service
    )
    
    rag_service.index()
    logger.info("RAG service initialized and documents indexed")
    
    return rag_service


@router.post("/query")
async def matrix_query(
    matrix_query: MatrixQuery,
    config = Depends(get_settings),
    rag_service: RAGService = Depends(get_rag_service)
    ):
    """Endpoint to handle queries to the RAG system"""
    try:
        result = await rag_service.query(
                question=matrix_query.query,
                top_k=config.rag_top_k,
                attach_documents=matrix_query.show_context)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
