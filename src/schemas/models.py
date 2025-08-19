import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MatrixQuery(BaseModel):
    """Model for a query to the Matrix agent"""
    query: str
    show_context: bool = False


class Document(BaseModel):
    """Enhanced document model with rich metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Document(id={self.id}, page_content={self.page_content[:100]}...)"


class RetrievalResult(BaseModel):
    """Result from retrieval with scoring information"""
    document: Document
    score: float
    rank: int


class MatrixResponse(BaseModel):
    """Structured response model for Matrix queries"""
    answer: str = Field(description="The answer to the user's query based on the script context")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of the answer")
    sources_used: List[str] = Field(description="List of document IDs used as sources")
    reasoning: Optional[str] = Field(default=None, description="Explanation of reasoning process")


class QueryDecomposition(BaseModel):
    """Decomposition of a complex query into subqueries"""
    subqueries: List[str] = Field(
        description="List of subqueries that break down the complex query into manageable parts"
    )
    reasoning: str = Field(
        description="Reasoning behind how the query was decomposed"
    )
    

class SubQueryResponse(BaseModel):
    """Response for a specific subquery"""
    subquery: str = Field(description="The subquery being answered")
    answer: str = Field(description="Answer to the subquery based on context")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level for this specific answer")
    sources_used: List[str] = Field(description="Document IDs used for this subquery")


class AdvancedMatrixResponse(BaseModel):
    """Advanced response model with multi-step reasoning"""
    final_answer: str = Field(description="The final comprehensive answer to the original query")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the answer")
    reasoning: str = Field(description="Detailed explanation of the reasoning process")
    subquery_responses: List[SubQueryResponse] = Field(description="List of all subquery responses")
    sources_used: List[str] = Field(description="All document IDs used as sources")


class QueryResult(BaseModel):
    """Complete result from RAG query"""
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: List[str] = Field(default_factory=list)
    retrieved_documents: List[RetrievalResult] = Field(default_factory=list)
    reasoning: Optional[str] = None
    