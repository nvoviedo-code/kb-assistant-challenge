from typing import List
from pydantic_ai import Agent
from src.schemas.models import Document, QueryResult, MatrixResponse
from ..interfaces.generator_service import GeneratorService


class MatrixGeneratorService(GeneratorService):
    """Specialized LLM generator service (agent) for Matrix script queries"""

    def __init__(self, model_name: str = "openai:gpt-4.1-mini-2025-04-14"):
        self.model_name = model_name
        self.agent = Agent(
            model_name,
            output_type=MatrixResponse,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a specialized assistant for analyzing The Matrix movie script. Your job is to answer questions based ONLY on the provided script context.

CRITICAL RULES:
1. ONLY use information from the provided script context - never use your general knowledge about The Matrix movie
2. If the context doesn't contain enough information to answer the question, explicitly state that
3. Always cite which parts of the context you used by mentioning document IDs or content snippets
4. Be precise and factual in your responses
5. Quote relevant dialogue or script excerpts when appropriate
6. Provide a confidence score based on how well the context supports your answer

RESPONSE FORMAT:
- Answer: Provide a clear, direct answer based on the script context
- Confidence: Rate 0.0-1.0 based on context quality and completeness
- Sources: List the document IDs you referenced

IMPORTANT: If you cannot find sufficient information in the context to answer the question, set confidence to 0.0 and explain what information is missing."""
    
    async def generate_response(self, query: str, context: List[Document]) -> QueryResult:
        """Generate response using Pydantic AI agent"""
        
        # Format context for the agent
        context_text = self._format_context(context)
        
        # Prepare the prompt
        prompt = f"""User Query: {query}

Script Context:
{context_text}

Please answer the user's query based on the provided script context."""
        
        try:
            result = await self.agent.run(prompt)
            
            query_result = QueryResult(
                query=query,
                answer=result.output.answer,
                confidence=result.output.confidence,
                sources_used=result.output.sources_used,
            )
            
            return query_result
            
        except Exception as e:
            return QueryResult(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                confidence=0.0,
                sources_used=[],
            )
        
    def _format_context(self, context: List[Document]) -> str:
        return "\n\n".join([f"Document ID: {doc.id}\nContent: {doc.page_content}" for doc in context])
