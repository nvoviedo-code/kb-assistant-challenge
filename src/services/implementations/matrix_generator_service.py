import logging
from typing import List, Set
from pydantic_ai import Agent
from src.schemas.models import AdvancedMatrixResponse, Document, QueryDecomposition, QueryResult, MatrixResponse, SubQueryResponse
from ..interfaces.generator_service import GeneratorService

logger = logging.getLogger(__name__)


class MatrixGeneratorService(GeneratorService):
    """Advanced LLM agent service for Matrix script queries with multi-step reasoning"""

    def __init__(self, model_name: str = "openai:gpt-4.1-mini-2025-04-14"):
        self.model_name = model_name
        
        # Main response agent
        self.response_agent = Agent(
            model_name,
            output_type=MatrixResponse,
            system_prompt=self._get_response_system_prompt()
        )
        
        # Query decomposition agent
        self.decomposition_agent = Agent(
            model_name,
            output_type=QueryDecomposition,
            system_prompt=self._get_decomposition_system_prompt()
        )
        
        # Advanced reasoning agent for complex queries
        self.advanced_agent = Agent(
            model_name,
            output_type=AdvancedMatrixResponse,
            system_prompt=self._get_advanced_system_prompt()
        )
    
    def _get_response_system_prompt(self) -> str:
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
- Reasoning: Explain your thought process (optional)

IMPORTANT: If you cannot find sufficient information in the context to answer the question, set confidence to 0.0 and explain what information is missing."""
    
    def _get_decomposition_system_prompt(self) -> str:
        return """You are a query decomposition expert for The Matrix movie script. Your job is to break down complex questions into simpler subqueries that can be answered independently and then combined.

TASK:
Analyze the user's complex query and break it down into 2-5 simpler subqueries that:
1. Are self-contained and can be answered independently
2. Together cover all aspects of the original query
3. Are specific enough to be matched with relevant script excerpts

CRITICAL RULES:
1. Make subqueries clear and explicit - avoid vague or overly broad questions
2. Ensure subqueries directly relate to the original question
3. Number and order subqueries logically
4. Include counting or frequency subqueries when questions ask "how many times"
5. For questions about character traits, include subqueries about specific actions or dialogue
6. For questions with multiple parts, create separate subqueries for each part

RESPONSE FORMAT:
- Subqueries: List of 2-5 specific subqueries
- Reasoning: Brief explanation of your decomposition approach

EXAMPLE:
For "Why are humans similar to a virus? And who says that?"
Subqueries:
1. "Who in The Matrix script compares humans to a virus?"
2. "What exact words does this character use to compare humans to viruses?"
3. "What reasoning or evidence does the character provide for this comparison?"
"""

    def _get_advanced_system_prompt(self) -> str:
        return """You are an advanced reasoning agent for analyzing The Matrix movie script. Your job is to synthesize answers to complex questions by combining information from multiple subqueries.

TASK:
Create a comprehensive final answer based on the responses to multiple subqueries. You will:
1. Review all subquery responses carefully
2. Identify connections between subquery answers
3. Resolve any contradictions or inconsistencies
4. Synthesize a complete answer to the original complex query

CRITICAL RULES:
1. ONLY use information from the subquery responses - never use general knowledge
2. For counting questions, explicitly state the exact count based on evidence
3. When describing character traits, support with specific examples from the responses
4. For multi-part questions, ensure all parts are addressed in your final answer
5. Calculate overall confidence based on the strength of evidence across all subqueries
6. Provide detailed reasoning that explains how you combined information

RESPONSE FORMAT:
- Final Answer: Comprehensive response to the original query
- Confidence: Overall confidence score (0.0-1.0)
- Reasoning: Detailed explanation of synthesis process
- Subquery Responses: Include all individual subquery responses
- Sources: Aggregate all document IDs used across subqueries
"""
    
    async def generate_response(self, query: str, context: List[Document]) -> QueryResult:
        """Generate response using a multi-step reasoning process for complex queries"""
        logger.info(f"Processing query: {query}")
        
        # Check if this is likely a complex query requiring decomposition
        if self._is_complex_query(query):
            logger.info("Detected complex query - using advanced reasoning approach")
            return await self._handle_complex_query(query, context)
        else:
            logger.info("Using standard response approach for simpler query")
            return await self._handle_simple_query(query, context)
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex based on keywords and structure"""
        # Look for indicators of complex queries
        complex_indicators = [
            "how many times",
            "why are",
            "describe",
            "personality",
            "offer",
            "exchange",
            "purpose",
            "who created",
            "similar to",
            "and who",
            "why did",
            "what is the relationship"
        ]
        
        # Check if query contains any complex indicators
        query_lower = query.lower()
        for indicator in complex_indicators:
            if indicator in query_lower:
                return True
        
        # Check if query has multiple question marks
        if query.count("?") > 1:
            return True
            
        return False
    
    async def _handle_simple_query(self, query: str, context: List[Document]) -> QueryResult:
        """Handle simpler queries directly with the response agent"""
        context_text = self._format_context(context)
        
        prompt = f"""User Query: {query}

Script Context:
{context_text}

Please answer the user's query based on the provided script context."""
        
        try:
            result = await self.response_agent.run(prompt)
            
            query_result = QueryResult(
                query=query,
                answer=result.output.answer,
                confidence=result.output.confidence,
                sources_used=result.output.sources_used,
                reasoning=result.output.reasoning
            )
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error in simple query handling: {str(e)}")
            return QueryResult(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                confidence=0.0,
                sources_used=[],
            )
    
    async def _handle_complex_query(self, query: str, context: List[Document]) -> QueryResult:
        """Handle complex queries using query decomposition and multi-step reasoning"""
        try:
            # Step 1: Decompose the query into subqueries
            decomposition = await self._decompose_query(query)
            logger.info(f"Query decomposed into {len(decomposition.subqueries)} subqueries")
            
            # Step 2: Answer each subquery individually
            subquery_responses = []
            all_sources: Set[str] = set()
            
            for subquery in decomposition.subqueries:
                logger.info(f"Processing subquery: {subquery}")
                # Use the response agent to answer each subquery
                subquery_result = await self._handle_simple_query(subquery, context)
                
                subquery_response = SubQueryResponse(
                    subquery=subquery,
                    answer=subquery_result.answer,
                    confidence=subquery_result.confidence,
                    sources_used=subquery_result.sources_used
                )
                
                subquery_responses.append(subquery_response)
                all_sources.update(subquery_result.sources_used)
            
            # Step 3: Synthesize the final answer using the advanced agent
            return await self._synthesize_final_answer(query, subquery_responses, list(all_sources))
            
        except Exception as e:
            logger.error(f"Error in complex query handling: {str(e)}")
            return QueryResult(
                query=query,
                answer=f"I encountered an error while processing your complex query: {str(e)}",
                confidence=0.0,
                sources_used=[],
            )
    
    async def _decompose_query(self, query: str) -> QueryDecomposition:
        """Decompose a complex query into subqueries"""
        prompt = f"""Please decompose the following complex query into simpler subqueries:

Original Query: {query}

Break this down into 2-5 specific subqueries that will help answer the original query."""

        result = await self.decomposition_agent.run(prompt)
        return result.output
    
    async def _synthesize_final_answer(self, 
                                      original_query: str, 
                                      subquery_responses: List[SubQueryResponse],
                                      all_sources: List[str]) -> QueryResult:
        """Synthesize a final answer from subquery responses"""
        
        # Format the subquery responses for the agent
        subqueries_text = "\n\n".join([
            f"Subquery: {resp.subquery}\nAnswer: {resp.answer}\nConfidence: {resp.confidence}\nSources: {', '.join(resp.sources_used)}"
            for resp in subquery_responses
        ])
        
        prompt = f"""Original Complex Query: {original_query}

Subquery Responses:
{subqueries_text}

Based on these subquery responses, please synthesize a comprehensive final answer to the original query."""

        try:
            result = await self.advanced_agent.run(prompt)
            
            query_result = QueryResult(
                query=original_query,
                answer=result.output.final_answer,
                confidence=result.output.confidence,
                sources_used=all_sources,
                reasoning=result.output.reasoning
            )
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error in answer synthesis: {str(e)}")
            # Fallback to using the highest confidence subquery response
            best_subquery = max(subquery_responses, key=lambda x: x.confidence)
            
            return QueryResult(
                query=original_query,
                answer=f"Based on partial analysis: {best_subquery.answer}",
                confidence=best_subquery.confidence * 0.8,  # Reduce confidence due to synthesis failure
                sources_used=all_sources,
                reasoning="Unable to complete full synthesis - using highest confidence partial result."
            )
    
    def _format_context(self, context: List[Document]) -> str:
        """Format context documents for agent prompts"""
        return "\n\n".join([f"Document ID: {doc.id}\nContent: {doc.page_content}" for doc in context])
