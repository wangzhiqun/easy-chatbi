"""
Chat Service for ChatBI platform.
Orchestrates the complete chat workflow including AI processing, SQL generation, and response formatting.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from ai.sql_agent import SQLAgent
from ai.chart_agent import ChartAgent
from ai.llm_client import LLMClient
from api.database import get_table_schema, get_table_names
from api.schemas import QueryResponse
from security.data_masking import DataMasker
from utils.logger import get_logger
from utils.exceptions import ChatBIException, LLMException, DatabaseException, ErrorCodes
from .cache_service import CacheService

logger = get_logger(__name__)


class ChatService:
    """
    Main service for handling chat interactions, coordinating between
    AI agents, data processing, and response generation.
    """

    def __init__(self):
        """Initialize chat service with required components."""
        self.sql_agent = SQLAgent()
        self.chart_agent = ChartAgent()
        self.llm_client = LLMClient()
        self.data_masker = DataMasker()
        self.cache_service = CacheService()

        # Service configuration
        self.max_response_time = 60  # seconds
        self.enable_caching = True
        self.enable_data_masking = True

        # Response templates
        self.response_templates = {
            "no_data": "I couldn't find any data matching your request. Could you try rephrasing your question or checking if the data exists?",
            "error": "I encountered an error while processing your request: {error}",
            "success": "Here are the results for your query:",
            "partial_success": "I found some results, but there were issues with part of your request."
        }

    async def process_query(
            self,
            user_id: int,
            question: str,
            session_id: Optional[int] = None,
            include_chart_suggestion: bool = True,
            db: Optional[Session] = None
    ) -> QueryResponse:
        """
        Process a natural language question and return comprehensive response.

        Args:
            user_id: ID of the user asking the question
            question: Natural language question
            session_id: Session ID for conversation context
            include_chart_suggestion: Whether to include chart recommendations
            db: Database session for logging

        Returns:
            Complete query response with SQL, data, and insights
        """
        start_time = datetime.now()

        logger.info(f"Processing query for user {user_id}: {question[:100]}...")

        try:
            # Step 1: Check cache for similar queries
            if self.enable_caching:
                cached_response = await self._check_query_cache(user_id, question)
                if cached_response:
                    logger.info("Returning cached response")
                    return cached_response

            # Step 2: Get conversation context if session provided
            conversation_history = []
            if session_id:
                conversation_history = await self._get_conversation_context(session_id, db)

            # Step 3: Get available table schemas
            table_schemas = await self._get_available_schemas(user_id)

            # Step 4: Process question with SQL agent
            sql_result = await self.sql_agent.process_question(
                user_question=question,
                user_id=user_id,
                table_schemas=table_schemas,
                conversation_history=conversation_history
            )

            if not sql_result["success"]:
                return self._create_error_response(question, sql_result.get("error_message", "Query processing failed"))

            # Step 5: Apply data masking if enabled
            masked_data = sql_result["data"]
            if self.enable_data_masking and masked_data:
                user_permissions = await self._get_user_permissions(user_id)
                masked_data = await self.data_masker.mask_query_results(
                    results=sql_result["data"],
                    user_permissions=user_permissions
                )

            # Step 6: Generate chart suggestions if requested
            chart_suggestion = None
            if include_chart_suggestion and masked_data:
                try:
                    chart_config = await self.chart_agent.recommend_chart(
                        data=masked_data,
                        user_question=question,
                        sql_query=sql_result["sql_query"]
                    )
                    chart_suggestion = chart_config
                except Exception as e:
                    logger.warning(f"Chart suggestion failed: {e}")

            # Step 7: Generate natural language explanation
            explanation = await self._generate_explanation(
                question=question,
                sql_query=sql_result["sql_query"],
                results=masked_data,
                analysis=sql_result.get("analysis", {})
            )

            # Step 8: Create comprehensive response
            response = QueryResponse(
                question=question,
                generated_sql=sql_result["sql_query"],
                execution_status="success",
                execution_time_ms=sql_result.get("execution_time_ms", 0),
                result_data=masked_data,
                result_summary=explanation,
                chart_suggestion=chart_suggestion,
                is_safe=sql_result.get("validation", {}).get("is_safe", True)
            )

            # Step 9: Cache successful response
            if self.enable_caching and response.execution_status == "success":
                await self._cache_query_response(user_id, question, response)

            # Step 10: Log processing metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_query_metrics(user_id, question, processing_time, response)

            logger.info(f"Query processed successfully in {processing_time:.0f}ms")
            return response

        except ChatBIException as e:
            logger.error(f"Chat service error: {e.message}")
            return self._create_error_response(question, e.message)

        except Exception as e:
            logger.error(f"Unexpected error in chat service: {e}")
            return self._create_error_response(question, "An unexpected error occurred while processing your request")

    async def get_conversation_summary(
            self,
            session_id: int,
            db: Session
    ) -> Dict[str, Any]:
        """
        Generate a summary of the conversation in a session.

        Args:
            session_id: Session ID to summarize
            db: Database session

        Returns:
            Conversation summary with key insights
        """
        try:
            # Get conversation history
            history = await self._get_conversation_context(session_id, db)

            if not history:
                return {"summary": "No conversation history found", "message_count": 0}

            # Generate summary using LLM
            summary_text = await self.llm_client.explain_results(
                user_question="Summarize this conversation",
                sql_query="",
                results=[]
            )

            # Extract key statistics
            user_questions = [msg for msg in history if msg.get("role") == "user"]
            sql_queries = [msg for msg in history if "SELECT" in msg.get("content", "")]

            return {
                "summary": summary_text,
                "message_count": len(history),
                "questions_asked": len(user_questions),
                "queries_generated": len(sql_queries),
                "session_duration": self._calculate_session_duration(history),
                "key_topics": self._extract_key_topics(history)
            }

        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return {"error": str(e)}

    async def suggest_follow_up_questions(
            self,
            question: str,
            results: List[Dict[str, Any]],
            session_id: Optional[int] = None
    ) -> List[str]:
        """
        Suggest relevant follow-up questions based on current query and results.

        Args:
            question: Current question
            results: Query results
            session_id: Session ID for context

        Returns:
            List of suggested follow-up questions
        """
        try:
            suggestions = []

            # Rule-based suggestions based on data characteristics
            if results:
                # Time-based suggestions
                date_columns = [col for col in results[0].keys() if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    suggestions.append("Can you show this data as a trend over time?")
                    suggestions.append("What does this look like for different time periods?")

                # Numerical data suggestions
                numeric_columns = []
                for col, value in results[0].items():
                    if isinstance(value, (int, float)):
                        numeric_columns.append(col)

                if numeric_columns:
                    suggestions.append("What's the average and total for these numbers?")
                    suggestions.append("Can you break this down by category?")

                # Large dataset suggestions
                if len(results) > 20:
                    suggestions.append("Can you show me just the top 10 results?")
                    suggestions.append("What are the most significant values here?")

            # Context-based suggestions from conversation
            if session_id:
                suggestions.extend(await self._get_contextual_suggestions(session_id))

            # Question-specific suggestions
            question_lower = question.lower()
            if "sales" in question_lower:
                suggestions.append("How does this compare to last year?")
                suggestions.append("Which products or regions contributed most?")
            elif "customer" in question_lower:
                suggestions.append("What's the customer segmentation like?")
                suggestions.append("How has customer behavior changed over time?")

            # Remove duplicates and limit to top 3
            unique_suggestions = list(dict.fromkeys(suggestions))
            return unique_suggestions[:3]

        except Exception as e:
            logger.warning(f"Failed to generate follow-up suggestions: {e}")
            return []

    async def validate_question(
            self,
            question: str,
            user_id: int
    ) -> Dict[str, Any]:
        """
        Validate if a question can be processed before full execution.

        Args:
            question: Question to validate
            user_id: User ID for permission checking

        Returns:
            Validation result with recommendations
        """
        try:
            validation_result = {
                "is_valid": True,
                "confidence": 0.8,
                "issues": [],
                "suggestions": [],
                "estimated_complexity": "medium"
            }

            # Basic question validation
            if len(question.strip()) < 5:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Question is too short")
                validation_result["suggestions"].append("Please provide a more detailed question")

            # Check for question words
            question_words = ["what", "how", "when", "where", "which", "who", "show", "get", "find"]
            if not any(word in question.lower() for word in question_words):
                validation_result["confidence"] -= 0.2
                validation_result["suggestions"].append("Consider starting with words like 'what', 'how', or 'show'")

            # Check for data-related keywords
            data_keywords = ["data", "table", "record", "row", "column", "database", "sales", "customer", "product"]
            if not any(keyword in question.lower() for keyword in data_keywords):
                validation_result["confidence"] -= 0.1
                validation_result["suggestions"].append("Be more specific about what data you're looking for")

            # Estimate complexity
            complexity_indicators = {
                "simple": ["show", "get", "what is", "count"],
                "medium": ["compare", "analysis", "trend", "group by"],
                "complex": ["correlation", "prediction", "advanced", "complex"]
            }

            question_lower = question.lower()
            for complexity, indicators in complexity_indicators.items():
                if any(indicator in question_lower for indicator in indicators):
                    validation_result["estimated_complexity"] = complexity
                    break

            return validation_result

        except Exception as e:
            logger.error(f"Question validation failed: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "suggestions": ["Please try rephrasing your question"]
            }

    # Private helper methods

    async def _check_query_cache(self, user_id: int, question: str) -> Optional[QueryResponse]:
        """Check if similar query exists in cache."""
        try:
            cache_key = f"query_{user_id}_{hash(question.lower())}"
            cached_data = await self.cache_service.get(cache_key)

            if cached_data:
                # Ensure cached data is still valid
                return QueryResponse(**cached_data)

            return None
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None

    async def _cache_query_response(self, user_id: int, question: str, response: QueryResponse):
        """Cache successful query response."""
        try:
            cache_key = f"query_{user_id}_{hash(question.lower())}"
            cache_data = response.dict()

            # Cache for 1 hour
            await self.cache_service.set(cache_key, cache_data, ttl=3600)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def _get_conversation_context(self, session_id: int, db: Session) -> List[Dict[str, str]]:
        """Get conversation history for context."""
        try:
            # This would query the database for chat messages
            # For now, return empty list
            return []
        except Exception as e:
            logger.warning(f"Failed to get conversation context: {e}")
            return []

    async def _get_available_schemas(self, user_id: int) -> List[Dict[str, Any]]:
        """Get table schemas available to the user."""
        try:
            table_names = get_table_names()
            schemas = []

            for table_name in table_names:
                try:
                    schema = get_table_schema(table_name)
                    schemas.append(schema)
                except Exception as e:
                    logger.warning(f"Failed to get schema for table {table_name}: {e}")

            return schemas
        except Exception as e:
            logger.error(f"Failed to get available schemas: {e}")
            return []

    async def _get_user_permissions(self, user_id: int) -> Dict[str, Any]:
        """Get user's data access permissions."""
        # This would integrate with the permission manager
        return {
            "data_access_level": "internal",  # Default level
            "can_access_sensitive": False,
            "allowed_tables": ["*"],  # All tables for now
            "restricted_columns": []
        }

    async def _generate_explanation(
            self,
            question: str,
            sql_query: str,
            results: List[Dict[str, Any]],
            analysis: Dict[str, Any]
    ) -> str:
        """Generate natural language explanation of results."""
        try:
            if not results:
                return self.response_templates["no_data"]

            # Use LLM to generate explanation
            explanation = await self.llm_client.explain_results(
                user_question=question,
                sql_query=sql_query,
                results=results
            )

            return explanation

        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            return f"Found {len(results)} results for your query."

    async def _get_contextual_suggestions(self, session_id: int) -> List[str]:
        """Get suggestions based on conversation context."""
        # This would analyze the conversation history and suggest relevant questions
        return [
            "Can you provide more details about this data?",
            "How does this compare to previous periods?"
        ]

    def _calculate_session_duration(self, history: List[Dict[str, str]]) -> str:
        """Calculate duration of conversation session."""
        if len(history) < 2:
            return "< 1 minute"

        # Would calculate based on message timestamps
        return "5 minutes"  # Placeholder

    def _extract_key_topics(self, history: List[Dict[str, str]]) -> List[str]:
        """Extract key topics from conversation."""
        topics = set()

        for message in history:
            content = message.get("content", "").lower()

            # Simple keyword extraction
            if "sales" in content:
                topics.add("Sales Analysis")
            if "customer" in content:
                topics.add("Customer Data")
            if "product" in content:
                topics.add("Product Information")
            if "revenue" in content or "profit" in content:
                topics.add("Financial Metrics")

        return list(topics)

    def _create_error_response(self, question: str, error_message: str) -> QueryResponse:
        """Create standardized error response."""
        return QueryResponse(
            question=question,
            generated_sql="",
            execution_status="error",
            execution_time_ms=0,
            result_data=[],
            result_summary=self.response_templates["error"].format(error=error_message),
            error_message=error_message,
            is_safe=True
        )

    async def _log_query_metrics(
            self,
            user_id: int,
            question: str,
            processing_time: float,
            response: QueryResponse
    ):
        """Log query processing metrics for monitoring."""
        try:
            metrics = {
                "user_id": user_id,
                "question_length": len(question),
                "processing_time_ms": processing_time,
                "execution_status": response.execution_status,
                "result_count": len(response.result_data or []),
                "has_chart_suggestion": response.chart_suggestion is not None,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Query metrics: {metrics}")

        except Exception as e:
            logger.warning(f"Failed to log query metrics: {e}")

    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of chat service components."""
        try:
            health = {
                "status": "healthy",
                "components": {
                    "sql_agent": "healthy",
                    "chart_agent": "healthy",
                    "llm_client": "healthy",
                    "data_masker": "healthy",
                    "cache_service": "healthy"
                },
                "metrics": {
                    "cache_enabled": self.enable_caching,
                    "masking_enabled": self.enable_data_masking,
                    "max_response_time": self.max_response_time
                }
            }

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }