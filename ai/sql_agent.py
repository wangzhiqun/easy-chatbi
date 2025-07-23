"""
SQL Agent for ChatBI platform.
Coordinates SQL generation, validation, and execution workflow.
"""

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from .llm_client import LLMClient
from .prompts import PromptTemplates, ConversationManager
from .tools.sql_validator import SQLValidator
from .tools.sql_executor import SQLExecutor
from .tools.data_analyzer import DataAnalyzer
from utils.logger import get_logger
from utils.exceptions import SQLSecurityException, LLMException, ErrorCodes

logger = get_logger(__name__)


class SQLAgent:
    """
    Intelligent SQL agent that handles the complete workflow of converting
    natural language questions to SQL queries and executing them safely.
    """

    def __init__(self):
        """Initialize SQL agent with required components."""
        self.llm_client = LLMClient()
        self.sql_validator = SQLValidator()
        self.sql_executor = SQLExecutor()
        self.data_analyzer = DataAnalyzer()
        self.conversation_manager = ConversationManager()

    async def process_question(
            self,
            user_question: str,
            user_id: int,
            table_schemas: List[Dict[str, Any]],
            conversation_history: Optional[List[Dict[str, str]]] = None,
            max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL query with results.

        Args:
            user_question: Natural language question from user
            user_id: ID of the user making the request
            table_schemas: Available database schemas
            conversation_history: Previous conversation context
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing SQL query, results, and metadata
        """
        logger.info(f"Processing question for user {user_id}: {user_question[:100]}...")

        start_time = datetime.now()
        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                # Step 1: Generate SQL from natural language
                sql_result = await self._generate_sql(
                    user_question=user_question,
                    table_schemas=table_schemas,
                    conversation_history=conversation_history,
                    previous_error=last_error if attempt > 0 else None
                )

                # Step 2: Validate SQL for safety and correctness
                validation_result = await self._validate_sql(
                    sql_query=sql_result["sql_query"],
                    table_schemas=table_schemas
                )

                if not validation_result["is_safe"]:
                    raise SQLSecurityException(
                        f"SQL query failed safety validation: {validation_result['issues']}",
                        ErrorCodes.SQL_INJECTION_DETECTED
                    )

                # Step 3: Execute SQL query
                execution_result = await self._execute_sql(
                    sql_query=sql_result["sql_query"],
                    user_id=user_id
                )

                # Step 4: Analyze results
                analysis_result = await self._analyze_results(
                    sql_query=sql_result["sql_query"],
                    results=execution_result["data"],
                    user_question=user_question
                )

                # Calculate total processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                return {
                    "success": True,
                    "user_question": user_question,
                    "sql_query": sql_result["sql_query"],
                    "sql_explanation": sql_result.get("explanation", ""),
                    "execution_status": "success",
                    "execution_time_ms": int(processing_time),
                    "data": execution_result["data"],
                    "row_count": len(execution_result["data"]),
                    "analysis": analysis_result,
                    "validation": validation_result,
                    "llm_metadata": {
                        "tokens_used": sql_result.get("tokens_used", 0),
                        "cost": sql_result.get("cost", 0),
                        "confidence": sql_result.get("confidence", 0.8),
                        "attempts": attempt + 1
                    }
                }

            except SQLSecurityException:
                # Don't retry security violations
                raise

            except Exception as e:
                attempt += 1
                last_error = str(e)
                logger.warning(f"SQL generation attempt {attempt} failed: {e}")

                if attempt > max_retries:
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    return {
                        "success": False,
                        "user_question": user_question,
                        "sql_query": None,
                        "execution_status": "error",
                        "execution_time_ms": int(processing_time),
                        "error_message": f"Failed to process question after {max_retries + 1} attempts: {last_error}",
                        "attempts": attempt
                    }

        # This shouldn't be reached, but just in case
        return {
            "success": False,
            "user_question": user_question,
            "error_message": "Unexpected error in SQL agent processing"
        }

    async def _generate_sql(
            self,
            user_question: str,
            table_schemas: List[Dict[str, Any]],
            conversation_history: Optional[List[Dict[str, str]]] = None,
            previous_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language question."""
        try:
            # Build conversation context
            context = ""
            if conversation_history:
                context = self.conversation_manager.build_context_from_history(conversation_history)

            # Add error context for retry attempts
            if previous_error:
                context += f"\n\nPrevious attempt failed with error: {previous_error}\nPlease generate a corrected SQL query."

            # Generate SQL using LLM
            result = await self.llm_client.generate_sql(
                user_question=user_question,
                table_schemas=table_schemas,
                conversation_history=conversation_history
            )

            logger.info(f"SQL generated successfully: {result['sql_query'][:100]}...")
            return result

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise LLMException(
                f"Failed to generate SQL query: {str(e)}",
                ErrorCodes.LLM_API_ERROR
            )

    async def _validate_sql(
            self,
            sql_query: str,
            table_schemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate SQL query for safety and correctness."""
        try:
            validation_result = await self.sql_validator.validate_query(
                sql_query=sql_query,
                available_tables=table_schemas
            )

            logger.info(f"SQL validation result: {validation_result['status']}")
            return validation_result

        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            # If validation fails, assume unsafe
            return {
                "is_safe": False,
                "status": "error",
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Please try rephrasing your question"]
            }

    async def _execute_sql(
            self,
            sql_query: str,
            user_id: int
    ) -> Dict[str, Any]:
        """Execute SQL query safely."""
        try:
            result = await self.sql_executor.execute_query(
                sql_query=sql_query,
                user_id=user_id
            )

            logger.info(f"SQL executed successfully, returned {len(result.get('data', []))} rows")
            return result

        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

    async def _analyze_results(
            self,
            sql_query: str,
            results: List[Dict[str, Any]],
            user_question: str
    ) -> Dict[str, Any]:
        """Analyze query results and generate insights."""
        try:
            analysis = await self.data_analyzer.analyze_results(
                sql_query=sql_query,
                results=results,
                user_question=user_question
            )

            # Generate natural language explanation
            explanation = await self.llm_client.explain_results(
                user_question=user_question,
                sql_query=sql_query,
                results=results
            )

            analysis["explanation"] = explanation
            return analysis

        except Exception as e:
            logger.warning(f"Result analysis failed: {e}")
            return {
                "summary": f"Query returned {len(results)} results",
                "explanation": "Analysis could not be generated",
                "insights": [],
                "suggestions": []
            }

    async def suggest_improvements(
            self,
            sql_query: str,
            execution_result: Dict[str, Any]
    ) -> List[str]:
        """Suggest improvements for SQL queries based on execution results."""
        suggestions = []

        try:
            # Analyze execution performance
            execution_time = execution_result.get("execution_time_ms", 0)
            row_count = len(execution_result.get("data", []))

            # Performance suggestions
            if execution_time > 5000:  # > 5 seconds
                suggestions.append("Consider adding WHERE clauses to limit the data scanned")
                suggestions.append("Check if appropriate indexes exist on filtered columns")

            if row_count == 0:
                suggestions.append("Try broadening your search criteria")
                suggestions.append("Check if the data exists in the specified time range")

            elif row_count >= 100:  # Hit the default limit
                suggestions.append("Consider adding more specific filters to narrow results")
                suggestions.append("Use ORDER BY to get the most relevant results first")

            # SQL structure suggestions
            if "GROUP BY" not in sql_query.upper() and row_count > 50:
                suggestions.append("Consider grouping the data for better insights")

            if "ORDER BY" not in sql_query.upper() and row_count > 10:
                suggestions.append("Add ORDER BY to sort results meaningfully")

            return suggestions

        except Exception as e:
            logger.warning(f"Could not generate suggestions: {e}")
            return ["Query executed successfully"]

    async def get_query_explanation(
            self,
            sql_query: str,
            table_schemas: List[Dict[str, Any]]
    ) -> str:
        """Get detailed explanation of what a SQL query does."""
        try:
            # Use LLM to explain the query
            schema_context = "\n".join([
                f"Table {schema['name']}: {', '.join([col['name'] for col in schema.get('columns', [])])}"
                for schema in table_schemas
            ])

            explanation_prompt = f"""Explain this SQL query in simple business terms:

SQL Query:
{sql_query}

Available Tables:
{schema_context}

Explain:
1. What data this query retrieves
2. How it filters and processes the data
3. What business question it answers
4. Any important details about the logic

Use plain English that business users can understand."""

            # This would use the LLM client to generate explanation
            # For now, return a basic explanation
            return f"This query retrieves data from the database and processes it according to the specified criteria."

        except Exception as e:
            logger.warning(f"Could not generate query explanation: {e}")
            return "Query explanation could not be generated."

    def get_conversation_suggestions(
            self,
            conversation_history: List[Dict[str, str]],
            current_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Get suggestions for follow-up questions based on conversation and results."""
        return self.conversation_manager.suggest_followup_questions(
            user_question=conversation_history[-1].get("content", "") if conversation_history else "",
            results=current_results
        )