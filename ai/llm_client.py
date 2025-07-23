"""
LLM client for ChatBI platform.
Handles communication with OpenAI and other language models using LangChain.
"""

from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback
import asyncio

from utils.config import settings, get_openai_config
from utils.logger import get_logger
from utils.exceptions import LLMException, ErrorCodes

logger = get_logger(__name__)


class LLMClient:
    """Client for interacting with Large Language Models."""

    def __init__(self):
        """Initialize LLM client with OpenAI configuration."""
        self.config = get_openai_config()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            self.client = ChatOpenAI(
                api_key=self.config["api_key"],
                model=self.config["model"],
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=2000,
                timeout=30,
                max_retries=3
            )
            logger.info(f"LLM client initialized with model: {self.config['model']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise LLMException(
                "Failed to initialize LLM client",
                ErrorCodes.LLM_API_ERROR,
                str(e)
            )

    async def generate_sql(
            self,
            user_question: str,
            table_schemas: List[Dict[str, Any]],
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.

        Args:
            user_question: User's natural language question
            table_schemas: List of available table schemas
            conversation_history: Previous conversation context

        Returns:
            Dictionary containing generated SQL and metadata
        """
        try:
            # Build context about available tables
            schema_context = self._build_schema_context(table_schemas)

            # Build conversation context
            messages = self._build_messages_for_sql(
                user_question=user_question,
                schema_context=schema_context,
                conversation_history=conversation_history
            )

            # Generate SQL using LLM
            with get_openai_callback() as cb:
                response = await self.client.ainvoke(messages)

                # Extract SQL from response
                sql_query = self._extract_sql_from_response(response.content)

                return {
                    "sql_query": sql_query,
                    "explanation": self._extract_explanation_from_response(response.content),
                    "confidence": self._assess_confidence(response.content),
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost
                }

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise LLMException(
                "Failed to generate SQL query",
                ErrorCodes.LLM_API_ERROR,
                str(e)
            )

    async def suggest_chart_type(
            self,
            sql_query: str,
            sample_data: List[Dict[str, Any]],
            user_question: str
    ) -> Dict[str, Any]:
        """
        Suggest appropriate chart type for the data.

        Args:
            sql_query: The SQL query that generated the data
            sample_data: Sample of the query results
            user_question: Original user question

        Returns:
            Chart suggestion with configuration
        """
        try:
            messages = self._build_messages_for_chart_suggestion(
                sql_query=sql_query,
                sample_data=sample_data,
                user_question=user_question
            )

            with get_openai_callback() as cb:
                response = await self.client.ainvoke(messages)

                chart_suggestion = self._parse_chart_suggestion(response.content)

                return {
                    "chart_type": chart_suggestion.get("chart_type", "bar"),
                    "x_axis": chart_suggestion.get("x_axis"),
                    "y_axis": chart_suggestion.get("y_axis"),
                    "title": chart_suggestion.get("title"),
                    "explanation": chart_suggestion.get("explanation"),
                    "tokens_used": cb.total_tokens
                }

        except Exception as e:
            logger.error(f"Chart suggestion failed: {e}")
            return {
                "chart_type": "bar",
                "explanation": "Could not generate chart suggestion",
                "error": str(e)
            }

    async def explain_results(
            self,
            user_question: str,
            sql_query: str,
            results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation of query results.

        Args:
            user_question: Original user question
            sql_query: Generated SQL query
            results: Query results

        Returns:
            Natural language explanation
        """
        try:
            messages = self._build_messages_for_explanation(
                user_question=user_question,
                sql_query=sql_query,
                results=results
            )

            response = await self.client.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return "I found some results for your query, but couldn't generate a detailed explanation."

    def _build_schema_context(self, table_schemas: List[Dict[str, Any]]) -> str:
        """Build context string describing available tables."""
        schema_parts = []

        for schema in table_schemas:
            table_name = schema.get("name", "unknown")
            columns = schema.get("columns", [])

            column_descriptions = []
            for col in columns:
                col_desc = f"{col['name']} ({col['type']})"
                if col.get('primary_key'):
                    col_desc += " [PK]"
                column_descriptions.append(col_desc)

            schema_parts.append(
                f"Table: {table_name}\n"
                f"Columns: {', '.join(column_descriptions)}\n"
            )

        return "\n".join(schema_parts)

    def _build_messages_for_sql(
            self,
            user_question: str,
            schema_context: str,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List:
        """Build message list for SQL generation."""
        system_prompt = f"""You are an expert SQL analyst. Generate accurate SQL queries based on user questions.

Available database schema:
{schema_context}

Rules:
1. Generate only valid SQL SELECT statements
2. Use proper table and column names from the schema
3. Include appropriate WHERE clauses, JOINs, and aggregations
4. Always use LIMIT to prevent large result sets (default: LIMIT 100)
5. Return only the SQL query, no explanations unless asked
6. For date/time queries, use appropriate date functions
7. Handle case-insensitive searches when appropriate

Format your response as:
```sql
[SQL QUERY HERE]
```

Explanation: [Brief explanation of the query logic]"""

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current question
        messages.append(HumanMessage(content=user_question))

        return messages

    def _build_messages_for_chart_suggestion(
            self,
            sql_query: str,
            sample_data: List[Dict[str, Any]],
            user_question: str
    ) -> List:
        """Build messages for chart type suggestion."""
        data_preview = str(sample_data[:3]) if sample_data else "No data"

        system_prompt = f"""You are a data visualization expert. Suggest the best chart type for the given data.

User Question: {user_question}
SQL Query: {sql_query}
Sample Data: {data_preview}

Available chart types: bar, line, pie, scatter, area, histogram

Respond in this format:
Chart Type: [chart_type]
X-Axis: [column_name]
Y-Axis: [column_name or list of columns]
Title: [suggested_title]
Explanation: [why this chart type is appropriate]"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please suggest the best visualization for this data.")
        ]

    def _build_messages_for_explanation(
            self,
            user_question: str,
            sql_query: str,
            results: List[Dict[str, Any]]
    ) -> List:
        """Build messages for result explanation."""
        result_summary = f"Found {len(results)} results" if results else "No results found"
        sample_results = str(results[:3]) if results else "No data"

        system_prompt = f"""You are a data analyst explaining query results to business users.

User Question: {user_question}
SQL Query: {sql_query}
Results Summary: {result_summary}
Sample Results: {sample_results}

Provide a clear, business-friendly explanation of the results. Include:
1. What the data shows
2. Key insights or patterns
3. Answer to the user's question
4. Any limitations or caveats

Keep the explanation concise and non-technical."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please explain these results.")
        ]

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Look for SQL code blocks
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()

        # Look for SQL statements (fallback)
        lines = response.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if line.endswith(';'):
                    break

        return '\n'.join(sql_lines).replace(';', '').strip()

    def _extract_explanation_from_response(self, response: str) -> str:
        """Extract explanation from LLM response."""
        if "Explanation:" in response:
            return response.split("Explanation:")[-1].strip()
        return ""

    def _assess_confidence(self, response: str) -> float:
        """Assess confidence level of the response."""
        # Simple heuristic based on response characteristics
        confidence = 0.8  # Default confidence

        if "```sql" in response:
            confidence += 0.1
        if "Explanation:" in response:
            confidence += 0.05
        if any(word in response.lower() for word in ["uncertain", "might", "possibly"]):
            confidence -= 0.2

        return min(max(confidence, 0.0), 1.0)

    def _parse_chart_suggestion(self, response: str) -> Dict[str, Any]:
        """Parse chart suggestion from LLM response."""
        suggestion = {}

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("Chart Type:"):
                suggestion["chart_type"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("X-Axis:"):
                suggestion["x_axis"] = line.split(":", 1)[1].strip()
            elif line.startswith("Y-Axis:"):
                suggestion["y_axis"] = line.split(":", 1)[1].strip()
            elif line.startswith("Title:"):
                suggestion["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("Explanation:"):
                suggestion["explanation"] = line.split(":", 1)[1].strip()

        return suggestion