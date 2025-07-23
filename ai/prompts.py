"""
Prompt templates for ChatBI platform.
Contains all LLM prompts used for SQL generation, chart suggestions, and explanations.
"""

from typing import Dict, List, Any


class PromptTemplates:
    """Collection of prompt templates for different AI tasks."""

    @staticmethod
    def sql_generation_system_prompt(schema_context: str) -> str:
        """System prompt for SQL generation."""
        return f"""You are an expert SQL analyst specializing in business intelligence queries. 
Your task is to convert natural language questions into accurate, efficient SQL queries.

AVAILABLE DATABASE SCHEMA:
{schema_context}

CORE RULES:
1. Generate ONLY valid SQL SELECT statements
2. Use exact table and column names from the provided schema
3. Always include LIMIT clause (default: LIMIT 100) unless user specifies otherwise
4. Use appropriate JOINs when querying multiple tables
5. Apply proper WHERE conditions for filtering
6. Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
7. Handle dates using standard SQL date functions
8. Make searches case-insensitive when dealing with text

QUERY OPTIMIZATION:
- Use indexes when available (primary keys, foreign keys)
- Prefer EXISTS over IN for subqueries when possible
- Use appropriate ORDER BY for meaningful results
- Group data logically when using aggregations

RESPONSE FORMAT:
```sql
[Your SQL query here]
```

Explanation: [Brief explanation of the query logic and assumptions]

SAFETY CHECKS:
- Never generate INSERT, UPDATE, DELETE, or DROP statements
- Avoid queries that could return extremely large datasets without LIMIT
- Don't access system tables or sensitive metadata tables"""

    @staticmethod
    def sql_generation_user_prompt(user_question: str, conversation_context: str = "") -> str:
        """User prompt for SQL generation."""
        context_part = f"\n\nConversation Context:\n{conversation_context}" if conversation_context else ""

        return f"""Convert this business question into a SQL query:

Question: {user_question}{context_part}

Please generate an appropriate SQL query that answers this question accurately."""

    @staticmethod
    def chart_suggestion_system_prompt() -> str:
        """System prompt for chart type suggestions."""
        return """You are a data visualization expert. Your task is to recommend the most appropriate chart type based on the data characteristics and user intent.

AVAILABLE CHART TYPES:
- bar: Comparing categories or showing distributions
- line: Showing trends over time or continuous data
- pie: Showing parts of a whole (best for <8 categories)
- scatter: Showing relationships between two continuous variables
- area: Showing trends with emphasis on volume/magnitude
- histogram: Showing distribution of continuous data

SELECTION CRITERIA:
1. Data types (categorical, numerical, temporal)
2. Number of data points and categories
3. User's analytical intent (comparison, trend, distribution, relationship)
4. Data cardinality (avoid pie charts with many categories)

RESPONSE FORMAT:
Chart Type: [recommended_chart_type]
X-Axis: [column_name]
Y-Axis: [column_name_or_list]
Title: [descriptive_title]
Color: [optional_color_column]
Explanation: [reasoning_for_choice]

GUIDELINES:
- Consider the story the data tells
- Prioritize clarity and readability
- Suggest appropriate aggregations if needed
- Recommend color encoding when it adds value"""

    @staticmethod
    def chart_suggestion_user_prompt(
            user_question: str,
            sql_query: str,
            sample_data: List[Dict[str, Any]],
            column_info: List[Dict[str, str]]
    ) -> str:
        """User prompt for chart suggestions."""
        data_preview = str(sample_data[:3]) if sample_data else "No data available"
        columns_info = ", ".join([f"{col['name']} ({col['type']})" for col in column_info])

        return f"""Recommend the best chart type for this data:

Original Question: {user_question}
SQL Query: {sql_query}

Data Structure:
Columns: {columns_info}
Sample Data: {data_preview}
Total Rows: {len(sample_data) if sample_data else 0}

Please suggest the most appropriate visualization for this data."""

    @staticmethod
    def results_explanation_system_prompt() -> str:
        """System prompt for explaining query results."""
        return """You are a business intelligence analyst. Your role is to explain data query results in clear, business-friendly language that non-technical users can understand.

EXPLANATION GUIDELINES:
1. Start with a direct answer to the user's question
2. Highlight key insights and patterns in the data
3. Use plain English, avoid technical jargon
4. Provide context and business relevance
5. Mention any limitations or caveats
6. Suggest follow-up questions if relevant

STRUCTURE YOUR RESPONSE:
- Summary: Direct answer to the question
- Key Findings: Most important insights (2-3 points)
- Details: Supporting information from the data
- Notes: Any limitations or caveats

TONE:
- Professional but conversational
- Confident in data interpretation
- Helpful and informative
- Accessible to business users"""

    @staticmethod
    def results_explanation_user_prompt(
            user_question: str,
            sql_query: str,
            results: List[Dict[str, Any]],
            execution_stats: Dict[str, Any]
    ) -> str:
        """User prompt for result explanations."""
        result_count = len(results) if results else 0
        sample_results = str(results[:5]) if results else "No results found"
        exec_time = execution_stats.get("execution_time_ms", 0)

        return f"""Please explain these query results in business terms:

Original Question: {user_question}
SQL Query: {sql_query}

Results Summary:
- Total Records: {result_count}
- Execution Time: {exec_time}ms
- Sample Data: {sample_results}

Provide a clear explanation of what this data shows and how it answers the user's question."""

    @staticmethod
    def sql_validation_prompt(sql_query: str, table_schemas: List[Dict[str, Any]]) -> str:
        """Prompt for SQL validation."""
        schema_names = [schema.get("name", "unknown") for schema in table_schemas]

        return f"""Validate this SQL query for safety and correctness:

SQL Query:
{sql_query}

Available Tables: {', '.join(schema_names)}

Check for:
1. SQL injection risks
2. Dangerous operations (DROP, DELETE, UPDATE, etc.)
3. Table and column name validity
4. Syntax correctness
5. Performance concerns (missing LIMIT, inefficient JOINs)

Respond with:
Status: [SAFE/UNSAFE/WARNING]
Issues: [List any problems found]
Suggestions: [Recommendations for improvement]"""

    @staticmethod
    def conversation_summary_prompt(messages: List[Dict[str, str]]) -> str:
        """Prompt for summarizing conversation context."""
        conversation = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages[-10:]  # Last 10 messages
        ])

        return f"""Summarize this conversation to provide context for the next query:

Recent Conversation:
{conversation}

Provide a brief summary focusing on:
- Key topics discussed
- Data tables mentioned
- Analysis patterns
- User's analytical goals

Keep the summary concise (2-3 sentences) and relevant for SQL generation context."""

    @staticmethod
    def error_explanation_prompt(error_message: str, sql_query: str) -> str:
        """Prompt for explaining SQL errors in user-friendly terms."""
        return f"""Explain this SQL error in simple business terms:

SQL Query:
{sql_query}

Error Message:
{error_message}

Provide:
1. What went wrong in plain English
2. Possible reasons for the error
3. Suggestions to fix the issue
4. Alternative approaches if applicable

Keep the explanation non-technical and actionable."""

    @staticmethod
    def data_insights_prompt(
            table_name: str,
            sample_data: List[Dict[str, Any]],
            column_stats: Dict[str, Any]
    ) -> str:
        """Prompt for generating data insights."""
        sample_preview = str(sample_data[:5]) if sample_data else "No data"

        return f"""Analyze this dataset and provide business insights:

Table: {table_name}
Sample Data: {sample_preview}
Column Statistics: {column_stats}

Generate insights about:
1. Data quality and completeness
2. Interesting patterns or trends
3. Business relevance and use cases
4. Potential analysis opportunities
5. Data anomalies or concerns

Focus on actionable insights that would be valuable for business analysis."""


class ConversationManager:
    """Manages conversation context and history for better AI responses."""

    @staticmethod
    def build_context_from_history(messages: List[Dict[str, str]], max_context: int = 5) -> str:
        """Build conversation context from message history."""
        if not messages:
            return ""

        # Get recent messages for context
        recent_messages = messages[-max_context:]

        context_parts = []
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]  # Truncate long messages

            if role == "user":
                context_parts.append(f"User asked: {content}")
            elif role == "assistant" and "sql" in content.lower():
                context_parts.append(f"Generated SQL query for previous question")

        return "\n".join(context_parts)

    @staticmethod
    def extract_table_references(conversation_history: List[Dict[str, str]]) -> List[str]:
        """Extract table names mentioned in conversation."""
        tables = set()

        for msg in conversation_history:
            content = msg.get("content", "").lower()
            # Simple heuristic to find table references
            if "from " in content or "join " in content:
                words = content.split()
                for i, word in enumerate(words):
                    if word in ["from", "join"] and i + 1 < len(words):
                        table_candidate = words[i + 1].strip("(),;")
                        tables.add(table_candidate)

        return list(tables)

    @staticmethod
    def suggest_followup_questions(
            user_question: str,
            results: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest relevant follow-up questions based on current results."""
        followups = []

        if not results:
            return ["Could you try rephrasing your question?", "What specific data are you looking for?"]

        # Generic follow-ups based on data characteristics
        if len(results) > 1:
            followups.append("What are the top 5 results?")
            followups.append("Can you show this data as a trend over time?")

        # Check for common column patterns
        if results and isinstance(results[0], dict):
            columns = list(results[0].keys())

            if any("date" in col.lower() or "time" in col.lower() for col in columns):
                followups.append("How does this look over different time periods?")

            if any("amount" in col.lower() or "total" in col.lower() for col in columns):
                followups.append("What's the average and total for these amounts?")

            if len(columns) > 2:
                followups.append("Can you break this down by category?")

        return followups[:3]  # Return top 3 suggestions