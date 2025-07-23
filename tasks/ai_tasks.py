"""
AI Tasks for ChatBI platform.
Handles asynchronous AI processing including query analysis and chart generation.
"""

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from .celery_app import app
from ai.sql_agent import SQLAgent
from ai.chart_agent import ChartAgent
from ai.llm_client import LLMClient
from ai.tools.data_analyzer import DataAnalyzer
from utils.logger import get_logger
from utils.exceptions import LLMException, DataProcessingException

logger = get_logger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_query(self, user_question: str, user_id: int, table_schemas: List[Dict[str, Any]],
                  conversation_history: Optional[List[Dict[str, str]]] = None):
    """
    Asynchronously analyze a user query and generate SQL with results.

    Args:
        user_question: Natural language question from user
        user_id: ID of the user making the request
        table_schemas: Available database schemas
        conversation_history: Previous conversation context

    Returns:
        Analysis result with SQL query and execution status
    """
    try:
        logger.info(f"Starting async query analysis for user {user_id}: {user_question[:100]}...")

        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing AI agents', 'progress': 10}
        )

        # Initialize SQL agent
        sql_agent = SQLAgent()

        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing query structure', 'progress': 30}
        )

        # Run async query processing in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                sql_agent.process_question(
                    user_question=user_question,
                    user_id=user_id,
                    table_schemas=table_schemas,
                    conversation_history=conversation_history
                )
            )
        finally:
            loop.close()

        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Processing results', 'progress': 80}
        )

        # Add task metadata
        result['task_info'] = {
            'task_id': self.request.id,
            'completed_at': datetime.now().isoformat(),
            'processing_mode': 'async'
        }

        logger.info(f"Async query analysis completed for user {user_id}")
        return result

    except Exception as exc:
        logger.error(f"Async query analysis failed: {exc}")

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying query analysis (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

        # Final failure
        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat(),
                'retries': self.request.retries
            }
        }


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def generate_chart_suggestion(self, data: List[Dict[str, Any]], user_question: str,
                              sql_query: str, column_metadata: Optional[List[Dict[str, str]]] = None):
    """
    Asynchronously generate chart suggestions for query results.

    Args:
        data: Query result data
        user_question: Original user question
        sql_query: SQL query that generated the data
        column_metadata: Metadata about columns

    Returns:
        Chart recommendation with configuration
    """
    try:
        logger.info(f"Generating chart suggestion for query: {user_question[:50]}...")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing data characteristics', 'progress': 20}
        )

        # Initialize chart agent
        chart_agent = ChartAgent()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating chart recommendations', 'progress': 60}
        )

        # Run async chart generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            chart_config = loop.run_until_complete(
                chart_agent.recommend_chart(
                    data=data,
                    user_question=user_question,
                    sql_query=sql_query,
                    column_metadata=column_metadata
                )
            )
        finally:
            loop.close()

        # Add task metadata
        chart_config['task_info'] = {
            'task_id': self.request.id,
            'completed_at': datetime.now().isoformat(),
            'data_points_analyzed': len(data)
        }

        logger.info("Chart suggestion generation completed")
        return chart_config

    except Exception as exc:
        logger.error(f"Chart suggestion generation failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))

        return {
            'chart_type': 'bar',
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=2)
def analyze_data_patterns(self, data: List[Dict[str, Any]], analysis_type: str = 'comprehensive'):
    """
    Asynchronously analyze data patterns and generate insights.

    Args:
        data: Data to analyze
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results with patterns and insights
    """
    try:
        logger.info(f"Starting data pattern analysis for {len(data)} records")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing data analyzer', 'progress': 10}
        )

        # Initialize data analyzer
        analyzer = DataAnalyzer()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing data characteristics', 'progress': 40}
        )

        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            analysis_result = loop.run_until_complete(
                analyzer.analyze_results(
                    sql_query="",  # Not needed for pattern analysis
                    results=data,
                    user_question=f"Analyze patterns in this data ({analysis_type})"
                )
            )
        finally:
            loop.close()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating insights', 'progress': 80}
        )

        # Add task metadata
        analysis_result['task_info'] = {
            'task_id': self.request.id,
            'analysis_type': analysis_type,
            'completed_at': datetime.now().isoformat(),
            'records_analyzed': len(data)
        }

        logger.info("Data pattern analysis completed")
        return analysis_result

    except Exception as exc:
        logger.error(f"Data pattern analysis failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=3)
def generate_natural_language_summary(self, query_results: List[Dict[str, Any]],
                                      user_question: str, sql_query: str):
    """
    Generate natural language summary of query results.

    Args:
        query_results: Results from SQL query
        user_question: Original user question
        sql_query: SQL query that was executed

    Returns:
        Natural language summary
    """
    try:
        logger.info("Generating natural language summary")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing language model', 'progress': 20}
        )

        # Initialize LLM client
        llm_client = LLMClient()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating explanation', 'progress': 60}
        )

        # Run async explanation generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            explanation = loop.run_until_complete(
                llm_client.explain_results(
                    user_question=user_question,
                    sql_query=sql_query,
                    results=query_results
                )
            )
        finally:
            loop.close()

        result = {
            'summary': explanation,
            'task_info': {
                'task_id': self.request.id,
                'completed_at': datetime.now().isoformat(),
                'result_count': len(query_results)
            }
        }

        logger.info("Natural language summary generated")
        return result

    except Exception as exc:
        logger.error(f"Summary generation failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'summary': f"Found {len(query_results)} results for your query.",
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=2)
def validate_sql_security(self, sql_query: str, user_id: int, table_schemas: List[Dict[str, Any]]):
    """
    Asynchronously validate SQL query for security issues.

    Args:
        sql_query: SQL query to validate
        user_id: ID of user submitting query
        table_schemas: Available table schemas

    Returns:
        Security validation results
    """
    try:
        logger.info(f"Validating SQL security for user {user_id}")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Running security checks', 'progress': 50}
        )

        # Import here to avoid circular imports
        from security.sql_guardian import SQLGuardian

        guardian = SQLGuardian()

        # Run async validation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            validation_result = loop.run_until_complete(
                guardian.validate_query(
                    sql_query=sql_query,
                    user_id=user_id
                )
            )
        finally:
            loop.close()

        validation_result['task_info'] = {
            'task_id': self.request.id,
            'completed_at': datetime.now().isoformat(),
            'user_id': user_id
        }

        logger.info(f"SQL security validation completed: {validation_result['risk_level']}")
        return validation_result

    except Exception as exc:
        logger.error(f"SQL security validation failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'is_safe': False,
            'risk_level': 'high',
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True)
def batch_analyze_queries(self, queries: List[Dict[str, Any]], user_id: int):
    """
    Analyze multiple queries in batch for efficiency.

    Args:
        queries: List of query dictionaries with question and metadata
        user_id: ID of user submitting queries

    Returns:
        List of analysis results
    """
    try:
        logger.info(f"Starting batch analysis of {len(queries)} queries for user {user_id}")

        results = []
        total_queries = len(queries)

        for i, query_data in enumerate(queries):
            try:
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'status': f'Processing query {i + 1} of {total_queries}',
                        'progress': int((i / total_queries) * 100)
                    }
                )

                # Process individual query
                query_result = analyze_query.apply_async(
                    args=[
                        query_data['question'],
                        user_id,
                        query_data.get('table_schemas', []),
                        query_data.get('conversation_history', [])
                    ]
                ).get(timeout=300)  # 5 minute timeout per query

                results.append({
                    'query_index': i,
                    'question': query_data['question'],
                    'result': query_result
                })

            except Exception as e:
                logger.error(f"Batch query {i} failed: {e}")
                results.append({
                    'query_index': i,
                    'question': query_data.get('question', 'Unknown'),
                    'error': str(e)
                })

        batch_result = {
            'total_queries': total_queries,
            'successful_queries': len([r for r in results if 'error' not in r]),
            'results': results,
            'task_info': {
                'task_id': self.request.id,
                'completed_at': datetime.now().isoformat(),
                'user_id': user_id
            }
        }

        logger.info(f"Batch analysis completed: {batch_result['successful_queries']}/{total_queries} successful")
        return batch_result

    except Exception as exc:
        logger.error(f"Batch query analysis failed: {exc}")
        return {
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True)
def precompute_common_queries(self, table_names: List[str]):
    """
    Precompute results for common queries to improve response times.

    Args:
        table_names: List of table names to generate queries for

    Returns:
        Results of precomputed queries
    """
    try:
        logger.info(f"Precomputing common queries for {len(table_names)} tables")

        # Common query patterns
        common_patterns = [
            "SELECT COUNT(*) FROM {table}",
            "SELECT * FROM {table} ORDER BY id DESC LIMIT 10",
            "SELECT COUNT(DISTINCT {column}) FROM {table}",
        ]

        precomputed_results = {}
        total_patterns = len(common_patterns) * len(table_names)
        processed = 0

        for table_name in table_names:
            table_results = {}

            for pattern in common_patterns:
                try:
                    self.update_state(
                        state='PROCESSING',
                        meta={
                            'status': f'Processing {table_name}',
                            'progress': int((processed / total_patterns) * 100)
                        }
                    )

                    # This would execute the query and cache results
                    # For now, just simulate the process
                    query = pattern.format(table=table_name, column='id')
                    table_results[pattern] = {
                        'query': query,
                        'cached_at': datetime.now().isoformat(),
                        'status': 'cached'
                    }

                    processed += 1

                except Exception as e:
                    logger.warning(f"Failed to precompute query for {table_name}: {e}")
                    table_results[pattern] = {'error': str(e)}

            precomputed_results[table_name] = table_results

        result = {
            'precomputed_queries': precomputed_results,
            'total_patterns': total_patterns,
            'task_info': {
                'task_id': self.request.id,
                'completed_at': datetime.now().isoformat()
            }
        }

        logger.info("Common query precomputation completed")
        return result

    except Exception as exc:
        logger.error(f"Query precomputation failed: {exc}")
        return {
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


# Task monitoring and utility functions

def get_ai_task_stats():
    """Get statistics for AI tasks."""
    try:
        inspect = app.control.inspect()

        # Get active AI tasks
        active_tasks = inspect.active()
        ai_tasks = []

        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if task['name'].startswith('tasks.ai_tasks'):
                        ai_tasks.append({
                            'worker': worker,
                            'task_name': task['name'],
                            'task_id': task['id'],
                            'args': task.get('args', []),
                            'time_start': task.get('time_start')
                        })

        return {
            'active_ai_tasks': len(ai_tasks),
            'tasks': ai_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get AI task stats: {e}")
        return {'error': str(e)}


def cancel_ai_task(task_id: str):
    """Cancel a specific AI task."""
    try:
        app.control.revoke(task_id, terminate=True)
        logger.info(f"AI task {task_id} cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel AI task {task_id}: {e}")
        return False