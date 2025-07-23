"""
SQL Executor for ChatBI platform.
Safely executes SQL queries with monitoring, logging, and resource limits.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
import pandas as pd

from utils.config import get_database_url
from utils.logger import get_logger
from utils.exceptions import DatabaseException, SQLSecurityException, ErrorCodes
from security.sql_guardian import SQLGuardian

logger = get_logger(__name__)


class SQLExecutor:
    """
    Safe SQL executor with built-in security checks, resource limits,
    and comprehensive logging for audit and monitoring.
    """

    def __init__(self):
        """Initialize SQL executor with security and monitoring."""
        self.engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False  # Don't log SQL queries for security
        )
        self.sql_guardian = SQLGuardian()

        # Execution limits
        self.max_execution_time = 30  # seconds
        self.max_result_rows = 10000
        self.default_limit = 100

        # Performance tracking
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_execution_time": 0
        }

    async def execute_query(
            self,
            sql_query: str,
            user_id: int,
            timeout_seconds: Optional[int] = None,
            max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query with comprehensive safety checks and monitoring.

        Args:
            sql_query: SQL query to execute
            user_id: ID of user executing the query
            timeout_seconds: Custom timeout for this query
            max_rows: Custom row limit for this query

        Returns:
            Dictionary containing execution results and metadata
        """
        start_time = time.time()
        execution_id = f"{user_id}_{int(start_time)}"

        logger.info(f"Executing SQL query {execution_id} for user {user_id}")

        try:
            # Update statistics
            self.execution_stats["total_queries"] += 1

            # Step 1: Security validation
            security_check = await self._validate_query_security(sql_query, user_id)
            if not security_check["is_safe"]:
                raise SQLSecurityException(
                    f"Query blocked by security validation: {security_check['reason']}",
                    ErrorCodes.SQL_INJECTION_DETECTED,
                    security_check
                )

            # Step 2: Query preparation and enhancement
            prepared_query = self._prepare_query(sql_query, max_rows or self.max_result_rows)

            # Step 3: Execute with timeout and monitoring
            result = await self._execute_with_monitoring(
                query=prepared_query,
                execution_id=execution_id,
                timeout=timeout_seconds or self.max_execution_time
            )

            # Step 4: Process and validate results
            processed_result = self._process_results(result, start_time, execution_id)

            # Step 5: Log successful execution
            await self._log_execution(
                execution_id=execution_id,
                user_id=user_id,
                sql_query=sql_query,
                result=processed_result,
                success=True
            )

            self.execution_stats["successful_queries"] += 1
            logger.info(f"Query {execution_id} executed successfully: {len(processed_result['data'])} rows")

            return processed_result

        except SQLSecurityException:
            # Re-raise security exceptions
            await self._log_execution(
                execution_id=execution_id,
                user_id=user_id,
                sql_query=sql_query,
                result=None,
                success=False,
                error="Security violation"
            )
            raise

        except Exception as e:
            # Handle all other exceptions
            self.execution_stats["failed_queries"] += 1
            execution_time = (time.time() - start_time) * 1000

            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time_ms": int(execution_time),
                "data": [],
                "metadata": {
                    "execution_id": execution_id,
                    "user_id": user_id,
                    "error_type": type(e).__name__
                }
            }

            await self._log_execution(
                execution_id=execution_id,
                user_id=user_id,
                sql_query=sql_query,
                result=error_result,
                success=False,
                error=str(e)
            )

            logger.error(f"Query {execution_id} failed: {e}")
            raise DatabaseException(
                f"Query execution failed: {str(e)}",
                ErrorCodes.DB_QUERY_ERROR,
                error_result
            )

    async def _validate_query_security(self, sql_query: str, user_id: int) -> Dict[str, Any]:
        """Validate query for security issues."""
        try:
            return await self.sql_guardian.validate_query(sql_query, user_id)
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {
                "is_safe": False,
                "reason": f"Security validation error: {str(e)}",
                "risk_level": "high"
            }

    def _prepare_query(self, sql_query: str, max_rows: int) -> str:
        """Prepare query by adding safety enhancements."""
        query = sql_query.strip()

        # Remove trailing semicolon if present
        if query.endswith(';'):
            query = query[:-1]

        # Add LIMIT if not present
        query_upper = query.upper()
        if 'LIMIT' not in query_upper:
            query += f" LIMIT {min(max_rows, self.max_result_rows)}"

        # Ensure it's a SELECT statement
        if not query_upper.strip().startswith('SELECT'):
            raise SQLSecurityException(
                "Only SELECT statements are allowed",
                ErrorCodes.SQL_UNAUTHORIZED_OPERATION
            )

        return query

    async def _execute_with_monitoring(
            self,
            query: str,
            execution_id: str,
            timeout: int
    ) -> List[Dict[str, Any]]:
        """Execute query with timeout and resource monitoring."""
        try:
            # Create async task for query execution
            task = asyncio.create_task(self._execute_query_async(query))

            # Wait for completion or timeout
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Cancel the task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                raise DatabaseException(
                    f"Query execution timed out after {timeout} seconds",
                    ErrorCodes.DB_QUERY_ERROR
                )

        except Exception as e:
            logger.error(f"Query execution monitoring failed for {execution_id}: {e}")
            raise

    async def _execute_query_async(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query asynchronously."""
        loop = asyncio.get_event_loop()

        def execute_sync():
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                # Convert to list of dictionaries
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]

        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(None, execute_sync)

    def _process_results(
            self,
            raw_result: List[Dict[str, Any]],
            start_time: float,
            execution_id: str
    ) -> Dict[str, Any]:
        """Process and format query results."""
        execution_time = (time.time() - start_time) * 1000

        # Update average execution time
        current_avg = self.execution_stats["avg_execution_time"]
        total_queries = self.execution_stats["total_queries"]
        self.execution_stats["avg_execution_time"] = (
                (current_avg * (total_queries - 1) + execution_time) / total_queries
        )

        # Process data types and clean results
        processed_data = self._clean_result_data(raw_result)

        return {
            "status": "success",
            "data": processed_data,
            "row_count": len(processed_data),
            "execution_time_ms": int(execution_time),
            "metadata": {
                "execution_id": execution_id,
                "columns": list(processed_data[0].keys()) if processed_data else [],
                "data_types": self._infer_data_types(processed_data),
                "timestamp": time.time()
            }
        }

    def _clean_result_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and format result data for JSON serialization."""
        if not raw_data:
            return []

        cleaned_data = []
        for row in raw_data:
            cleaned_row = {}
            for key, value in row.items():
                # Handle different data types for JSON serialization
                if pd.isna(value):
                    cleaned_row[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    cleaned_row[key] = str(value)
                elif hasattr(value, 'isoformat'):  # datetime objects
                    cleaned_row[key] = value.isoformat()
                elif isinstance(value, bytes):
                    cleaned_row[key] = value.decode('utf-8', errors='ignore')
                else:
                    cleaned_row[key] = value
            cleaned_data.append(cleaned_row)

        return cleaned_data

    def _infer_data_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Infer data types for result columns."""
        if not data:
            return {}

        data_types = {}
        sample_row = data[0]

        for column, value in sample_row.items():
            if value is None:
                # Check other rows for non-null values
                for row in data[1:6]:  # Check up to 5 more rows
                    if row.get(column) is not None:
                        value = row[column]
                        break

            if value is None:
                data_types[column] = "unknown"
            elif isinstance(value, bool):
                data_types[column] = "boolean"
            elif isinstance(value, int):
                data_types[column] = "integer"
            elif isinstance(value, float):
                data_types[column] = "float"
            elif isinstance(value, str):
                # Try to determine if it's a date string
                if self._looks_like_date(value):
                    data_types[column] = "datetime"
                else:
                    data_types[column] = "string"
            else:
                data_types[column] = str(type(value).__name__)

        return data_types

    def _looks_like_date(self, value: str) -> bool:
        """Simple heuristic to detect date strings."""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}/\d{2}/\d{2}'
        ]
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        return False

    async def _log_execution(
            self,
            execution_id: str,
            user_id: int,
            sql_query: str,
            result: Optional[Dict[str, Any]],
            success: bool,
            error: Optional[str] = None
    ):
        """Log query execution for audit and monitoring."""
        log_entry = {
            "execution_id": execution_id,
            "user_id": user_id,
            "timestamp": time.time(),
            "success": success,
            "query_hash": hash(sql_query),  # Don't log full query for security
            "result_rows": len(result["data"]) if result and "data" in result else 0,
            "execution_time_ms": result.get("execution_time_ms") if result else None,
            "error": error
        }

        # In a production system, this would write to an audit log
        logger.info(f"SQL execution logged: {log_entry}")

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring."""
        return {
            "statistics": self.execution_stats.copy(),
            "limits": {
                "max_execution_time": self.max_execution_time,
                "max_result_rows": self.max_result_rows,
                "default_limit": self.default_limit
            },
            "status": "operational"
        }

    async def validate_connection(self) -> bool:
        """Validate database connection health."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False