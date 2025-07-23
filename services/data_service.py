"""
Data Service for ChatBI platform.
Handles data retrieval, transformation, and export operations.
"""

import pandas as pd
import json
import csv
import io
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sqlalchemy import create_engine, text
from fastapi.responses import StreamingResponse

from utils.config import get_database_url
from utils.logger import get_logger
from utils.exceptions import DatabaseException, DataProcessingException, ErrorCodes
from .cache_service import CacheService

logger = get_logger(__name__)


class DataService:
    """
    Service for handling data operations including retrieval,
    transformation, caching, and export functionality.
    """

    def __init__(self):
        """Initialize data service with database connection and cache."""
        self.engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_recycle=300
        )
        self.cache_service = CacheService()

        # Configuration
        self.max_sample_size = 1000
        self.default_cache_ttl = 3600  # 1 hour
        self.export_row_limit = 50000

        # Data type mapping for better pandas integration
        self.sql_to_pandas_types = {
            'VARCHAR': 'string',
            'TEXT': 'string',
            'INT': 'int64',
            'INTEGER': 'int64',
            'BIGINT': 'int64',
            'FLOAT': 'float64',
            'DOUBLE': 'float64',
            'DECIMAL': 'float64',
            'BOOLEAN': 'boolean',
            'DATE': 'datetime64[ns]',
            'DATETIME': 'datetime64[ns]',
            'TIMESTAMP': 'datetime64[ns]'
        }

    async def get_sample_data(
            self,
            table_name: str,
            limit: int = 10,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get sample data from a table.

        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            use_cache: Whether to use cached results

        Returns:
            List of sample records
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = f"sample_{table_name}_{limit}"
                cached_data = await self.cache_service.get(cache_key)
                if cached_data:
                    return cached_data

            # Validate table name for security
            if not self._is_valid_table_name(table_name):
                raise DatabaseException(
                    f"Invalid table name: {table_name}",
                    ErrorCodes.DB_QUERY_ERROR
                )

            # Execute query
            query = f"SELECT * FROM {table_name} LIMIT {min(limit, self.max_sample_size)}"

            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()

                # Convert to list of dictionaries
                sample_data = [dict(zip(columns, row)) for row in rows]

                # Process data types
                sample_data = self._process_data_types(sample_data)

                # Cache the result
                if use_cache:
                    await self.cache_service.set(cache_key, sample_data, self.default_cache_ttl)

                logger.info(f"Retrieved {len(sample_data)} sample records from {table_name}")
                return sample_data

        except Exception as e:
            logger.error(f"Failed to get sample data from {table_name}: {e}")
            raise DatabaseException(
                f"Failed to retrieve sample data: {str(e)}",
                ErrorCodes.DB_QUERY_ERROR
            )

    async def get_table_data(
            self,
            table_name: str,
            where_clause: Optional[str] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get data from a table with optional filtering and ordering.

        Args:
            table_name: Name of the table
            where_clause: WHERE clause for filtering
            order_by: ORDER BY clause for sorting
            limit: Maximum number of rows
            columns: Specific columns to select

        Returns:
            List of records
        """
        try:
            # Validate inputs
            if not self._is_valid_table_name(table_name):
                raise DatabaseException(
                    f"Invalid table name: {table_name}",
                    ErrorCodes.DB_QUERY_ERROR
                )

            # Build query
            if columns:
                column_list = ", ".join(self._validate_column_names(columns))
            else:
                column_list = "*"

            query = f"SELECT {column_list} FROM {table_name}"

            if where_clause:
                # Basic validation of WHERE clause
                if self._is_safe_where_clause(where_clause):
                    query += f" WHERE {where_clause}"
                else:
                    raise DatabaseException(
                        "Unsafe WHERE clause detected",
                        ErrorCodes.SQL_INJECTION_DETECTED
                    )

            if order_by:
                query += f" ORDER BY {order_by}"

            if limit:
                query += f" LIMIT {min(limit, self.export_row_limit)}"

            # Execute query
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()

                # Convert to list of dictionaries
                data = [dict(zip(columns, row)) for row in rows]
                data = self._process_data_types(data)

                logger.info(f"Retrieved {len(data)} records from {table_name}")
                return data

        except Exception as e:
            logger.error(f"Failed to get table data from {table_name}: {e}")
            raise DatabaseException(
                f"Failed to retrieve table data: {str(e)}",
                ErrorCodes.DB_QUERY_ERROR
            )

    async def get_column_statistics(
            self,
            table_name: str,
            column_name: str
    ) -> Dict[str, Any]:
        """
        Get statistical information about a column.

        Args:
            table_name: Name of the table
            column_name: Name of the column

        Returns:
            Dictionary containing column statistics
        """
        try:
            # Validate inputs
            if not self._is_valid_table_name(table_name) or not self._is_valid_column_name(column_name):
                raise DatabaseException(
                    "Invalid table or column name",
                    ErrorCodes.DB_QUERY_ERROR
                )

            # Build statistics query
            stats_query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT({column_name}) as non_null_count,
                COUNT(DISTINCT {column_name}) as unique_count,
                MIN({column_name}) as min_value,
                MAX({column_name}) as max_value
            FROM {table_name}
            """

            with self.engine.connect() as connection:
                result = connection.execute(text(stats_query))
                row = result.fetchone()

                stats = {
                    "table_name": table_name,
                    "column_name": column_name,
                    "total_count": row[0],
                    "non_null_count": row[1],
                    "unique_count": row[2],
                    "min_value": row[3],
                    "max_value": row[4],
                    "null_count": row[0] - row[1],
                    "null_percentage": ((row[0] - row[1]) / row[0] * 100) if row[0] > 0 else 0,
                    "uniqueness_ratio": (row[2] / row[1]) if row[1] > 0 else 0
                }

                # Try to get additional numeric statistics
                try:
                    numeric_query = f"""
                    SELECT 
                        AVG(CAST({column_name} AS FLOAT)) as avg_value,
                        STDDEV(CAST({column_name} AS FLOAT)) as std_dev
                    FROM {table_name}
                    WHERE {column_name} IS NOT NULL
                    """

                    numeric_result = connection.execute(text(numeric_query))
                    numeric_row = numeric_result.fetchone()

                    if numeric_row[0] is not None:
                        stats["avg_value"] = float(numeric_row[0])
                        stats["std_dev"] = float(numeric_row[1]) if numeric_row[1] else 0

                except Exception:
                    # Column is not numeric, skip numeric stats
                    pass

                return stats

        except Exception as e:
            logger.error(f"Failed to get column statistics: {e}")
            raise DatabaseException(
                f"Failed to retrieve column statistics: {str(e)}",
                ErrorCodes.DB_QUERY_ERROR
            )

    async def get_table_summary(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a table.

        Args:
            table_name: Name of the table

        Returns:
            Table summary with metadata and statistics
        """
        try:
            if not self._is_valid_table_name(table_name):
                raise DatabaseException(
                    f"Invalid table name: {table_name}",
                    ErrorCodes.DB_QUERY_ERROR
                )

            # Get basic table info
            info_query = f"""
            SELECT COUNT(*) as row_count
            FROM {table_name}
            """

            with self.engine.connect() as connection:
                result = connection.execute(text(info_query))
                row_count = result.fetchone()[0]

                # Get column information
                columns_query = f"SELECT * FROM {table_name} LIMIT 1"
                columns_result = connection.execute(text(columns_query))
                column_names = list(columns_result.keys())

                summary = {
                    "table_name": table_name,
                    "row_count": row_count,
                    "column_count": len(column_names),
                    "columns": column_names,
                    "last_analyzed": datetime.now().isoformat(),
                    "size_category": self._categorize_table_size(row_count)
                }

                # Get sample data for data type inference
                if row_count > 0:
                    sample_data = await self.get_sample_data(table_name, limit=5, use_cache=False)
                    summary["sample_data"] = sample_data
                    summary["data_types"] = self._infer_data_types(sample_data)

                return summary

        except Exception as e:
            logger.error(f"Failed to get table summary for {table_name}: {e}")
            raise DatabaseException(
                f"Failed to retrieve table summary: {str(e)}",
                ErrorCodes.DB_QUERY_ERROR
            )

    def export_to_csv(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> StreamingResponse:
        """
        Export data to CSV format.

        Args:
            data: Data to export
            filename: Optional filename for download

        Returns:
            StreamingResponse with CSV data
        """
        try:
            if not data:
                raise DataProcessingException(
                    "No data to export",
                    ErrorCodes.DATA_VALIDATION_ERROR
                )

            # Convert to DataFrame for easier CSV handling
            df = pd.DataFrame(data)

            # Create CSV string
            output = io.StringIO()
            df.to_csv(output, index=False)
            csv_data = output.getvalue()

            # Create streaming response
            def generate():
                yield csv_data

            filename = filename or f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            return StreamingResponse(
                generate(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise DataProcessingException(
                f"Failed to export CSV: {str(e)}",
                ErrorCodes.DATA_TRANSFORMATION_ERROR
            )

    def export_to_json(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> StreamingResponse:
        """
        Export data to JSON format.

        Args:
            data: Data to export
            filename: Optional filename for download

        Returns:
            StreamingResponse with JSON data
        """
        try:
            if not data:
                raise DataProcessingException(
                    "No data to export",
                    ErrorCodes.DATA_VALIDATION_ERROR
                )

            # Create JSON string
            json_data = json.dumps(data, indent=2, default=str)

            def generate():
                yield json_data

            filename = filename or f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            return StreamingResponse(
                generate(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise DataProcessingException(
                f"Failed to export JSON: {str(e)}",
                ErrorCodes.DATA_TRANSFORMATION_ERROR
            )

    def export_to_excel(
            self,
            data: List[Dict[str, Any]],
            filename: Optional[str] = None,
            sheet_name: str = "Data"
    ) -> StreamingResponse:
        """
        Export data to Excel format.

        Args:
            data: Data to export
            filename: Optional filename for download
            sheet_name: Name of the Excel sheet

        Returns:
            StreamingResponse with Excel data
        """
        try:
            if not data:
                raise DataProcessingException(
                    "No data to export",
                    ErrorCodes.DATA_VALIDATION_ERROR
                )

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Create Excel file in memory
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            output.seek(0)

            def generate():
                yield output.read()

            filename = filename or f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            return StreamingResponse(
                generate(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            raise DataProcessingException(
                f"Failed to export Excel: {str(e)}",
                ErrorCodes.DATA_TRANSFORMATION_ERROR
            )

    async def cache_table_metadata(self, table_name: str, force_refresh: bool = False):
        """
        Cache table metadata for improved performance.

        Args:
            table_name: Name of the table
            force_refresh: Whether to force refresh of cache
        """
        try:
            cache_key = f"metadata_{table_name}"

            if not force_refresh:
                existing = await self.cache_service.get(cache_key)
                if existing:
                    return

            # Get table summary and cache it
            summary = await self.get_table_summary(table_name)
            await self.cache_service.set(cache_key, summary, ttl=86400)  # Cache for 24 hours

            logger.info(f"Cached metadata for table {table_name}")

        except Exception as e:
            logger.warning(f"Failed to cache metadata for {table_name}: {e}")

    async def get_data_freshness(self, table_name: str) -> Dict[str, Any]:
        """
        Check data freshness of a table.

        Args:
            table_name: Name of the table

        Returns:
            Information about data freshness
        """
        try:
            if not self._is_valid_table_name(table_name):
                raise DatabaseException(
                    f"Invalid table name: {table_name}",
                    ErrorCodes.DB_QUERY_ERROR
                )

            # Look for timestamp columns
            timestamp_columns = []
            sample_data = await self.get_sample_data(table_name, limit=1, use_cache=False)

            if sample_data:
                for column, value in sample_data[0].items():
                    column_lower = column.lower()
                    if any(keyword in column_lower for keyword in
                           ['created', 'updated', 'modified', 'timestamp', 'date']):
                        timestamp_columns.append(column)

            freshness_info = {
                "table_name": table_name,
                "timestamp_columns": timestamp_columns,
                "last_checked": datetime.now().isoformat()
            }

            # Get latest timestamp if available
            if timestamp_columns:
                latest_query = f"SELECT MAX({timestamp_columns[0]}) as latest_timestamp FROM {table_name}"

                with self.engine.connect() as connection:
                    result = connection.execute(text(latest_query))
                    latest_timestamp = result.fetchone()[0]

                    if latest_timestamp:
                        freshness_info["latest_record"] = str(latest_timestamp)

                        # Calculate age
                        if isinstance(latest_timestamp, datetime):
                            age = datetime.now() - latest_timestamp
                            freshness_info["data_age_hours"] = age.total_seconds() / 3600

            return freshness_info

        except Exception as e:
            logger.error(f"Failed to check data freshness for {table_name}: {e}")
            return {"error": str(e)}

    # Private helper methods

    def _is_valid_table_name(self, table_name: str) -> bool:
        """Validate table name for security."""
        import re
        # Allow alphanumeric, underscore, and dots for schema.table notation
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$', table_name))

    def _is_valid_column_name(self, column_name: str) -> bool:
        """Validate column name for security."""
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name))

    def _validate_column_names(self, columns: List[str]) -> List[str]:
        """Validate list of column names."""
        validated = []
        for col in columns:
            if self._is_valid_column_name(col):
                validated.append(col)
            else:
                raise DatabaseException(
                    f"Invalid column name: {col}",
                    ErrorCodes.DB_QUERY_ERROR
                )
        return validated

    def _is_safe_where_clause(self, where_clause: str) -> bool:
        """Basic validation of WHERE clause for safety."""
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'EXEC', 'EXECUTE']
        clause_upper = where_clause.upper()

        return not any(keyword in clause_upper for keyword in dangerous_keywords)

    def _process_data_types(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean data types for JSON serialization."""
        if not data:
            return data

        processed_data = []

        for row in data:
            processed_row = {}
            for key, value in row.items():
                # Handle datetime objects
                if isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                # Handle None values
                elif value is None:
                    processed_row[key] = None
                # Handle bytes
                elif isinstance(value, bytes):
                    processed_row[key] = value.decode('utf-8', errors='ignore')
                else:
                    processed_row[key] = value

            processed_data.append(processed_row)

        return processed_data

    def _infer_data_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Infer data types from sample data."""
        if not data:
            return {}

        data_types = {}
        sample_row = data[0]

        for column, value in sample_row.items():
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
                try:
                    pd.to_datetime(value)
                    data_types[column] = "datetime"
                except:
                    data_types[column] = "string"
            else:
                data_types[column] = str(type(value).__name__)

        return data_types

    def _categorize_table_size(self, row_count: int) -> str:
        """Categorize table size based on row count."""
        if row_count < 1000:
            return "small"
        elif row_count < 100000:
            return "medium"
        elif row_count < 1000000:
            return "large"
        else:
            return "very_large"

    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of the data service."""
        try:
            # Test database connection
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return {
                "status": "healthy",
                "database_connection": "active",
                "cache_service": "available",
                "configuration": {
                    "max_sample_size": self.max_sample_size,
                    "export_row_limit": self.export_row_limit,
                    "default_cache_ttl": self.default_cache_ttl
                }
            }

        except Exception as e:
            logger.error(f"Data service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }