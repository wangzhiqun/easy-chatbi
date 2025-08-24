import io
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ai.tools import ValidationTool
from connectors import MySQLConnector, ConnectionConfig
from utils import logger, get_config, DatabaseError


class DataService:

    def __init__(self):
        self.config = get_config()
        self.connector = None
        self.validator = ValidationTool()
        self._init_connector()
        logger.info("Initialized Data Service")

    def _init_connector(self):
        self.connector = MySQLConnector(
            ConnectionConfig(
                host=self.config.mysql_host,
                port=self.config.mysql_port,
                database=self.config.mysql_database,
                username=self.config.mysql_user,
                password=self.config.mysql_password
            )
        )

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if not self.validator.validate_sql_safety(query):
                raise DatabaseError("Query contains potentially dangerous operations")

            self.connector.connect()
            df = self.connector.execute_query(query, params)

            result = {
                'status': 'success',
                'row_count': len(df),
                'columns': df.columns.tolist(),
                'data': df.to_dict('records'),
                'query': query
            }

            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query
            }
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    async def get_schema(self) -> Dict[str, Any]:
        try:
            self.connector.connect()
            schema = self.connector.get_schema()
            logger.info(f"Retrieved schema for {len(schema.get('tables', {}))} tables")
            return schema

        except Exception as e:
            logger.error(f"Failed to get schema: {str(e)}")
            raise DatabaseError(f"Failed to get schema: {str(e)}")
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            self.connector.connect()

            schema = self.connector.get_schema()
            tables = schema.get('tables', {})

            if table_name not in tables:
                raise DatabaseError(f"Table {table_name} not found")

            table_info = tables[table_name]

            sample_df = self.connector.get_sample_data(table_name, limit=10)

            stats_query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT {table_info['columns'][0]['COLUMN_NAME']}) as distinct_first_col
                FROM `{table_name}`
            """
            stats_df = self.connector.execute_query(stats_query)

            return {
                'table_name': table_name,
                'columns': table_info.get('columns', []),
                'comment': table_info.get('comment', ''),
                'row_count': table_info.get('row_count', 0),
                'sample_data': sample_df.to_dict('records'),
                'statistics': stats_df.to_dict('records')[0] if not stats_df.empty else {}
            }

        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            raise DatabaseError(f"Failed to get table info: {str(e)}")
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    async def import_data(
            self,
            table_name: str,
            data: pd.DataFrame,
            if_exists: str = 'append'
    ) -> Dict[str, Any]:
        try:
            self.connector.connect()

            engine = create_engine(self.config.mysql_url)

            rows_before = self._get_row_count(table_name)

            data.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False
            )

            rows_after = self._get_row_count(table_name)
            rows_imported = rows_after - rows_before

            logger.info(f"Imported {rows_imported} rows into {table_name}")

            return {
                'status': 'success',
                'table_name': table_name,
                'rows_imported': rows_imported,
                'total_rows': rows_after
            }

        except Exception as e:
            logger.error(f"Data import failed: {str(e)}")
            raise DatabaseError(f"Data import failed: {str(e)}")
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    async def export_data(
            self,
            query: str,
            format: str = 'csv'
    ) -> Dict[str, Any]:
        try:
            result = await self.execute_query(query)

            if result['status'] != 'success':
                raise DatabaseError(f"Query failed: {result.get('error')}")

            df = pd.DataFrame(result['data'])

            if format == 'csv':
                output = df.to_csv(index=False)
                content_type = 'text/csv'
            elif format == 'json':
                output = df.to_json(orient='records')
                content_type = 'application/json'
            elif format == 'excel':
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                output = buffer.getvalue()
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported {len(df)} rows in {format} format")

            return {
                'status': 'success',
                'format': format,
                'content_type': content_type,
                'data': output,
                'row_count': len(df)
            }

        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            raise DatabaseError(f"Data export failed: {str(e)}")

    async def create_view(self, view_name: str, query: str) -> Dict[str, Any]:
        try:
            if not self.validator.validate_sql_safety(query):
                raise DatabaseError("Query contains potentially dangerous operations")

            self.connector.connect()

            create_view_query = f"CREATE OR REPLACE VIEW `{view_name}` AS {query}"
            self.connector.execute_query(create_view_query)

            logger.info(f"Created view: {view_name}")

            return {
                'status': 'success',
                'view_name': view_name,
                'query': query
            }

        except Exception as e:
            logger.error(f"Failed to create view: {str(e)}")
            raise DatabaseError(f"Failed to create view: {str(e)}")
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    async def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        pass

    async def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        try:
            self.connector.connect()

            explain_query = f"EXPLAIN {query}"
            explain_df = self.connector.execute_query(explain_query)

            analysis = {
                'query': query,
                'execution_plan': explain_df.to_dict('records'),
                'recommendations': []
            }

            explain_text = str(explain_df.to_dict())

            if 'filesort' in explain_text.lower():
                analysis['recommendations'].append('Consider adding an index to avoid filesort')

            if 'temporary' in explain_text.lower():
                analysis['recommendations'].append('Query uses temporary table, consider optimization')

            if 'full table scan' in explain_text.lower() or 'all' in explain_text.lower():
                analysis['recommendations'].append('Full table scan detected, consider adding WHERE clause or index')

            logger.info(f"Analyzed query performance: {len(analysis['recommendations'])} recommendations")

            return analysis

        except Exception as e:
            logger.error(f"Query performance analysis failed: {str(e)}")
            raise DatabaseError(f"Query performance analysis failed: {str(e)}")
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    def _get_row_count(self, table_name: str) -> int:
        try:
            query = f"SELECT COUNT(*) as count FROM `{table_name}`"
            df = self.connector.execute_query(query)
            return df.iloc[0]['count'] if not df.empty else 0
        except:
            return 0

    async def test_connection(self) -> bool:
        try:
            self.connector.connect()
            result = self.connector.test_connection()
            logger.info(f"Database connection test: {'Success' if result else 'Failed'}")
            return result
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
        finally:
            if self.connector.is_connected:
                self.connector.disconnect()

    def clean_numpy_types(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: self.clean_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.clean_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self.clean_numpy_types(obj.tolist())
        elif isinstance(obj, (pd.Series, pd.Index)):
            return self.clean_numpy_types(obj.tolist())
        elif isinstance(obj, pd.DataFrame):
            return self.clean_numpy_types(obj.to_dict('records'))
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
