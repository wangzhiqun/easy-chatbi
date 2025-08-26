from typing import Any, Dict, List, Optional

import pandas as pd

from ai.agents import SQLAgent, ChartAgent, AnalysisAgent
from connectors import MySQLConnector, ConnectionConfig
from utils import logger, get_config


class MCPTools:

    def __init__(self):
        self.config = get_config()
        self.sql_agent = SQLAgent()
        self.chart_agent = ChartAgent()
        self.analysis_agent = AnalysisAgent()
        self._init_database_connector()
        logger.info("Initialized MCP Tools")

    def _init_database_connector(self):
        self.db_connector = MySQLConnector(
            ConnectionConfig(
                host=self.config.mysql_host,
                port=self.config.mysql_port,
                database=self.config.mysql_database,
                username=self.config.mysql_user,
                password=self.config.mysql_password
            )
        )

    async def query_database(self, query: str, database: Optional[str] = None) -> Dict[str, Any]:
        try:
            self.db_connector.connect()

            df = self.db_connector.execute_query(query)

            result = {
                'status': 'success',
                'row_count': len(df),
                'columns': df.columns.tolist(),
                'data': df.to_dict('records')[:100],
                'query': query
            }

            if len(df) > 100:
                result['message'] = f'Showing first 100 of {len(df)} rows'

            return result

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query
            }
        finally:
            self.db_connector.disconnect()

    async def generate_sql(self, question: str) -> Dict[str, Any]:
        try:
            self.db_connector.connect()
            schema_info = self.db_connector.get_schema()

            sql_query, explanation = self.sql_agent.generate_sql(
                question=question,
                schema_info=schema_info
            )

            return {
                'status': 'success',
                'sql': sql_query,
                'explanation': explanation,
                'question': question
            }

        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'question': question
            }
        finally:
            self.db_connector.disconnect()

    async def create_chart(
            self,
            data: List[Dict],
            chart_type: str = 'auto',
            options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        try:
            df = pd.DataFrame(data)

            if chart_type == 'auto':
                recommendation = self.chart_agent.recommend_chart(df)
                chart_type = recommendation['chart_type']

            config = self.chart_agent.generate_chart_config(
                df=df,
                chart_type=chart_type,
                **(options or {})
            )

            return {
                'status': 'success',
                'chart_type': chart_type,
                'config': config,
                'data_points': len(data)
            }

        except Exception as e:
            logger.error(f"Chart creation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def analyze_data(
            self,
            data: List[Dict],
            analysis_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        try:
            df = pd.DataFrame(data)

            if analysis_type == 'comprehensive':
                results = self.analysis_agent.analyze_data(df)
            elif analysis_type == 'correlation':
                results = self.analysis_agent.find_correlations(df)
            elif analysis_type == 'anomaly':
                results = self.analysis_agent.detect_anomalies(df)
            elif analysis_type == 'trend':
                results = self.analysis_agent.trend_analysis(df)
            else:
                results = self.analysis_agent.analyze_data(df, analysis_type)

            return {
                'status': 'success',
                'analysis_type': analysis_type,
                'results': results,
                'data_points': len(data)
            }

        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def validate_sql(self, sql_query: str) -> Dict[str, Any]:
        try:
            from ai.tools import ValidationTool

            is_safe = ValidationTool.validate_sql_safety(sql_query)

            is_valid, error = ValidationTool.validate_sql_syntax(sql_query)

            return {
                'status': 'success',
                'is_safe': is_safe,
                'is_valid': is_valid,
                'error': error,
                'query': sql_query
            }

        except Exception as e:
            logger.error(f"SQL validation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            self.db_connector.connect()

            schema = self.db_connector.get_schema()
            tables = schema.get('tables', {})

            if table_name not in tables:
                return {
                    'status': 'error',
                    'error': f'Table {table_name} not found'
                }

            table_info = tables[table_name]

            sample_df = self.db_connector.get_sample_data(table_name, limit=5)

            return {
                'status': 'success',
                'table_name': table_name,
                'columns': table_info.get('columns', []),
                'row_count': table_info.get('row_count', 0),
                'comment': table_info.get('comment', ''),
                'sample_data': sample_df.to_dict('records')
            }

        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            self.db_connector.disconnect()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'query_database',
                'description': 'Execute SQL query on database',
                'parameters': {
                    'query': 'SQL query string',
                    'database': 'Optional database name'
                }
            },
            {
                'name': 'generate_sql',
                'description': 'Generate SQL from natural language',
                'parameters': {
                    'question': 'Natural language question'
                }
            },
            # {
            #     'name': 'create_chart',
            #     'description': 'Create chart from data',
            #     'parameters': {
            #         'data': 'Array of data objects',
            #         'chart_type': 'Chart type (auto, line, bar, pie, etc.)',
            #         'options': 'Optional chart configuration'
            #     }
            # },
            {
                'name': 'analyze_data',
                'description': 'Perform data analysis',
                'parameters': {
                    'data': 'Array of data objects',
                    'analysis_type': 'Type of analysis (comprehensive, correlation, anomaly, trend)'
                }
            },
            # {
            #     'name': 'validate_sql',
            #     'description': 'Validate SQL query for safety and syntax',
            #     'parameters': {
            #         'sql_query': 'SQL query to validate'
            #     }
            # },
            {
                'name': 'get_table_info',
                'description': 'Get information about a database table',
                'parameters': {
                    'table_name': 'Name of the table'
                }
            }
        ]
