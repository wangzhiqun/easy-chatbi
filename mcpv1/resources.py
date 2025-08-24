from datetime import datetime
from typing import Any, Dict, List, Optional

from connectors import MySQLConnector, ConnectionConfig
from utils import logger, get_config


class MCPResources:

    def __init__(self):
        self.config = get_config()
        self._cache = {}
        self._init_database_connector()
        logger.info("Initialized MCP Resources")

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

    async def get_database_schema(self) -> Dict[str, Any]:
        try:
            self.db_connector.connect()
            schema = self.db_connector.get_schema()

            schema['retrieved_at'] = datetime.now().isoformat()
            schema['version'] = '1.0.0'

            self._cache['schema'] = schema

            return schema

        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            return {
                'error': str(e),
                'cached': self._cache.get('schema') is not None
            }
        finally:
            self.db_connector.disconnect()

    async def get_table_list(self) -> List[str]:
        try:
            self.db_connector.connect()
            tables = self.db_connector.get_tables()

            self._cache['tables'] = tables

            return tables

        except Exception as e:
            logger.error(f"Failed to get table list: {str(e)}")
            return self._cache.get('tables', [])
        finally:
            self.db_connector.disconnect()

    async def get_query_history(self) -> List[Dict[str, Any]]:

        return [
            {
                'id': 1,
                'query': 'SELECT COUNT(*) FROM users',
                'executed_at': None,
                'execution_time': 0.023,
                'row_count': 1,
                'status': 'success'
            }
        ]

    async def get_analysis_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                'id': 'sales_analysis',
                'name': 'Sales Analysis',
                'description': 'Comprehensive sales performance analysis',
                'queries': [
                    'SELECT DATE(created_at) as date, SUM(amount) as total FROM orders GROUP BY DATE(created_at)',
                    'SELECT product_id, COUNT(*) as count FROM order_items GROUP BY product_id ORDER BY count DESC LIMIT 10'
                ],
                'charts': ['line', 'bar'],
                'metrics': ['total_revenue', 'order_count', 'avg_order_value']
            },
            {
                'id': 'user_engagement',
                'name': 'User Engagement',
                'description': 'User activity and engagement metrics',
                'queries': [
                    'SELECT DATE(last_login) as date, COUNT(*) as active_users FROM users GROUP BY DATE(last_login)',
                    'SELECT COUNT(*) as new_users FROM users WHERE created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)'
                ],
                'charts': ['line', 'pie'],
                'metrics': ['dau', 'mau', 'retention_rate']
            }
        ]

    async def get_chart_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                'id': 'time_series',
                'name': 'Time Series',
                'chart_type': 'line',
                'description': 'Display data over time',
                'required_columns': ['date', 'value'],
                'options': {
                    'x_axis': 'date',
                    'y_axis': 'value',
                    'show_trend': True
                }
            },
            {
                'id': 'category_comparison',
                'name': 'Category Comparison',
                'chart_type': 'bar',
                'description': 'Compare values across categories',
                'required_columns': ['category', 'value'],
                'options': {
                    'x_axis': 'category',
                    'y_axis': 'value',
                    'orientation': 'vertical'
                }
            },
            {
                'id': 'distribution',
                'name': 'Distribution',
                'chart_type': 'pie',
                'description': 'Show distribution of values',
                'required_columns': ['label', 'value'],
                'options': {
                    'show_percentages': True,
                    'show_legend': True
                }
            }
        ]

    async def get_data_dictionary(self) -> Dict[str, Any]:
        return {
            'users': {
                'id': 'Unique user identifier',
                'email': 'User email address',
                'name': 'User full name',
                'created_at': 'Account creation timestamp',
                'last_login': 'Last login timestamp',
                'status': 'Account status (active, inactive, suspended)'
            },
            'orders': {
                'id': 'Unique order identifier',
                'user_id': 'Reference to users.id',
                'amount': 'Total order amount',
                'status': 'Order status (pending, completed, cancelled)',
                'created_at': 'Order creation timestamp'
            },
            'products': {
                'id': 'Unique product identifier',
                'name': 'Product name',
                'price': 'Product price',
                'category': 'Product category',
                'stock': 'Available stock quantity'
            }
        }

    async def get_cached_results(self, cache_key: Optional[str] = None) -> Dict[str, Any]:
        if cache_key:
            return self._cache.get(cache_key, {})

        cached_items = []
        for key, value in self._cache.items():
            cached_items.append({
                'key': key,
                'type': type(value).__name__,
                'size': len(str(value))
            })

        return {
            'cached_items': cached_items,
            'total_items': len(cached_items)
        }

    def cache_result(self, key: str, value: Any) -> bool:
        try:
            self._cache[key] = value
            logger.info(f"Cached result with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache result: {str(e)}")
            return False

    def clear_cache(self, key: Optional[str] = None) -> bool:
        try:
            if key:
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"Cleared cache key: {key}")
            else:
                self._cache.clear()
                logger.info("Cleared all cache")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    def get_resource_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                'uri': 'db://schema',
                'name': 'Database Schema',
                'description': 'Complete database schema information',
                'mimeType': 'application/json'
            },
            {
                'uri': 'db://tables',
                'name': 'Table List',
                'description': 'List of all available tables',
                'mimeType': 'application/json'
            },
            {
                'uri': 'history://queries',
                'name': 'Query History',
                'description': 'History of executed queries',
                'mimeType': 'application/json'
            },
            {
                'uri': 'templates://analysis',
                'name': 'Analysis Templates',
                'description': 'Predefined analysis templates',
                'mimeType': 'application/json'
            },
            {
                'uri': 'templates://charts',
                'name': 'Chart Templates',
                'description': 'Predefined chart templates',
                'mimeType': 'application/json'
            },
            {
                'uri': 'metadata://dictionary',
                'name': 'Data Dictionary',
                'description': 'Column descriptions and metadata',
                'mimeType': 'application/json'
            },
            {
                'uri': 'cache://results',
                'name': 'Cached Results',
                'description': 'Cached query and analysis results',
                'mimeType': 'application/json'
            }
        ]
