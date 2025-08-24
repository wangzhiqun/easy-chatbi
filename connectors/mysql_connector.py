from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd
import pymysql

from utils import logger, DatabaseError
from .base_connector import BaseConnector


class MySQLConnector(BaseConnector):

    def connect(self) -> None:
        try:
            self.connection = pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                **self.config.options
            )
            self._is_connected = True
            logger.info(f"Connected to MySQL database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise DatabaseError(f"MySQL connection failed: {str(e)}")

    def disconnect(self) -> None:
        if self.connection:
            try:
                self.connection.close()
                self._is_connected = False
                logger.info("Disconnected from MySQL database")
            except Exception as e:
                logger.error(f"Error disconnecting from MySQL: {str(e)}")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not self._is_connected:
            self.connect()

        if not self.validate_query(query):
            raise DatabaseError("Query validation failed - potentially unsafe operation")

        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                results = cursor.fetchall()

                if results:
                    df = pd.DataFrame(results)
                else:
                    df = pd.DataFrame()

                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        if not self._is_connected:
            self.connect()

        schema = {
            'database': self.config.database,
            'tables': {}
        }

        try:
            tables_query = """
                SELECT TABLE_NAME, TABLE_COMMENT, TABLE_ROWS
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s
            """

            with self.connection.cursor() as cursor:
                cursor.execute(tables_query, (self.config.database,))
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table['TABLE_NAME']

                    columns_query = """
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, 
                               COLUMN_KEY, COLUMN_COMMENT, COLUMN_DEFAULT
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                        ORDER BY ORDINAL_POSITION
                    """

                    cursor.execute(columns_query, (self.config.database, table_name))
                    columns = cursor.fetchall()

                    schema['tables'][table_name] = {
                        'comment': table['TABLE_COMMENT'],
                        'row_count': table['TABLE_ROWS'],
                        'columns': columns
                    }

            logger.info(f"Retrieved schema for {len(schema['tables'])} tables")
            return schema

        except Exception as e:
            logger.error(f"Failed to get schema: {str(e)}")
            raise DatabaseError(f"Failed to get schema: {str(e)}")

    def test_connection(self) -> bool:
        try:
            if not self._is_connected:
                self.connect()

            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None

        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    @contextmanager
    def transaction(self):
        if not self._is_connected:
            self.connect()

        try:
            self.connection.begin()
            yield self
            self.connection.commit()
            logger.info("Transaction committed successfully")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction rolled back: {str(e)}")
            raise DatabaseError(f"Transaction failed: {str(e)}")

    def get_sample_data(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        return self.execute_query(query)
