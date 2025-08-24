from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ai.tools import ValidationTool
from utils import logger


@dataclass
class ConnectionConfig:
    host: str
    port: int
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    options: Dict[str, Any] = None

    def __post_init__(self):
        self.options = self.options or {}


class BaseConnector(ABC):

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self._is_connected = False
        self.validator = ValidationTool()
        logger.info(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def get_tables(self) -> List[str]:
        schema = self.get_schema()
        return list(schema.get('tables', {}).keys())

    def get_columns(self, table_name: str) -> List[Dict[str, str]]:
        schema = self.get_schema()
        tables = schema.get('tables', {})
        if table_name in tables:
            return tables[table_name].get('columns', [])
        return []

    def validate_query(self, query: str) -> bool:
        return self.validator.validate_sql_syntax(query)[0]
