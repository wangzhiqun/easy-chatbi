import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Database Configuration
    mysql_host: str = Field(default=os.getenv("MYSQL_HOST", "localhost"))
    mysql_port: int = Field(default=int(os.getenv("MYSQL_PORT", "3306")))
    mysql_database: str = Field(default=os.getenv("MYSQL_DATABASE", "chatbi"))
    mysql_user: str = Field(default=os.getenv("MYSQL_USER", "admin"))
    mysql_password: str = Field(default=os.getenv("MYSQL_PASSWORD", "admin"))
    mysql_root_password: str = Field(default=os.getenv("MYSQL_ROOT_PASSWORD", "root"))

    # Milvus Configuration
    milvus_host: str = Field(default=os.getenv("MILVUS_HOST", "localhost"))
    milvus_port: int = Field(default=int(os.getenv("MILVUS_PORT", "19530")))

    # Redis Configuration
    redis_host: str = Field(default=os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default=int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = Field(default=int(os.getenv("REDIS_DB", "0")))

    # API Configuration
    api_host: str = Field(default=os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default=int(os.getenv("API_PORT", "8000")))
    api_prefix: str = Field(default=os.getenv("API_PREFIX", "/api/v1"))

    # Security
    secret_key: str = Field(default=os.getenv("SECRET_KEY", "your-secret-key"))
    algorithm: str = Field(default=os.getenv("ALGORITHM", "HS256"))
    access_token_expire_minutes: int = Field(
        default=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    )

    # MCP Configuration
    mcp_server_name: str = Field(default=os.getenv("MCP_SERVER_NAME", "chatbi-mcp"))
    mcp_server_version: str = Field(default=os.getenv("MCP_SERVER_VERSION", "1.0.0"))

    # Logging
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = Field(default=os.getenv("LOG_FILE", "logs/chatbi.log"))

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def load_yaml_config(self, config_path: str = "data/config.yaml") -> Dict[str, Any]:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_config() -> Config:
    return Config()
