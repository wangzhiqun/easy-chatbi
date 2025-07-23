"""
Configuration management module for ChatBI platform.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application settings
    app_name: str = "ChatBI"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "default-secret-key"

    # OpenAI settings
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"

    # Database settings
    database_url: str
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "chatbi_user"
    mysql_password: str = "chatbi_password"
    mysql_database: str = "chatbi_db"

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Milvus settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Celery settings
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Security settings
    jwt_secret_key: str = "jwt-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Streamlit settings
    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = 8501

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get database connection URL."""
    return settings.database_url


def get_redis_url() -> str:
    """Get Redis connection URL."""
    return settings.redis_url


def get_milvus_config() -> dict:
    """Get Milvus connection configuration."""
    return {
        "host": settings.milvus_host,
        "port": settings.milvus_port
    }


def get_openai_config() -> dict:
    """Get OpenAI configuration."""
    return {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model
    }