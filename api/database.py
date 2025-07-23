"""
Database connection and session management for ChatBI platform.
Handles SQLAlchemy configuration and connection pooling.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import redis

from utils.config import settings, get_database_url, get_redis_url
from utils.logger import get_logger
from utils.exceptions import DatabaseException

logger = get_logger(__name__)

# SQLAlchemy setup
engine = create_engine(
    get_database_url(),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(get_redis_url(), decode_responses=True)

# Metadata for reflection
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Used with FastAPI dependency injection.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise DatabaseException(f"Database operation failed: {str(e)}")
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Used for manual session management.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction error: {e}")
        raise DatabaseException(f"Database transaction failed: {str(e)}")
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    return redis_client


def test_database_connection() -> bool:
    """Test database connectivity."""
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def test_redis_connection() -> bool:
    """Test Redis connectivity."""
    try:
        redis_client.ping()
        logger.info("Redis connection test successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise DatabaseException(f"Table creation failed: {str(e)}")


def get_table_names() -> list:
    """Get list of all table names in the database."""
    try:
        metadata.reflect(bind=engine)
        return list(metadata.tables.keys())
    except Exception as e:
        logger.error(f"Failed to get table names: {e}")
        raise DatabaseException(f"Failed to retrieve table names: {str(e)}")


def get_table_schema(table_name: str) -> dict:
    """Get schema information for a specific table."""
    try:
        metadata.reflect(bind=engine)
        if table_name not in metadata.tables:
            raise DatabaseException(f"Table '{table_name}' not found")

        table = metadata.tables[table_name]
        schema = {
            "name": table_name,
            "columns": []
        }

        for column in table.columns:
            schema["columns"].append({
                "name": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "default": str(column.default) if column.default else None
            })

        return schema
    except Exception as e:
        logger.error(f"Failed to get table schema for {table_name}: {e}")
        raise DatabaseException(f"Failed to retrieve table schema: {str(e)}")


# Health check functions
async def health_check() -> dict:
    """Check health of database and Redis connections."""
    return {
        "database": test_database_connection(),
        "redis": test_redis_connection()
    }