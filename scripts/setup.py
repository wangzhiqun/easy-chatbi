#!/usr/bin/env python3
"""
Environment setup script for ChatBI platform
Sets up database, vector store, and initial configuration
"""

import os
import sys
import logging
import asyncio
import pymysql
import yaml
from pathlib import Path
from typing import Dict, Any
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


class DatabaseSetup:
    """Handle MySQL database setup"""

    def __init__(self, config: Config):
        self.config = config
        self.connection = None

    def connect(self):
        """Connect to MySQL server"""
        try:
            self.connection = pymysql.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                charset='utf8mb4'
            )
            logger.info("Connected to MySQL server")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise

    def create_database(self):
        """Create database if not exists"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.DB_NAME}")
                cursor.execute(f"USE {self.config.DB_NAME}")
                self.connection.commit()
            logger.info(f"Database {self.config.DB_NAME} created/verified")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise

    def create_tables(self):
        """Create required tables"""
        tables = {
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role ENUM('admin', 'analyst', 'viewer', 'guest') DEFAULT 'guest',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    INDEX idx_username (username),
                    INDEX idx_email (email)
                )
            """,

            'chat_sessions': """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id INT NOT NULL,
                    title VARCHAR(200),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id),
                    INDEX idx_created_at (created_at)
                )
            """,

            'chat_messages': """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    user_id INT NOT NULL,
                    message_type ENUM('user', 'assistant', 'system') NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_session_id (session_id),
                    INDEX idx_created_at (created_at)
                )
            """,

            'query_logs': """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(36),
                    query_text TEXT NOT NULL,
                    query_type ENUM('sql', 'natural') NOT NULL,
                    executed_sql TEXT,
                    execution_time FLOAT,
                    row_count INT,
                    status ENUM('success', 'error', 'blocked') NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id),
                    INDEX idx_created_at (created_at),
                    INDEX idx_status (status)
                )
            """,

            'audit_logs': """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    event_type VARCHAR(50) NOT NULL,
                    event_details JSON,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                    INDEX idx_user_id (user_id),
                    INDEX idx_event_type (event_type),
                    INDEX idx_created_at (created_at)
                )
            """,

            # Sample business tables
            'customers': """
                CREATE TABLE IF NOT EXISTS customers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    address TEXT,
                    city VARCHAR(50),
                    country VARCHAR(50),
                    registration_date DATE,
                    status ENUM('active', 'inactive', 'suspended') DEFAULT 'active',
                    INDEX idx_email (email),
                    INDEX idx_city (city),
                    INDEX idx_status (status)
                )
            """,

            'products': """
                CREATE TABLE IF NOT EXISTS products (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    category VARCHAR(50),
                    price DECIMAL(10,2),
                    cost DECIMAL(10,2),
                    stock_quantity INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_category (category),
                    INDEX idx_price (price)
                )
            """,

            'orders': """
                CREATE TABLE IF NOT EXISTS orders (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    customer_id INT NOT NULL,
                    order_date DATE NOT NULL,
                    total_amount DECIMAL(10,2) NOT NULL,
                    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
                    shipping_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE,
                    INDEX idx_customer_id (customer_id),
                    INDEX idx_order_date (order_date),
                    INDEX idx_status (status)
                )
            """,

            'order_items': """
                CREATE TABLE IF NOT EXISTS order_items (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    order_id INT NOT NULL,
                    product_id INT NOT NULL,
                    quantity INT NOT NULL,
                    unit_price DECIMAL(10,2) NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    INDEX idx_order_id (order_id),
                    INDEX idx_product_id (product_id)
                )
            """
        }

        try:
            with self.connection.cursor() as cursor:
                for table_name, table_sql in tables.items():
                    cursor.execute(table_sql)
                    logger.info(f"Table {table_name} created/verified")
                self.connection.commit()
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class VectorStoreSetup:
    """Handle Milvus vector store setup"""

    def __init__(self, config: Config):
        self.config = config
        self.collection_name = "chat_embeddings"

    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.config.MILVUS_HOST,
                port=self.config.MILVUS_PORT
            )
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self):
        """Create embeddings collection"""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="message_id", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # OpenAI embedding dimension
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Chat message embeddings for semantic search"
            )

            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )

            # Create index
            index_params = {
                "metric_type": "IP",  # Inner Product
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            logger.info(f"Collection {self.collection_name} created with index")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise


class ConfigSetup:
    """Handle configuration setup"""

    def __init__(self, config: Config):
        self.config = config
        self.project_root = project_root

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'logs',
            'data/uploads',
            'data/exports',
            'data/cache'
        ]

        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory {directory} created/verified")

    def validate_env_file(self):
        """Validate .env file exists and has required variables"""
        env_file = self.project_root / '.env'

        if not env_file.exists():
            logger.warning(".env file not found, creating template...")
            self.create_env_template()
            return False

        # Check required variables
        required_vars = [
            'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME',
            'MILVUS_HOST', 'MILVUS_PORT',
            'OPENAI_API_KEY',
            'SECRET_KEY'
        ]

        load_dotenv(env_file)
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        logger.info("Environment variables validated")
        return True

    def create_env_template(self):
        """Create .env template file"""
        template = """# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=chatbi

# Vector Database Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Security
SECRET_KEY=your_secret_key_here

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
MAX_QUERY_ROWS=10000
QUERY_TIMEOUT=30
"""

        env_file = self.project_root / '.env'
        with open(env_file, 'w') as f:
            f.write(template)

        logger.info("Created .env template file - please update with your values")


def check_dependencies():
    """Check if required services are available"""
    logger.info("Checking dependencies...")

    # Check Python version
    if sys.version_info < (3, 11):
        logger.error("Python 3.11+ is required")
        return False

    logger.info(f"Python version: {sys.version}")

    # TODO: Add checks for MySQL, Milvus, Redis availability
    # This would involve attempting connections

    return True


async def main():
    """Main setup function"""
    logger.info("Starting ChatBI environment setup...")

    try:
        # Check dependencies
        if not check_dependencies():
            logger.error("Dependency check failed")
            return False

        # Load configuration
        config = Config()

        # Setup configuration
        config_setup = ConfigSetup(config)
        config_setup.create_directories()

        if not config_setup.validate_env_file():
            logger.error("Please update .env file with correct values and run setup again")
            return False

        # Setup database
        logger.info("Setting up database...")
        db_setup = DatabaseSetup(config)
        db_setup.connect()
        db_setup.create_database()
        db_setup.create_tables()
        db_setup.close()

        # Setup vector store
        logger.info("Setting up vector store...")
        vector_setup = VectorStoreSetup(config)
        vector_setup.connect()
        vector_setup.create_collection()

        logger.info("✅ Environment setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Run 'python scripts/seed_data.py' to populate sample data")
        logger.info("2. Run 'python scripts/run.py' to start the application")

        return True

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)