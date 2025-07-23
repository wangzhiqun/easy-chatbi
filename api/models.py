"""
SQLAlchemy models for ChatBI platform.
Defines database table structures and relationships.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .database import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    queries = relationship("QueryHistory", back_populates="user")


class ChatSession(Base):
    """Chat session model to track conversations."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String(200), default="New Chat")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

    # Index
    __table_args__ = (Index("idx_user_session", "user_id", "created_at"),)


class ChatMessage(Base):
    """Individual chat messages within sessions."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    message_type = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    metadata_data = Column(JSON)  # Store additional message metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    # Index
    __table_args__ = (Index("idx_session_time", "session_id", "created_at"),)


class QueryHistory(Base):
    """History of SQL queries generated and executed."""

    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    user_question = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=False)
    execution_status = Column(String(20), default="pending")  # pending, success, error
    execution_time_ms = Column(Integer)
    result_rows = Column(Integer)
    error_message = Column(Text)
    is_safe = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="queries")

    # Index
    __table_args__ = (
        Index("idx_user_query", "user_id", "created_at"),
        Index("idx_status", "execution_status"),
    )


class DataSource(Base):
    """Configuration for different data sources."""

    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    connection_type = Column(String(50), nullable=False)  # mysql, postgresql, etc.
    connection_config = Column(JSON, nullable=False)  # Store connection parameters
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DataTable(Base):
    """Metadata about tables in data sources."""

    __tablename__ = "data_tables"

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    table_name = Column(String(100), nullable=False)
    table_description = Column(Text)
    column_metadata = Column(JSON)  # Store column information
    sample_data = Column(JSON)  # Store sample rows for context
    business_context = Column(Text)  # Business meaning of the table
    is_accessible = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Index
    __table_args__ = (Index("idx_datasource_table", "data_source_id", "table_name"),)


class AuditLog(Base):
    """Audit log for tracking system activities."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action_type = Column(String(50), nullable=False)  # login, query, data_access, etc.
    resource_type = Column(String(50))  # table, query, session, etc.
    resource_id = Column(String(100))
    action_details = Column(JSON)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(String(500))
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Index
    __table_args__ = (
        Index("idx_user_action", "user_id", "action_type", "created_at"),
        Index("idx_resource", "resource_type", "resource_id"),
    )


class SystemConfig(Base):
    """System configuration and settings."""

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(JSON, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())