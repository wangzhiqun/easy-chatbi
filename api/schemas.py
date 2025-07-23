"""
Pydantic schemas for request/response validation in ChatBI platform.
Defines data models for API input/output validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, EmailStr, Field


# User schemas
class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """Schema for user data in database."""
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class User(UserInDB):
    """Public user schema (no sensitive data)."""
    pass


# Authentication schemas
class Token(BaseModel):
    """JWT token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[int] = None


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


# Chat schemas
class ChatMessageBase(BaseModel):
    """Base chat message schema."""
    content: str = Field(..., min_length=1)
    message_type: str = Field(..., pattern="^(user|assistant|system)$")


class ChatMessageCreate(ChatMessageBase):
    """Schema for creating chat messages."""
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(ChatMessageBase):
    """Chat message response schema."""
    id: int
    session_id: int
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    """Schema for creating chat sessions."""
    session_name: Optional[str] = Field("New Chat", max_length=200)


class ChatSessionUpdate(BaseModel):
    """Schema for updating chat sessions."""
    session_name: Optional[str] = Field(None, max_length=200)
    is_active: Optional[bool] = None


class ChatSession(BaseModel):
    """Chat session response schema."""
    id: int
    user_id: int
    session_name: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: Optional[int] = 0

    class Config:
        from_attributes = True


class ChatSessionWithMessages(ChatSession):
    """Chat session with messages."""
    messages: List[ChatMessage] = []


# Query schemas
class QueryRequest(BaseModel):
    """Schema for natural language query requests."""
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[int] = None
    include_chart_suggestion: bool = True


class QueryResponse(BaseModel):
    """Schema for query responses."""
    question: str
    generated_sql: str
    execution_status: str
    execution_time_ms: Optional[int]
    result_data: Optional[List[Dict[str, Any]]] = None
    result_summary: Optional[str] = None
    chart_suggestion: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    is_safe: bool = True


class QueryHistory(BaseModel):
    """Query history response schema."""
    id: int
    user_question: str
    generated_sql: str
    execution_status: str
    execution_time_ms: Optional[int]
    result_rows: Optional[int]
    error_message: Optional[str]
    is_safe: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Data source schemas
class DataSourceCreate(BaseModel):
    """Schema for creating data sources."""
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    connection_type: str = Field(..., max_length=50)
    connection_config: Dict[str, Any]


class DataSource(BaseModel):
    """Data source response schema."""
    id: int
    name: str
    description: Optional[str]
    connection_type: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DataTableSchema(BaseModel):
    """Schema for table metadata."""
    table_name: str
    table_description: Optional[str]
    columns: List[Dict[str, Any]]
    sample_data: Optional[List[Dict[str, Any]]]
    business_context: Optional[str]


# Chart schemas
class ChartConfig(BaseModel):
    """Chart configuration schema."""
    chart_type: str = Field(..., pattern="^(bar|line|pie|scatter|area|histogram)$")
    x_axis: str
    y_axis: Union[str, List[str]]
    title: Optional[str] = None
    color_column: Optional[str] = None
    aggregate_function: Optional[str] = Field(None, pattern="^(sum|count|avg|min|max)$")


class ChartData(BaseModel):
    """Chart data response schema."""
    config: ChartConfig
    data: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


# System schemas
class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    services: Dict[str, bool]
    version: str


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int